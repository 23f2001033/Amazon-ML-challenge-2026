"""Meaningful metric/ownership tests; run with unittest, no model dependencies."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import polars as pl

from audit_confident_pairs import (
    accepted_pairs, audit, entity_metrics, main, repair_oracle, truth_pairs_from_gt,
)
from evaluate import macro_f05


def scored(rows):
    return pl.DataFrame(rows, schema={"s1": pl.String, "mid": pl.String, "p": pl.Float64}, orient="row")


def gt(rows):
    return pl.DataFrame(rows, schema={"source1_entity_id": pl.String,
                                    "matched_entity_ids": pl.String}, orient="row")


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.queries = pl.DataFrame({"s1": ["A", "B", "C", "D"],
                                     "country": ["US", "US", "IN", "IN"],
                                     "region": ["east", "east", "west", "west"]})
        self.truth = gt([("A", "x,y"), ("B", "z"), ("C", ""), ("D", None),
                         ("outside", "orphan")])
        self.stage1 = scored([("A", "x", .995), ("B", "x", .5),
                              ("A", "y", .01), ("B", "z", .99),
                              ("C", "bad", .999), ("A", "orphan", .02)])
        self.final = scored([("A", "x", .8), ("B", "x", .9),
                             ("B", "z", .7), ("C", "bad", .999),
                             ("A", "orphan", .725)])

    def test_baseline_matches_official_metric_and_disjoint_pair_counts(self):
        report = audit(self.stage1, self.final, self.truth, self.queries)
        expected, _ = macro_f05({"A": {"orphan"}, "B": {"x"}, "C": {"bad"}},
                                {"A": {"x", "y"}, "B": {"z"}}, ["A", "B", "C", "D"])
        self.assertEqual(report["baseline"]["macro_f05"], expected)
        self.assertEqual(report["baseline"]["true_positives"], 0)
        self.assertEqual(report["baseline"]["false_positives"], 3)
        self.assertEqual(report["baseline"]["false_negatives"], 3)
        self.assertEqual(report["baseline"]["wrong_owner_records"], 1)
        self.assertEqual(report["baseline"]["accepted_records_without_query_owner"], 2)
        bins = report["stage1_bins"]
        self.assertEqual(bins["below_0.02"]["pairs"], 1)
        self.assertEqual(bins["band_0.02_to_0.99"]["pairs"], 2)
        self.assertEqual(bins["confident_0.99_plus"]["pairs"], 3)
        self.assertEqual(bins["below_0.02"]["final_rows_missing"], 1)

    def test_record_repair_updates_both_owners_and_singletons(self):
        truth_pairs = truth_pairs_from_gt(self.truth, self.queries)
        accepted = accepted_pairs(self.final, .725)
        metrics = entity_metrics(self.queries, truth_pairs, accepted)
        eligible = pl.DataFrame({"mid": ["x", "bad"]})
        report, scores = repair_oracle(metrics, accepted, truth_pairs, eligible, "oracle")
        repaired = {"A": {"x", "orphan"}}
        expected, per_entity = macro_f05(repaired, {"A": {"x", "y"}, "B": {"z"}},
                                          ["A", "B", "C", "D"])
        self.assertAlmostEqual(report["macro_f05"], expected)
        self.assertEqual(report["affected_entities"], 3)
        self.assertEqual(report["changed_records"], 2)
        self.assertEqual(scores.sort("s1")["oracle"].to_list(), per_entity.tolist())

    def test_missing_retrieval_scope_repairs_wrong_owner(self):
        stage1 = self.stage1.filter(~((pl.col("s1") == "A") & (pl.col("mid") == "x")))
        report = audit(stage1, self.final, self.truth, self.queries)
        oracle = report["oracles"]["missing_stage1_true_pair_ceiling"]
        self.assertEqual(report["coverage"]["true_pairs_missing_stage1"], 1)
        self.assertTrue(oracle["eligibility_uses_labels"])
        self.assertEqual(oracle["affected_entities"], 2)
        expected, _ = macro_f05({"A": {"x", "orphan"}, "C": {"bad"}},
                                {"A": {"x", "y"}, "B": {"z"}}, ["A", "B", "C", "D"])
        self.assertAlmostEqual(oracle["macro_f05"], expected)

    def test_bin_scope_is_label_free_overlapping_and_can_recover_missing_pair(self):
        report = audit(self.stage1, self.final, self.truth, self.queries)
        high = report["oracles"]["confident_0.99_plus"]
        band = report["oracles"]["band_0.02_to_0.99"]
        self.assertFalse(high["eligibility_uses_labels"])
        self.assertFalse(band["eligibility_uses_labels"])
        self.assertEqual(high["eligible_records"], 3)
        self.assertEqual(band["eligible_records"], 2)  # x belongs to both scopes.
        new_truth = gt([("A", "y"), ("B", "z"), ("C", None), ("D", "x")])
        changed = audit(self.stage1, self.final, new_truth, self.queries)
        self.assertEqual(changed["oracles"]["confident_0.99_plus"]["eligible_records"], 3)
        self.assertEqual(changed["oracles"]["confident_0.99_plus"]["eligible_true_pairs_missing_stage1"], 1)

    def test_tie_break_threshold_and_outside_query_filtering(self):
        final = scored([("B", "x", .8), ("A", "x", .8), ("outside", "x", .999),
                        ("B", "z", .725)])
        report = audit(self.stage1, final, self.truth, self.queries)
        self.assertEqual(report["baseline"]["true_positives"], 2)
        self.assertEqual(report["coverage"]["final"]["rows_outside_query_universe"], 1)

    def test_float32_boundaries_match_pipeline_comparisons(self):
        pairs = scored([("A", "x", .02)]).with_columns(pl.col("p").cast(pl.Float32))
        report = audit(pairs, pairs, gt([("A", "x")]),
                       pl.DataFrame({"s1": ["A"]}), threshold=.02)
        self.assertEqual(report["stage1_bins"]["band_0.02_to_0.99"]["pairs"], 1)
        self.assertEqual(report["stage1_bins"]["below_0.02"]["pairs"], 0)
        self.assertEqual(report["baseline"]["true_positives"], 1)

    def test_query_flags_exclude_dropped_owners_and_multi_universe_requires_selection(self):
        queries = self.queries.with_columns(pl.Series("is_query", [True, False, True, True]))
        report = audit(self.stage1, self.final, self.truth, queries)
        self.assertEqual(report["baseline"]["queries"], 3)
        self.assertEqual(report["baseline"]["true_pairs"], 2)
        self.assertEqual(report["baseline"]["true_positives"], 1)  # B's higher claim to x is excluded.
        self.assertTrue(report["query_selection"]["is_query_filter_applied"])
        queries = queries.with_columns(pl.Series("universe", ["val", "train", "val", "val"]))
        with self.assertRaisesRegex(ValueError, "multiple universes"):
            audit(self.stage1, self.final, self.truth, queries)
        selected = audit(self.stage1, self.final, self.truth, queries, universe="val")
        self.assertEqual(selected["baseline"], report["baseline"])
        strings = queries.with_columns(pl.col("is_query").cast(pl.String))
        self.assertEqual(audit(self.stage1, self.final, self.truth, strings, universe="val")["baseline"],
                         report["baseline"])

    def test_missing_truth_duplicate_ids_and_invalid_probabilities_fail(self):
        with self.assertRaisesRegex(ValueError, "no row"):
            audit(self.stage1, self.final, self.truth.filter(pl.col("source1_entity_id") != "D"), self.queries)
        with self.assertRaisesRegex(ValueError, "duplicate S1"):
            audit(self.stage1, self.final, self.truth, pl.concat([self.queries, self.queries.head(1)]))
        with self.assertRaisesRegex(ValueError, "multiple query"):
            audit(self.stage1, self.final, gt([("A", "x"), ("B", "x"), ("C", ""), ("D", "")]), self.queries)
        with self.assertRaisesRegex(ValueError, "duplicate .* pairs"):
            audit(pl.concat([self.stage1, self.stage1.head(1)]), self.final, self.truth, self.queries)
        for probability in (float("nan"), float("inf"), -0.1, 1.1, None):
            with self.subTest(probability=probability), self.assertRaisesRegex(ValueError, "probabilities"):
                audit(scored([("A", "x", probability)]), self.final, self.truth, self.queries)

    def test_empty_candidates_preserve_all_queries_and_singletons(self):
        report = audit(scored([]), scored([]), self.truth, self.queries)
        self.assertEqual(report["baseline"]["macro_f05"], .5)
        self.assertEqual(report["coverage"]["true_pairs_missing_stage1"], 3)
        self.assertEqual(report["oracles"]["confident_0.99_plus"]["delta_macro_f05"], 0)
        self.assertEqual(report["oracles"]["missing_stage1_true_pair_ceiling"]["macro_f05"], 1)
        singleton = audit(scored([]), scored([]), gt([("A", "")]), pl.DataFrame({"s1": ["A"]}))
        self.assertIsNone(singleton["coverage"]["stage1_true_pair_coverage"])
        self.assertEqual(singleton["baseline"]["macro_f05"], 1)

    def test_cli_roundtrip_and_input_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: Path(directory) / (name + ".parquet")
                     for name in ("stage1", "final", "truth", "queries")}
            for name, data in (("stage1", self.stage1), ("final", self.final),
                               ("truth", self.truth), ("queries", self.queries.rename({"s1": "entity_id"}))):
                data.write_parquet(paths[name])
            args = [part for name, path in paths.items() for part in ("--" + name, str(path))]
            output = Path(directory) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                main(args + ["--out", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report["baseline"]["queries"], 4)
            self.assertEqual(len(report["by_country"]), 2)
            self.assertEqual(len(report["by_region"]), 2)
            before = paths["stage1"].read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--out", str(paths["stage1"])])
            self.assertEqual(paths["stage1"].read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
