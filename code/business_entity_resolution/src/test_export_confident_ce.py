"""Run: python -m unittest discover -s code/business_entity_resolution/src -p test_export_confident_ce.py"""
from pathlib import Path
import tempfile
import unittest

import polars as pl

from export_confident_ce import export_pairs, merge_scores


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage1 = self.write("stage.parquet", pl.DataFrame({
            "s1": ["a", "a", "b", "a"], "mid": ["x", "y", "z", "w"],
            "p": [0.999, 0.02, 0.7, 0.019],
        }))
        self.s1 = self.write("s1.parquet", pl.DataFrame({
            "entity_id": ["a", "b"], "business_name": ["Café", "Source B"],
            "business_address": [None, "Road"], "country": ["France", "India"],
        }))
        self.s2 = self.write("s2.parquet", pl.DataFrame({
            "entity_id": ["x", "z"], "business_name": [None, "Pool Z"],
            "business_address": [None, "Road"],
        }))
        self.s3 = self.write("s3.parquet", pl.DataFrame({
            "entity_id": ["y", "w"], "business_name": ["Café", "W"],
            "business_address": ["Rue", "Street"],
        }))

    def write(self, name, frame):
        path = self.root / name
        frame.write_parquet(path)
        return path

    def export(self, **kwargs):
        return export_pairs(self.stage1, self.s1, self.s2, self.s3,
                            self.root / "export", **kwargs)

    def scores(self, mids, probs=None, logits=None):
        frame = pl.DataFrame({"s1": ["a"] * len(mids), "mid": mids,
                              "p_ce": probs if probs is not None else [0.7] * len(mids)})
        return frame.with_columns(pl.Series("ce_logit", logits)) if logits is not None else frame

    def test_complete_french_set_keeps_confident_and_threshold_pairs(self):
        stats = self.export(country="france", shard_size=1)
        self.assertEqual((stats["retained_pairs"], stats["confident_pairs"]), (2, 1))
        self.assertEqual(len(stats["shards"]), 2)
        manifest = pl.read_parquet(self.root / "export/manifest.parquet")
        self.assertEqual(manifest["mid"].to_list(), ["x", "y"])
        first = pl.read_parquet(self.root / "export/pairs_00000.parquet")
        self.assertEqual(first["text_a"][0], "Café | ")
        self.assertEqual(first["text_b"][0], " | ")

    def test_float32_boundaries_match_original_stage1_comparisons(self):
        stage = pl.read_parquet(self.stage1).with_columns(
            pl.Series("p", [0.99, 0.02, 0.019999, 0.989999], dtype=pl.Float32))
        self.write("stage.parquet", stage)
        stats = self.export()
        manifest = pl.read_parquet(self.root / "export/manifest.parquet")
        self.assertEqual(manifest.schema["p"], pl.Float32)
        self.assertEqual(manifest["mid"].to_list(), ["w", "x", "y"])
        self.assertEqual(stats["confident_pairs"], 1)
        self.assertEqual(stats["retained_pairs"], stage.filter(pl.col("p") >= 0.02).height)

    def test_reuse_preserves_full_manifest_and_missing_raw_logits(self):
        existing = self.write("existing.parquet", self.scores(["y"]))
        stats = self.export(country="France", existing_ce=[existing])
        self.assertEqual((stats["retained_pairs"], stats["inference_pairs"]), (2, 1))
        reused = pl.read_parquet(self.root / "export/reused_ce.parquet")
        self.assertEqual(reused["ce_logit"].null_count(), 1)
        new = self.write("new.parquet", self.scores(["x"], [0.8], [1.386294]))
        merged = merge_scores(self.root / "export/manifest.parquet",
                              [self.root / "export/reused_ce.parquet", new],
                              self.root / "merged.parquet")
        self.assertEqual(merged.height, 2)
        self.assertEqual(merged["ce_logit"].null_count(), 1)

    def test_missing_raw_id_fails_even_if_score_is_reused(self):
        self.write("s3.parquet", pl.read_parquet(self.s3).filter(pl.col("entity_id") != "y"))
        existing = self.write("existing.parquet", self.scores(["y"]))
        with self.assertRaisesRegex(ValueError, "Missing raw mid"):
            self.export(country="France", existing_ce=[existing])

    def test_duplicate_stage1_fails(self):
        stage = pl.read_parquet(self.stage1)
        self.write("stage.parquet", pl.concat([stage, stage.head(1)]))
        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            self.export()

    def test_empty_and_whitespace_stage1_ids_fail(self):
        stage = pl.read_parquet(self.stage1)
        for invalid in ("", "   "):
            with self.subTest(invalid=repr(invalid)):
                self.write("stage.parquet", stage.with_columns(
                    pl.when(pl.col("mid") == "x").then(pl.lit(invalid))
                    .otherwise(pl.col("mid")).alias("mid")))
                with self.assertRaisesRegex(ValueError, "empty pair IDs"):
                    self.export()

    def test_duplicate_raw_pool_id_fails(self):
        pool = pl.read_parquet(self.s2)
        self.write("s2.parquet", pl.concat([pool, pool.head(1)]))
        with self.assertRaisesRegex(ValueError, "duplicate selected IDs"):
            self.export()

    def test_bad_existing_probability_fails(self):
        existing = self.write("existing.parquet", self.scores(["x"], [float("nan")]))
        with self.assertRaisesRegex(ValueError, "invalid probabilities"):
            self.export(existing_ce=[existing])

    def test_inconsistent_probability_and_raw_logit_fail(self):
        existing = self.write("existing.parquet", self.scores(["x"], [0.9], [-5.0]))
        with self.assertRaisesRegex(ValueError, "disagree with sigmoid"):
            self.export(existing_ce=[existing])

    def test_empty_existing_score_ids_fail(self):
        existing = self.write("existing.parquet", self.scores([""]))
        with self.assertRaisesRegex(ValueError, "empty pair IDs"):
            self.export(existing_ce=[existing])

    def test_merge_rejects_equal_count_wrong_coverage(self):
        self.export(country="France")
        wrong = self.write("wrong.parquet", self.scores(["x", "not-y"]))
        with self.assertRaisesRegex(ValueError, "1 missing and 1 extra"):
            merge_scores(self.root / "export/manifest.parquet", [wrong], self.root / "merged.parquet")

    def test_merge_rejects_duplicate_pairs(self):
        self.export(country="France")
        scores = self.write("scores.parquet", self.scores(["x", "y"]))
        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            merge_scores(self.root / "export/manifest.parquet", [scores, scores], self.root / "merged.parquet")

    def test_no_matching_country_fails(self):
        with self.assertRaisesRegex(ValueError, "Country selection"):
            self.export(country="FR")

    def test_export_refuses_to_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_empty_retained_set_can_merge(self):
        stats = self.export(min_p=1.0)
        self.assertEqual(stats["shards"], [])
        merged = merge_scores(self.root / "export/manifest.parquet",
                              [self.root / "export/reused_ce.parquet"], self.root / "empty.parquet")
        self.assertEqual(merged.height, 0)


class LocalInferenceTest(unittest.TestCase):
    def test_saved_tiny_checkpoint_cpu_outputs_actual_logits_and_exact_coverage(self):
        # Random tiny model created locally: exercises saved-tokenizer loading,
        # ragged batches, output coverage and sigmoid semantics without network.
        import math
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
        from infer_confident_ce import infer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model"
            tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2,
                                             "b": 3, "|": 4}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            saved_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                                      unk_token="[UNK]", pad_token="[PAD]")
            saved_tokenizer.save_pretrained(checkpoint)
            torch.manual_seed(123)
            BertForSequenceClassification(BertConfig(vocab_size=5, hidden_size=16,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=20,
                num_labels=1)).save_pretrained(checkpoint)
            pairs = pl.DataFrame({"s1": ["a", "a", "b"], "mid": ["x", "y", "z"],
                                  "text_a": ["a |", "a |", "b |"],
                                  "text_b": ["a", "b", "a b"]})
            pairs.write_parquet(root / "pairs.parquet")
            result = infer(root / "pairs.parquet", checkpoint, root / "scores.parquet",
                           batch_size=2, device="cpu")
            self.assertTrue(result.select("s1", "mid").equals(pairs.select("s1", "mid")))
            for p, logit in zip(result["p_ce"], result["ce_logit"]):
                self.assertAlmostEqual(p, 1 / (1 + math.exp(-logit)), places=6)
            self.assertTrue(pl.read_parquet(root / "scores.parquet").equals(result))
            self.assertFalse((root / "scores.parquet.partial").exists())


if __name__ == "__main__":
    unittest.main()
