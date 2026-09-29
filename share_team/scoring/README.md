# Scoring and error-analysis code

This is the code behind the numbers in the team chat. It covers the metric, the recall breakdown, the oracle error budget and the France size-bias test.

## The metric
`evaluate.py`: `macro_f05(pred_map, truth_map, s1_ids)` is the official macro F0.5.
- F0.5 is computed per S1, then averaged.
- A singleton scores 1 if predicted empty, else 0.
- `truth_map_from_gt(gt, s1_ids)` builds `{s1: set(ids)}` from `train_ground_truth.tsv`.

`model.py`: `assign(df, thr, thr_by_country=None)` is our decision rule.
- Each S2/S3 record goes to at most one S1: the one with the highest p.
- It is kept only if p ≥ threshold.
- `df` has the columns `s1, mid, p` (plus `country` if you pass per-country thresholds).

## Held-out sets to score on
Use `../val_universe.parquet` (8% of regions) and `../wide_universe.parquet` (20 more regions).
- Columns: `entity_id, universe, is_query`.
- Score **only the `is_query` S1s**. The other 19% of S1s in those regions were dropped on purpose, so their records act as orphans, like on test.
- Candidates come only from those regions' records. Never train on these regions.

## Recall breakdown ("~70% scored below threshold, ~30% blocking misses")
`error_analysis.py` and `blocking_misses.py` split every missed true pair into:
1. not a candidate (blocking miss);
2. pruned by stage 1 (p < 0.02);
3. scored below threshold;
4. taken by another S1.

Each split is further broken down by address presence and name ambiguity. `blocking_misses.py` also prints examples: Indian truncated addresses such as "Office No. 815, Thane, MH", and the "Washington, AZ" region-parsing bug.

## Oracle error budget
`error_budget.py`: for each error class it fixes only that class using the truth and recomputes macro F0.5. That gives the most the class can ever be worth.

The scoring is done in test-like conditions: orphan records are subsampled to the test density (US about 9%, India about 2%).

| Class (val+wide, US/India) | Ceiling |
|---|---|
| Blocking misses (with address / empty address) | +0.0018 / +0.0012 |
| Ambiguous empty-address records (name shared by 2+ S1s) | +0.0016 |
| False positives: orphans / distractors | +0.0009 / +0.0008 |

## France: label-free size-bias test
`france_sizebias_test.py` takes a set of accepted French pairs and estimates the share that are distractors, without labels.
- A **true copy** makes its S1's per-source record count *size-biased*: more S1s with exactly one record, almost none above the generator's maximum.
- An **extra distractor** adds 1 on top of the normal count.

The script compares the observed count distribution against both predictions.

Result: the 20,153 French "category-swap" pairs (e.g. "Pornic Club SAS" → "Pornic Sportive SAS" at the same address) match the distractor model (L1 distance 0.035 vs 0.135), so about 90% of them are false merges.

Note: the paths inside the scripts point to our AWS layout (`/home/ubuntu/er/data/...`). Change `F`, `P` and `Q` at the top to your own prepared, parquet and prediction files.
