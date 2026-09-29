# Team protocol: benchmarking on our validation, and plugging a model into our stage 2

Updated 26 Sep, 13:00. Our pipeline scores 0.981 on validation and 0.968 on the leaderboard.

## Files
| File | Rows | What |
|---|---|---|
| `val_universe.parquet` | 242,936 | our **validation universe**: `entity_id, country, region, is_query` |
| `wide_universe.parquet` | 782,734 | **hard check set**: 20 more held-out regions (Maharashtra, Tamil Nadu, Telangana, UP + 16 US states), same columns |
| `xenc_val_band_v3b.parquet` | 210,838 | uncertain **validation** pairs (our p in [0.02, 0.99)) **with labels `y`** |
| `xenc_test_band_v3b.parquet` | 2,083,327 | uncertain **test** pairs (no labels; includes France) |

Band columns: `s1, mid, p_stage1, [y], country, s1_name, s1_address, s1_name_canon, s1_addr_canon, c_name, c_address, c_name_canon, c_addr_canon`. The `_canon` fields are our cleaned text: Indic scripts translated, accents folded, legal forms canonical, empty = missing address.

## A. Benchmark your full pipeline on our validation (comparable to our numbers)
1. **Queries** = rows with `is_query = true` (196,781 in val; 633,561 in wide). The other 19% of S1s in these regions are deliberately *not* queried, but **their S2/S3 records stay in the pool** as orphan distractors. That reproduces test's distractor density.
2. **Pool** = **all** `train_s2` + `train_s3` records. Not only the queries' matches.
3. **Metric** = macro F0.5 over the query S1s. It is computed per S1 and then averaged. A singleton scores 1 if you predict nothing, else 0. An S1 with matches scores 0 if you predict nothing. Each S2/S3 record goes to at most one S1.
4. **Train only on S1s outside both universes**, and exclude their matches too.

Our reference numbers (stage 1 v3b, threshold 0.75):

| Set | Overall | US | India |
|---|---|---|---|
| val | 0.9788 | 0.9815 | 0.9746 |
| wide | **0.9698** | 0.9812 | **0.9579** (Telangana 0.876, Maharashtra 0.968) |

The wide set predicts our leaderboard (0.968) much better than val does.

## B. Plug your model into our stage 2
1. Score **every row** of both band files.
2. Return `band_val_<yourname>.parquet` (210,838 rows) and `band_test_<yourname>.parquet` (2,083,327 rows), with columns `s1, mid, p_<yourname>`.
3. **Self-check:** on the val band, compare the AUC and log-loss of your `p` against `p_stage1` on the same rows.
4. License: MIT/Apache models only.

## Lessons from our side (26 Sep)
- **India, region mix-ups (biggest find):**
  - S2/S3 write Hyderabad and other Telangana addresses with the old state name **"Andhra Pradesh"**. That's 18.5% of Telangana's true pairs.
  - Region-scoped blocking misses all of them, so Telangana scores 0.876.
  - Similar: **Washington DC ↔ "WA"** (77%), and Chandigarh → Punjab.
  - **Fix:** learn from train labels which regions leak into which, and search those regions too. Telangana blocking recall goes 0.80 → 0.99.
- **Big regions crowd out matches.** In Maharashtra, a name like "Classic Projects" appears ~35 times in the pool, and many S2/S3 addresses are truncated ("401 mumbai mumbai"). Top-25 by name alone drops true pairs. **Rank on name + address combined:** recall 0.963 → 0.985.
- **Count and competition features:** removing them all *lowered* our leaderboard (0.9661 vs 0.9681). Keep them, but compute them the way test computes them: same batch sizes, and ambiguity counts on an S1 set of test's size.
- **France (15% of test, no labels):** names are generic, e.g. 530 distinct "Bordeaux Club SAS/SARL/EURL" in one city. Address and legal form carry the identity.
- The metric is per-S1 macro F0.5 including singletons. Micro F0.5 reads ~0.006 higher.
