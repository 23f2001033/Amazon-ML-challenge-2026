# Cross-encoder task: uncertain pairs only

**Goal:** a fine-tuned cross-encoder that re-scores the ~1 pair per S1 our pipeline is unsure about. Its score becomes one extra feature in our stage-2 LightGBM. If every one of these pairs were decided correctly, validation would go from **0.981 to 0.994**.

## File: `xenc_val_band_v2a.parquet` (193,698 labelled pairs, 65,227 positive)
| Column | Meaning |
|---|---|
| `s1`, `mid` | S1 id, candidate S2/S3 id |
| `y` | 1 = true match, 0 = not |
| `p_stage1` | our current model's probability (all rows in [0.02, 0.99), i.e. the uncertain band) |
| `country` | US / India (test also has France) |
| `s1_name`, `s1_address`, `c_name`, `c_address` | raw text |
| `s1_name_canon`, `s1_addr_canon`, `c_name_canon`, `c_addr_canon` | our cleaned text: Indic scripts translated, accents folded, legal forms canonical, abbreviations normalised, empty = missing address |

## What to build
1. **Input:** use the **canonical** fields, e.g.
   `name: {s1_name_canon} | addr: {s1_addr_canon}` `[SEP]` `name: {c_name_canon} | addr: {c_addr_canon}`.
   Keep numbers as they are. The key signals are house numbers (515 vs 516 = decoy) and small name mutations (Pranva vs Pranoro = decoy; Ca1lahan vs Callahan = noise).
2. **Model:** a license-safe cross-encoder (MIT/Apache, far under 8B), e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0). Check the license on the model card. **Fine-tune it**; zero-shot scores are not meaningful here.
3. **Honest evaluation (important):**
   - Split **by S1 id** into 2 folds (e.g. `hash(s1) % 2`). Train on one fold, predict the other, then swap.
   - Report AUC and log-loss **vs `p_stage1`** on the same rows. It only helps if it adds information beyond `p_stage1`.
4. **Return:** a parquet with `s1, mid, p_xenc` for **all 193,698 rows** (out-of-fold predictions).
5. **Later (test):** I'll send `xenc_test_band_v2a.parquet` (same columns, no `y`, includes France). Train on all val rows and return `s1, mid, p_xenc` for test.

Hardware: a Kaggle T4 is enough. About 200k pairs × 2–3 epochs of MiniLM-L6 takes roughly 20–40 minutes.
