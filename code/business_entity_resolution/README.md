# Business Entity Resolution: reproduction guide (team Embedding)

**Final submission:** `v6fr3zF2`, public leaderboard **0.988574**. It is `output/matching_results.tsv`, plus `output/candidate_pairs.tsv`, the pruned candidate set that every match comes from.

## 0. Pipeline in one view
1. **Canonicalisation** (`prepare.py`, `textnorm.py`, `resources.py`): inverts the observed noise. Dictionaries are learned from train only.
2. **Blocking** (`blocking.py`): multi-pass, region-scoped TF-IDF search with a name pass, an address pass, a combined pass, a region-less fallback and learned region-leak expansion. About 69 candidates per S1.
3. **Learned pruning, two stage-1 LightGBM views** (`features.py`, `model.py`):
   - `lgbm_v6`, and `lgbm_v6t`, which is trained with test-faithful sample weights;
   - pairs with p ≥ 0.02 in either view are kept. The union, **4.51 pairs per S1** (7,814,413 pairs), is `candidate_pairs.tsv`.
4. **Cross-encoders** (GPU): CE v1 (Multilingual-MiniLM), CE v2 (XLM-R base), CE 3-FR (bge-reranker-base) and CE 4-FR (bge-reranker-v2-m3) score the pruned pairs.
5. **Stage 2** (`stage2.py`, `s2_tune.py`): LightGBM re-scores the uncertain band (0.02 ≤ p < 0.99) with competition, sibling and cross-encoder features.
   - US/India: mean of stage 2 'w2' on both stage-1 views (`blend_preds.py`).
   - France: stage 2 'w' on v6.
6. **Decision layer** (`final_decide.py`, config `models/v6fr3zF2_final.json`):
   - partition assignment (each record goes to at most one S1);
   - US/India thresholds per country and per S1 candidate-count bucket; France threshold 0.50;
   - France category-swap veto (and its re-application to the final decisions);
   - acceptance of French drop-word copies and of acronym copies (all countries);
   - empty-S1 rescue.

Everything is learned from the provided challenge data. There are no external data, APIs or lookups. The only downloads are the four public pretrained checkpoints (MIT/Apache-2.0, at most 568M parameters).

## 1. Setup
```bash
pip install -r requirements.txt          # CPU pipeline (Python 3.11)
pip install -r requirements-gpu.txt      # cross-encoder steps only (GPU)
export PYTHONIOENCODING=utf-8            # the data contains Indic scripts
```
- **Data.** A dataset folder with `train/` and `test/` holding the original TSVs. It is found automatically in `<root>/dataset/`, `<root>/student_resource/dataset/` or `<root>/6ab10eb3b23ba_student_resource/student_resource/dataset/`. Otherwise set `ER_RAW_DIR=/path/to/dataset`.
- **`<root>`** is the folder that contains `code/` (set `ER_ROOT` to change it). Intermediate files go to `<root>/data/` and results to `<root>/output/`.
- **Hardware.** Every CPU step runs within a few hours on 16 vCPU / 64 GB. The cross-encoders were trained and run on Kaggle T4 x2 GPUs.
- Run all commands from `code/business_entity_resolution/src`.

## 2. Folder contents
- `src/`: all source code. Section 6 lists which scripts form the pipeline and which are analyses.
- `models/`, the exact artefacts behind the final submission:

| File | What it is |
|---|---|
| `lgbm_v6.txt`, `v6_config.json` | stage-1 view v6 and its config (feature list, validation results) |
| `lgbm_v6t.txt`, `v6t_config.json` | stage-1 view v6t (test-faithful training weights) |
| `region_leaks.json` | region-leak pairs learned from train labels (`region_leaks.py`) |
| `s2_v6_rich_ce_ce2_ce3fr_ce4fr_w.txt` | stage 2 'w' on v6 (orphan-like pairs down-weighted to test density), used for France |
| `s2_v6_rich_ce_ce2_ce3fr_ce4fr_w2.txt` | stage 2 'w2' on v6 (w + records owned outside the universe at 0.05), US/India |
| `s2_v6t_rich_ce_ce2_ce3fr_ce4fr_w2.txt` (+ `.json`) | stage 2 'w2' on v6t, US/India |
| `s2tune_w.json`, `s2tune_w2.json`, `s2tune_w2_v6t.json` | boosting-round counts from the out-of-fold runs (needed only to retrain stage 2) |
| **`v6fr3zF2_final.json`** | **decision config of the final submission** |
| other `*_final.json` / `*_config.json` | decision configs of earlier leaderboard submissions (history; see `Documentation_template.md` §5) |

## 3. Reproduce the submitted outputs
With the shipped models and the same cross-encoder scores, the pipeline regenerates the submitted files.
- **Verified:** `final_decide.py` regenerated each decision-layer upload byte-for-byte from the stage-2 probabilities. The final file's md5 is `83a25eaa473c18cceec1f48084ef4acf`.
- **Scoring steps:** stage-1 and stage-2 scoring (`run_test.py`, `s2_tune.py` with `ER_S2_LOAD`) use the shipped LightGBM boosters. They do not retrain.

```bash
cd code/business_entity_resolution/src
export PYTHONIOENCODING=utf-8 ER_WORKERS=8 ER_UNIVERSES=big ER_BATCH_S1=100000 ER_COMB=1 ER_LEAKS=1 ER_SAVE_BAND=1 ER_TESTLIKE_AMB=1 \
       ER_DROP_FEATS=pool_cnt_c,pool_cnt_s,rare_tok_idf,nm_idf_total1,nm_extra_tok_idf
```

### 3.1 Canonicalisation, test blocking and both stage-1 views (CPU)
```bash
python prepare.py                                   # raw TSV -> canonicalised parquet (+ dictionaries learned from train)
mkdir -p ../../../data/artifacts && cp ../models/region_leaks.json ../../../data/artifacts/   # or: python region_leaks.py
ER_TAG=v6  python run_test.py ../models/lgbm_v6.txt  ../models/v6_config.json  ../../../output_v6  v6
ER_TAG=v6t python run_test.py ../models/lgbm_v6t.txt ../models/v6t_config.json ../../../output_v6t v6t
```
Output: `data/features/test_pred_{v6,v6t}.parquet`, the stage-1 probabilities of all pairs with p ≥ 0.02, and `data/features/test_band_{v6,v6t}/`, the stage-1 features of the uncertain band.

### 3.2 Cross-encoder scores (GPU; the only step that is not bit-exact)
Stage 2 reads the scores from `<CE> = <root>/data/ce` (override with `ER_CE_DIR`), laid out as follows. Every file has columns `s1, mid, p_ce`.

| Directory | Files | Model, script | Pairs scored |
|---|---|---|---|
| `<CE>/v6/` | `{val,wide,test}_ce.parquet` | CE v1, `ce_train_infer.py` (+ `export_ce_v6.py` for new pairs) | every v6 pair with p ≥ 0.02 |
| `<CE>/ce2v6/` | `band_{val,wide,test}_ce.parquet` | CE v2, `ce2_train_infer.py` | v6 uncertain band |
| `<CE>/ce3frv6/` | `band_{val,wide,test}_ce3fr.parquet` | CE 3-FR, `ce_fr_train_infer.py` (from the CE 3 checkpoint, `ce3_train_infer.py`) | v6 uncertain band |
| `<CE>/ce4frv6/` | `band_{val,wide,test}_ce4fr.parquet` | CE 4-FR, `ce_fr_train_infer.py` | v6 uncertain band |

- **Training data for the cross-encoders:** `export_ce_data.py`, `export_ce2_train.py`, `export_ce_band.py`, `export_ce_raw.py` and `make_fr_synth.py`. The French pairs are built only from the (unlabelled) test France S1 records. Each `*_train_infer.py` docstring gives the exact call.
- **Pairs that only v6t keeps** (643,622 test pairs) are scored with the *same* four checkpoints and merged into `*_v6t` directories:
```bash
python export_ce_new.py v6t <pairs_dir> test               # canon_test.parquet + raw_test.parquet (pairs lacking a score)
# GPU: ER_CE_IN=<dir with pairs_dir and the four model folders> ER_CE_OUT=<scores_dir> python score_new_kaggle.py
python export_ce_new.py merge v6t <scores_dir>             # -> <CE>/v6_v6t, ce2v6_v6t, ce3frv6_v6t, ce4frv6_v6t
```
(For stage-2 retraining, run the same three commands with `val,wide` as well.)

### 3.3 Stage 2 with the shipped models (CPU, exact)
`ER_S2_LOAD` makes `s2_tune.py final` load the shipped booster instead of retraining it, then score the test band.
```bash
CE=../../../data/ce
export ER_S2_RICH=1 ER_S2_ROUNDS=5000 ER_S2_LOAD=../models
# stage 2 on v6: 'w' (France) and 'w2' (US/India)
ER_TAG=v6 ER_S1TAG=v6 ER_S2_CE=$CE/v6 \
  ER_S2_CE_BAND="ce2=$CE/ce2v6/band_{kind}_ce.parquet;ce3fr=$CE/ce3frv6/band_{kind}_ce3fr.parquet;ce4fr=$CE/ce4frv6/band_{kind}_ce4fr.parquet" \
  sh -c 'python s2_tune.py final w && python s2_tune.py final w2'
# stage 2 on v6t: 'w2' (US/India)
ER_TAG=v6t ER_S1TAG=v6t ER_S2_CE=$CE/v6_v6t \
  ER_S2_CE_BAND="ce2=$CE/ce2v6_v6t/band_{kind}_ce.parquet;ce3fr=$CE/ce3frv6_v6t/band_{kind}_ce3fr.parquet;ce4fr=$CE/ce4frv6_v6t/band_{kind}_ce4fr.parquet" \
  python s2_tune.py final w2
```
Output: `data/features/test_pred_s2_v6_rich_ce_ce2_ce3fr_ce4fr_{w,w2}.parquet` and `test_pred_s2_v6t_rich_ce_ce2_ce3fr_ce4fr_w2.parquet`.

### 3.4 Blend and decision layer (CPU, exact)
```bash
python blend_preds.py s2_blend_v6v6t_frv6 s2_v6_rich_ce_ce2_ce3fr_ce4fr_w2 s2_v6t_rich_ce_ce2_ce3fr_ce4fr_w2 s2_v6_rich_ce_ce2_ce3fr_ce4fr_w
python final_decide.py ../models/v6fr3zF2_final.json ../../../output
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- **Blend:** US/India get the mean of the two w2 predictions (the available one where only one view keeps a pair); France gets v6 'w'.
- **Decision config keys** (all documented in `final_decide.py`): `thr_by_country`, `thr_by_bucket`, `swap_veto`, `swap_veto_final`, `add_drop_noise`, `add_acronym`, `empty_rescue`.

### 3.5 Validate
```bash
python <dataset_parent>/utils/validate_submission.py --matching ../../../output/matching_results.tsv \
       --candidate ../../../output/candidate_pairs.tsv --test-dir <dataset>/test --check-ids
```

## 4. Full retrain from scratch (optional)
```bash
python prepare.py && python region_leaks.py
# held-out design: val (8% of regions) and wide (20 regions); 45 other regions train stage 1
ER_TAG=v6 ER_PRESAMPLE=1 ER_BLOCK_UNIVERSES=val,wide,train python run_dev.py block
ER_TAG=v6 ER_PRESAMPLE=1 python run_dev.py features
ER_TAG=v6 ER_PRESAMPLE=1 python run_dev.py train          # -> data/artifacts/lgbm_v6.txt, result_v6.json (= v6_config.json)
python eval_regions.py v6 0.25 0.75                         # wide held-out check
# v6t: same candidates and training features as v6, test-faithful sample weights
ln -sfn dev_train_v6 ../../../data/features/dev_train_v6t
ln -sfn dev_val_v6.parquet  ../../../data/candidates/dev_val_v6t.parquet
ln -sfn dev_wide_v6.parquet ../../../data/candidates/dev_wide_v6t.parquet
ER_TAG=v6t ER_PRESAMPLE=1 ER_S1_TESTW=1 python run_dev.py train && python eval_regions.py v6t 0.25 0.75
```
- **Cross-encoders:** section 3.2 (training + scoring on val, wide and test).
- **Stage 2:** out-of-fold runs over val + wide (4 region folds, test-like evaluation), then the final refit and test scoring.
```bash
ER_S2_ROUNDS=5000 ER_EVAL_B=1 ... (ER_TAG/ER_S1TAG/ER_S2_CE/ER_S2_CE_BAND as in 3.3, without ER_S2_LOAD)
ER_CFGS=w python s2_tune.py dev ; python s2_tune.py final w
ER_OUT_W=0.05 ER_CFGS=w2 python s2_tune.py dev ; ER_OUT_W=0.05 python s2_tune.py final w2      # for v6 and for v6t
```
- **Threshold studies behind the decision config:**
  - `blend_eval.py`: v6 vs v6t vs blend, test-like;
  - `bucket_thr.py`: per-bucket thresholds, cross-fitted val ↔ wide;
  - `loco.py`: leave-one-country-out stage 2;
  - `fr_cells2.py`: France-vs-US/India pattern cells (these need `ER_SAVE_OOF=1` out-of-fold files).

## 5. Determinism
- **Scoring** uses the shipped boosters, and LightGBM prediction and the decision layer are deterministic given their inputs.
- **Retraining** LightGBM can differ slightly with the thread count. Section 3.3 therefore loads the shipped boosters instead.
- GPU fine-tuning is not bit-for-bit reproducible, so retrained cross-encoders reproduce the scores closely but not exactly. The exact score files used for the submission are available from the team on request.

## 6. Script index
- **Pipeline:**
  - data: `config.py`, `prepare.py`, `textnorm.py`, `resources.py`, `region_leaks.py`;
  - blocking and stage 1: `blocking.py`, `features.py`, `model.py`, `run_dev.py`, `eval_regions.py`, `run_test.py`;
  - cross-encoders: `export_ce_*.py`, `make_fr_synth.py`, `ce*_train_infer.py`, `score_new_kaggle.py`;
  - stage 2 and decision: `stage2.py`, `run_stage2.py`, `s2_tune.py`, `blend_preds.py`, `final_decide.py`, `merge_preds.py`;
  - metric: `evaluate.py`.
- **Analyses** behind the decisions in `Documentation_template.md`: `blend_eval.py`, `bucket_thr.py`, `loco.py`, `fr_cells2.py`, `cell_compare.py`, `sizebias_scan.py`, `count_fit.py`, `count_groups*.py`, `hub_ambig.py`, `typo_swap.py`, `name_class.py`, `single_match.py`, `obvious_copies.py`, `thr_sweep_fit.py`, `group_thr.py`, `s2_round2.py`, `s2_catboost.py`, `st1_cmp.py`, `error_analysis.py`, `llm_eval.py` / `export_llm.py` (rejected LLM-judge experiment) and others.
  - They are not needed to reproduce the outputs.
  - Some read intermediate files from fixed paths in our cloud workspace (`/home/ubuntu/er/...`).
