#!/bin/bash
# v3b: keep v1-era ambiguity counts, drop only the v2a count features. Reuses v3 candidates/features.
set -e
while ! grep -q "EXIT=" ~/er/overnight_v3.log; do sleep 60; done
cd ~/er/code/business_entity_resolution/src
PY=~/er/.venv/bin/python
export PYTHONIOENCODING=utf-8 ER_TAG=v3b ER_WORKERS=6 ER_SAVE_BAND=1
export ER_DROP_FEATS=pool_cnt_c,pool_cnt_s,rare_tok_idf,nm_idf_total1,nm_extra_tok_idf
D=~/er/data
ln -sf $D/candidates/dev_val_v3.parquet $D/candidates/dev_val_v3b.parquet
ln -sf $D/candidates/dev_train_v3.parquet $D/candidates/dev_train_v3b.parquet
ln -sfn $D/features/dev_train_v3 $D/features/dev_train_v3b
echo "== $(date +%T) train+val v3b" && $PY run_dev.py train
echo "== $(date +%T) stage2 base dev" && $PY run_stage2.py dev v3b
echo "== $(date +%T) stage2 rich dev" && ER_S2_RICH=1 $PY run_stage2.py dev v3b
cp $D/artifacts/lgbm_v3b.txt ../models/lgbm_v3b.txt && cp $D/artifacts/result_v3b.json ../models/v3b_config.json
echo "== $(date +%T) test run v3b" && $PY run_test.py ../models/lgbm_v3b.txt ../models/v3b_config.json ~/er/output_v3b v3b
$PY export_band.py test v3b $D/features/xenc_test_band_v3b.parquet
echo "== $(date +%T) V3B_DONE"
