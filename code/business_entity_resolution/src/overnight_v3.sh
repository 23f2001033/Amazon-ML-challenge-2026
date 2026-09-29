#!/bin/bash
# Overnight AWS pipeline (v3): scale-free stage 1 + rich stage 2 + full test run with band saving.
set -e
cd ~/er/code/business_entity_resolution/src
PY=~/er/.venv/bin/python
export PYTHONIOENCODING=utf-8 ER_TAG=v3 ER_WORKERS=6 ER_DROP_COUNTS=1 ER_SAVE_BAND=1
echo "== $(date +%T) block"      && $PY run_dev.py block
echo "== $(date +%T) features"   && $PY run_dev.py features
echo "== $(date +%T) train+val"  && $PY run_dev.py train
echo "== $(date +%T) stage2 rich dev" && ER_S2_RICH=1 $PY run_stage2.py dev v3
echo "== $(date +%T) stage2 base dev" && $PY run_stage2.py dev v3
cp ~/er/data/artifacts/lgbm_v3.txt ../models/lgbm_v3.txt
cp ~/er/data/artifacts/result_v3.json ../models/v3_config.json
echo "== $(date +%T) test run"   && $PY run_test.py ../models/lgbm_v3.txt ../models/v3_config.json ~/er/output_v3 v3
echo "== $(date +%T) export band for cross-encoder"
$PY export_band.py val v3 ~/er/data/features/xenc_val_band_v3.parquet
$PY export_band.py test v3 ~/er/data/features/xenc_test_band_v3.parquet
echo "== $(date +%T) OVERNIGHT_DONE"
