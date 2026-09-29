# Run the cross-encoder on Kaggle (free GPU fallback)

Takes about 1.5–2 hours on a Kaggle T4. The output is two small files we plug into our stage-2 model.

## 1. Upload the data as a private dataset (5 min)
kaggle.com → **Datasets → New Dataset** → drag in `train.parquet`, `val.parquet`, `test.parquet` (≈460 MB) → name it `er-ce-data` → **Private** → Create.

## 2. Create the notebook
kaggle.com → **Code → New Notebook**. In the right panel:
- **Accelerator: GPU T4 x2** (or P100)
- **Internet: On** (needed once to download the MIT-licensed model `microsoft/Multilingual-MiniLM-L12-H384`; phone verification required)
- **Add Input** → your dataset `er-ce-data`

## 3. Cell 1: setup
```python
!pip -q install polars
```
## 4. Cell 2: paste the full contents of `ce_train_infer.py`, then run:
```python
import glob, os
data_dir = os.path.dirname(glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0])
main(data_dir, "/kaggle/working/ce_out", "1")   # 1 epoch over 2.23M hard pairs
```
Progress prints every 200 steps. At the end it prints **val AUC for the cross-encoder vs our stage 1**, which is the number we care about.

## 5. Send back
From `/kaggle/working/ce_out/` download **`val_ce.parquet`** and **`test_ce.parquet`** (columns `s1, mid, p_ce`) and share them. Also send a screenshot of the printed "val AUC … logloss …" line.

Notes: the batch sizes are set for a 24 GB GPU. On a T4 (16 GB), if you get out-of-memory, set `BS_TRAIN = 128` and `BS_INFER = 512` at the top of the script.
