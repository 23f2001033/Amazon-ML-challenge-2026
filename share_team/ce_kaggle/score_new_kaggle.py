"""Score new stage-1 pairs with the four saved fine-tuned cross-encoders (inference only, Kaggle GPU T4 x2).
Inputs (Add Input):
  - the dataset with canon_<kind>.parquet / raw_<kind>.parquet (kind = val, wide, and later test)
  - the notebooks whose output holds ce_out/model (CE v1), ce2_out/model (CE v2), ce3fr_out/model (CE 3-FR), ce4fr_out/model (CE 4-FR)
Output: /kaggle/working/new_scores.zip with <name>_<kind>.parquet (s1, mid, p_ce), name in v1, ce2, ce3fr, ce4fr.
Paste this whole file into one cell (after `!pip -q install polars`) and run it."""
import glob
import os
import shutil
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizerFast

# name, model folder, text file prefix, max length used in training, inference batch per GPU
JOBS = [("v1", "ce_out", "canon", 96, 1024), ("ce2", "ce2_out", "canon", 96, 512),
        ("ce3fr", "ce3fr_out", "raw", 128, 512), ("ce4fr", "ce4fr_out", "raw", 128, 192)]
IN = os.environ.get("ER_CE_IN", "/kaggle/input")   # searched recursively for the pair files and the */model checkpoint folders
OUT = os.environ.get("ER_CE_OUT", "/kaggle/working/new_scores")
ONLY = os.environ.get("ONLY", "v1,ce2,ce3fr,ce4fr").split(",")   # e.g. %env ONLY=ce4fr to split the work over two notebooks


class Pairs(Dataset):
    def __init__(self, df):
        self.a, self.b = df["text_a"].fill_null("").to_list(), df["text_b"].fill_null("").to_list()

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i]


def predict(model, df, tok, max_len, bs):
    def collate(batch):
        a, b = zip(*batch)
        return tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    dl = DataLoader(Pairs(df), batch_size=bs * max(1, torch.cuda.device_count()), shuffle=False, collate_fn=collate, num_workers=4)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for enc in dl:
            logits = model(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


os.makedirs(OUT, exist_ok=True)
t0 = time.time()
for name, folder, prefix, max_len, bs in [j for j in JOBS if j[0] in ONLY]:
    files = sorted(f for k in ("val", "wide", "test") for f in glob.glob(f"{IN}/**/{prefix}_{k}.parquet", recursive=True))
    models = [m for m in glob.glob(f"{IN}/**/{folder}/model", recursive=True) if os.path.exists(os.path.join(m, "config.json"))]
    if not models:
        print(f"!! MISSING model folder {folder}/model: attach the notebook that trained {name}", flush=True)
        continue
    print(f"== {name}: model {models[0]} | files {[os.path.basename(f) for f in files]}", flush=True)
    tok = XLMRobertaTokenizerFast.from_pretrained(models[0])
    model = AutoModelForSequenceClassification.from_pretrained(models[0], num_labels=1).cuda().eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    for f in files:
        kind = os.path.basename(f)[len(prefix) + 1:-len(".parquet")]
        dst = os.path.join(OUT, f"{name}_{kind}.parquet")
        if os.path.exists(dst):
            continue
        df = pl.read_parquet(f)
        p = predict(model, df, tok, max_len, bs)
        df.select("s1", "mid").with_columns(pl.Series("p_ce", p)).write_parquet(dst)
        print(f"   {name} {kind}: {df.height:,} pairs, mean p {p.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
shutil.make_archive(OUT, "zip", OUT)
print("DONE ->", OUT + ".zip", sorted(os.listdir(OUT)), flush=True)
