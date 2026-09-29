"""Cross-encoder v2 (GPU, Kaggle T4 x2): fine-tune FacebookAI/xlm-roberta-base (MIT, 278M params)
as a pair classifier on hard pairs from 45 training regions, then score validation pairs.

Notebook use:
    train_path = glob.glob("/kaggle/input/**/train2.parquet", recursive=True)[0]
    val_path   = glob.glob("/kaggle/input/**/val.parquet", recursive=True)[0]
    main(train_path, val_path, "/kaggle/working/ce2_out")          # ~2 h on T4 x2
Later, inference only (no training):
    infer("/kaggle/input/<this notebook's output>/ce2_out/model", "<dir with *_new.parquet>", "/kaggle/working/ce2_scores", "val_new", ...)
Output: out_dir/val_ce.parquet (s1, mid, p_ce) and the model in out_dir/model.
"""
import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizerFast, get_linear_schedule_with_warmup

MODEL = "FacebookAI/xlm-roberta-base"
MAX_LEN, BS_TRAIN, BS_INFER, LR, EPOCHS = 96, 128, 512, 2e-5, 1   # batch sizes are per GPU


class Pairs(Dataset):
    def __init__(self, df, with_y):
        self.a, self.b = df["text_a"].fill_null("").to_list(), df["text_b"].fill_null("").to_list()
        self.y = df["y"].to_numpy().astype(np.float32) if with_y else None

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def collate(tok):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def n_gpu():
    return max(1, torch.cuda.device_count())


def predict(model, df, tok):
    dl = DataLoader(Pairs(df, False), batch_size=BS_INFER * n_gpu(), shuffle=False, collate_fn=collate(tok), num_workers=4)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for enc in dl:
            enc.pop("labels")
            logits = model(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def wrap(model):
    model = model.cuda()
    return torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model


def main(train_path, val_path, out_dir, max_train="0"):
    os.makedirs(out_dir, exist_ok=True)
    tok = XLMRobertaTokenizerFast.from_pretrained(MODEL)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    model = wrap(base)
    tr = pl.read_parquet(train_path)
    if int(max_train):
        tr = tr.head(int(max_train))
    print(f"GPUs: {n_gpu()} | training pairs {tr.height:,} (positives {int(tr['y'].sum()):,})", flush=True)
    dl = DataLoader(Pairs(tr, True), batch_size=BS_TRAIN * n_gpu(), shuffle=True, collate_fn=collate(tok),
                    num_workers=4, drop_last=True)
    steps = len(dl) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    t0, step = time.time(), 0
    model.train()
    for ep in range(EPOCHS):
        for enc in dl:
            y = enc.pop("labels").cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1)
                loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} ({el:.0f}s, ETA train {el / step * (steps - step) / 60:.0f} min)", flush=True)
    base.save_pretrained(os.path.join(out_dir, "model"))
    tok.save_pretrained(os.path.join(out_dir, "model"))
    va = pl.read_parquet(val_path)
    p = predict(model, va, tok)
    va.select("s1", "mid").with_columns(pl.Series("p_ce", p)).write_parquet(os.path.join(out_dir, "val_ce.parquet"))
    from sklearn.metrics import roc_auc_score, log_loss
    y = va["y"].to_numpy()
    print(f"val: scored {va.height:,} pairs ({time.time() - t0:.0f}s)\n"
          f"  val AUC  CE2={roc_auc_score(y, p):.4f}  stage1={roc_auc_score(y, va['p'].to_numpy()):.4f}  "
          f"logloss CE2={log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}", flush=True)


def infer(model_dir, data_dir, out_dir, *names):
    """Score pair files with the fine-tuned model: <data_dir>/<name>.parquet -> <out_dir>/<name>_ce.parquet."""
    os.makedirs(out_dir, exist_ok=True)
    tok = XLMRobertaTokenizerFast.from_pretrained(model_dir)
    model = wrap(AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1))
    t0 = time.time()
    for name in names:
        df = pl.read_parquet(os.path.join(data_dir, f"{name}.parquet"))
        p = predict(model, df, tok)
        df.select("s1", "mid").with_columns(pl.Series("p_ce", p)).write_parquet(os.path.join(out_dir, f"{name}_ce.parquet"))
        print(f"{name}: scored {df.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
