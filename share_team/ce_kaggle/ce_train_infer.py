"""Cross-encoder (GPU): fine-tune Multilingual-MiniLM-L12-H384 (MIT, ~117M params) as a pair
classifier on our hard pairs, then score validation and test pairs.

  python ce_train_infer.py <data_dir> <out_dir> [epochs=1] [max_train_pairs=0 (all)]

Input files (from export_ce_data.py): train.parquet (text_a, text_b, y), val.parquet, test.parquet.
Output: out_dir/val_ce.parquet and test_ce.parquet with columns s1, mid, p_ce ; model in out_dir/model.
"""
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, XLMRobertaTokenizerFast, get_linear_schedule_with_warmup

MODEL = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER = "FacebookAI/xlm-roberta-base"  # same 250k sentencepiece vocab as the model (MIT); its fast tokenizer loads cleanly
MAX_LEN, BS_TRAIN, BS_INFER, LR = 96, 256, 1024, 5e-5


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


def predict(model, df, tok, dev):
    dl = DataLoader(Pairs(df, False), batch_size=BS_INFER, shuffle=False, collate_fn=collate(tok), num_workers=4)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for enc in dl:
            enc.pop("labels")
            logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def main(data_dir, out_dir, epochs="1", max_train="0"):
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    tok = XLMRobertaTokenizerFast.from_pretrained(TOKENIZER)  # fast (Rust) tokenizer: keeps the GPU fed
    model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    tr = pl.read_parquet(os.path.join(data_dir, "train.parquet"))
    if int(max_train):
        tr = tr.head(int(max_train))
    dl = DataLoader(Pairs(tr, True), batch_size=BS_TRAIN, shuffle=True, collate_fn=collate(tok), num_workers=4, drop_last=True)
    steps = len(dl) * int(epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    t0, step = time.time(), 0
    model.train()
    for ep in range(int(epochs)):
        for enc in dl:
            y = enc.pop("labels").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1)
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
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(os.path.join(out_dir, "model"))
    tok.save_pretrained(os.path.join(out_dir, "model"))
    for name in ("val", "test"):
        df = pl.read_parquet(os.path.join(data_dir, f"{name}.parquet"))
        p = predict(model, df, tok, dev)
        df.select("s1", "mid").with_columns(pl.Series("p_ce", p)).write_parquet(os.path.join(out_dir, f"{name}_ce.parquet"))
        print(f"{name}: scored {df.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
        if name == "val":
            from sklearn.metrics import roc_auc_score, log_loss
            y = df["y"].to_numpy()
            print(f"  val AUC  CE={roc_auc_score(y, p):.4f}  stage1={roc_auc_score(y, df['p'].to_numpy()):.4f}  "
                  f"logloss CE={log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f} stage1={log_loss(y, np.clip(df['p'].to_numpy(), 1e-6, 1 - 1e-6)):.4f}", flush=True)


def infer(model_dir, data_dir, out_dir, *names):
    """Score more pair files with an already fine-tuned model (no training):
    <data_dir>/<name>.parquet (s1, mid, text_a, text_b) -> <out_dir>/<name>_ce.parquet (s1, mid, p_ce)."""
    os.makedirs(out_dir, exist_ok=True)
    tok = XLMRobertaTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=1).to("cuda")
    t0 = time.time()
    for name in names:
        df = pl.read_parquet(os.path.join(data_dir, f"{name}.parquet"))
        p = predict(model, df, tok, "cuda")
        df.select("s1", "mid").with_columns(pl.Series("p_ce", p)).write_parquet(os.path.join(out_dir, f"{name}_ce.parquet"))
        print(f"{name}: scored {df.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__" and len(sys.argv) >= 3 and not sys.argv[1].startswith("-"):
    main(*sys.argv[1:])   # CLI use; in a notebook, call main(data_dir, out_dir, "1") yourself
