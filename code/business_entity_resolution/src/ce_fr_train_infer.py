"""French-aware cross-encoders (Kaggle T4 x2), trained on train_fr.parquet: 828k French pairs with
guaranteed labels built from the test France S1 records (S1 vs noisy copy = match; two different S1
at the same address / same name / same city word = non-match; shifted house = decoy) + 450k real
US/India raw pairs. Raw text "business_name | business_address".

  CE 3-FR : continue from the fine-tuned CE 3 checkpoint (bge-reranker-base, MIT)
            main(train_path, band_dir, hold_path, "/kaggle/working/ce3fr_out", "ce3fr", init=<ce3 model dir>)
  CE 4-FR : large reranker BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M), gradient checkpointing
            main(train_path, band_dir, hold_path, "/kaggle/working/ce4fr_out", "ce4fr",
                 init="BAAI/bge-reranker-v2-m3", max_train="800000", bs="24", lr="1e-5", ckpt="1")
Output: out_dir/band_{val,wide,test}_<tag>.parquet (s1, mid, p_ce) + printed French hold-out AUC.
"""
import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizerFast, get_linear_schedule_with_warmup

MAX_LEN, BS_INFER, EPOCHS = 128, 256, 1


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


def main(train_path, band_dir, hold_path, out_dir, tag, init, max_train="0", bs="96", lr="2e-5", ckpt="0"):
    os.makedirs(out_dir, exist_ok=True)
    try:
        tok = XLMRobertaTokenizerFast.from_pretrained(init)
    except Exception as e:
        print("tokenizer fallback to FacebookAI/xlm-roberta-base:", e)
        tok = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    base = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1)
    if ckpt == "1":
        base.gradient_checkpointing_enable()
    model = wrap(base)
    tr = pl.read_parquet(train_path)
    if int(max_train):
        tr = tr.head(int(max_train))
    print(f"[{tag}] init {init} | GPUs {n_gpu()} | training pairs {tr.height:,} (positives {int(tr['y'].sum()):,})", flush=True)
    dl = DataLoader(Pairs(tr, True), batch_size=int(bs) * n_gpu(), shuffle=True, collate_fn=collate(tok), num_workers=4, drop_last=True)
    steps = len(dl) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
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
    from sklearn.metrics import roc_auc_score
    ho = pl.read_parquet(hold_path)
    ho = ho.with_columns(pl.Series("p", predict(model, ho, tok)))
    print(f"[{tag}] French hold-out AUC {roc_auc_score(ho['y'].to_numpy(), ho['p'].to_numpy()):.4f}", flush=True)
    for k in ho["kind"].unique().to_list():
        h = ho.filter(pl.col("kind") == k)
        print(f"   {k:<6} n={h.height:,} mean p={h['p'].mean():.3f} (label {h['y'][0]})", flush=True)
    for kind in ("val", "wide", "test"):
        df = pl.read_parquet(os.path.join(band_dir, f"raw_band_{kind}.parquet"))
        p = predict(model, df, tok)
        df.select("s1", "mid").with_columns(pl.Series("p_ce", p)).write_parquet(os.path.join(out_dir, f"band_{kind}_{tag}.parquet"))
        print(f"{kind}: scored {df.height:,} band pairs ({time.time() - t0:.0f}s)", flush=True)
