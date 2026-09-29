"""Cross-encoder on CPU, applied only to the uncertain band (stage-1 p in [0.02, 0.99)).

  python ce_cpu_band.py bench                      # measure train/infer pairs per second
  python ce_cpu_band.py run <n_train> <out_dir>    # train on n_train hard pairs, score val + test band

Model: microsoft/Multilingual-MiniLM-L12-H384 (MIT; 21M transformer params + 96M embeddings).
Train pairs: ce_data/train.parquet (train universe, positives + top-12 hard negatives).
Band pairs:  data/features/xenc_{val,test}_band_v3b.parquet (canonical text).
Output: out_dir/{val,test}_band_ce.parquet with columns s1, mid, p_ce.
"""
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
MAX_LEN, BS_TRAIN, BS_INFER, LR = 80, 64, 512, 5e-5
ER = os.path.expanduser("~/er")
torch.set_num_threads(os.cpu_count())


class Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

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


def band_texts(path):
    d = pl.read_parquet(path)
    a = (d["s1_name_canon"].fill_null("") + " ; " + d["s1_addr_canon"].fill_null("")).to_list()
    b = (d["c_name_canon"].fill_null("") + " ; " + d["c_addr_canon"].fill_null("")).to_list()
    return d, a, b


def predict(model, tok, a, b):
    dl = DataLoader(Pairs(a, b), batch_size=BS_INFER, shuffle=False, collate_fn=collate(tok), num_workers=2)
    out = []
    model.eval()
    with torch.inference_mode():
        for enc in dl:
            enc.pop("labels")
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).numpy())
    return np.concatenate(out)


def train(model, tok, a, b, y):
    dl = DataLoader(Pairs(a, b, y), batch_size=BS_TRAIN, shuffle=True, collate_fn=collate(tok), num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * len(dl)), len(dl))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time()
    for i, enc in enumerate(dl):
        yy = enc.pop("labels")
        loss = lossf(model(**enc).logits.squeeze(-1), yy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if (i + 1) % 200 == 0:
            done = (i + 1) * BS_TRAIN
            print(f"train {done:,}/{len(a):,} loss {loss.item():.4f} ({done / (time.time() - t0):.0f} pairs/s)", flush=True)


def load():
    tok = XLMRobertaTokenizerFast.from_pretrained(TOKENIZER)
    model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    return tok, model


def bench():
    tok, model = load()
    tr = pl.read_parquet(f"{ER}/ce_data/train.parquet").head(3200)
    t = time.time()
    train(model, tok, tr["text_a"].to_list(), tr["text_b"].to_list(), tr["y"].to_numpy().astype(np.float32))
    tr_rate = 3200 / (time.time() - t)
    _, a, b = band_texts(f"{ER}/data/features/xenc_val_band_v3b.parquet")
    t = time.time()
    predict(model, tok, a[:20000], b[:20000])
    inf_rate = 20000 / (time.time() - t)
    print(f"BENCH train {tr_rate:.0f} pairs/s | infer {inf_rate:.0f} pairs/s")
    print(f"  e.g. 1.0M train pairs -> {1e6 / tr_rate / 60:.0f} min; val+test band 2.29M -> {2.29e6 / inf_rate / 60:.0f} min")


def run(n_train, out):
    os.makedirs(out, exist_ok=True)
    tok, model = load()
    tr = pl.read_parquet(f"{ER}/ce_data/train.parquet").head(int(n_train))
    print(f"training on {tr.height:,} pairs ({int(tr['y'].sum()):,} positive)", flush=True)
    train(model, tok, tr["text_a"].to_list(), tr["text_b"].to_list(), tr["y"].to_numpy().astype(np.float32))
    model.save_pretrained(f"{out}/model")
    tok.save_pretrained(f"{out}/model")
    for kind in ("val", "test"):
        d, a, b = band_texts(f"{ER}/data/features/xenc_{kind}_band_v3b.parquet")
        t = time.time()
        p = predict(model, tok, a, b)
        d.select("s1", "mid").with_columns(pl.Series("p_ce", p)).write_parquet(f"{out}/{kind}_band_ce.parquet")
        print(f"{kind}: scored {d.height:,} band pairs in {time.time() - t:.0f}s", flush=True)
        if kind == "val":
            from sklearn.metrics import roc_auc_score, log_loss
            y = d["y"].to_numpy()
            ps = d["p_stage1"].to_numpy()
            print(f"  VAL band AUC  CE={roc_auc_score(y, p):.4f}  stage1={roc_auc_score(y, ps):.4f} | "
                  f"logloss CE={log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f} stage1={log_loss(y, np.clip(ps, 1e-6, 1 - 1e-6)):.4f}", flush=True)
    print("CE_DONE", flush=True)


if __name__ == "__main__":
    {"bench": lambda: bench(), "run": lambda: run(*sys.argv[2:4])}[sys.argv[1]]()
