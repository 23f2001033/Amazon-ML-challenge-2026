"""Inference-only scoring of one raw-text pair shard with a LOCAL checkpoint.

Uses the checkpoint's saved tokenizer and one GPU (cuda:0) by default, never
downloads a model, never trains, and never wraps the model in DataParallel.
Output columns: s1, mid, p_ce, ce_logit (actual unrounded classifier logits).
"""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import polars as pl

from export_confident_ce import KEYS, validate_keys


def infer(pairs, checkpoint, output, batch_size=256, max_length=128,
          device="cuda:0"):
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    checkpoint = Path(checkpoint).resolve()
    output = Path(output)
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise ValueError("checkpoint must be a local saved model directory containing config.json")
    if output.exists():
        raise FileExistsError("Inference output already exists")
    frame = pl.read_parquet(pairs, columns=[*KEYS, "text_a", "text_b"]).with_columns(
        *[pl.col(c).cast(pl.String) for c in KEYS])
    validate_keys(frame, "Inference shard")
    if any(frame[c].null_count() for c in ("text_a", "text_b")):
        raise ValueError("Inference text must be non-null; export raw text first")

    # Lazy imports make --help/export/coverage checks usable without GPU deps.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    target = torch.device(device)
    if target.type not in {"cuda", "cpu"}:
        raise ValueError("device must select one CUDA GPU or cpu")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a CPU smoke test")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False)
    if model.config.num_labels != 1:
        raise ValueError("Checkpoint must be a one-logit binary cross-encoder")
    model.to(target).eval()
    probabilities, logits_out = [], []
    start = time.monotonic()
    amp = (torch.autocast("cuda", dtype=torch.float16)
           if target.type == "cuda" else nullcontext())
    with torch.inference_mode(), amp:
        for offset in range(0, frame.height, batch_size):
            batch = frame.slice(offset, batch_size)
            encoded = tokenizer(batch["text_a"].to_list(), batch["text_b"].to_list(),
                                truncation=True, max_length=max_length,
                                padding=True, return_tensors="pt")
            logits = model(**{k: v.to(target) for k, v in encoded.items()}).logits
            if logits.ndim != 2 or logits.shape != (batch.height, 1):
                raise ValueError("Unexpected classifier logit shape")
            logits = logits[:, 0].float()
            if not torch.isfinite(logits).all().item():
                raise ValueError("Classifier produced non-finite logits")
            logits_out.extend(logits.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            processed = min(offset + batch_size, frame.height)
            if (offset // batch_size + 1) % 100 == 0 or processed == frame.height:
                elapsed = time.monotonic() - start
                print(f"Scored {processed:,}/{frame.height:,} pairs in {elapsed:.1f}s", flush=True)
    result = frame.select(KEYS).with_columns(
        pl.Series("p_ce", probabilities, dtype=pl.Float32),
        pl.Series("ce_logit", logits_out, dtype=pl.Float32),
    )
    validate_keys(result, "Inference result")
    if result.height != frame.height or not result.select(KEYS).equals(frame.select(KEYS)):
        raise ValueError("Inference result does not exactly cover the input shard")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError("Partial output already exists; inspect it before retrying")
    result.write_parquet(temporary)
    written = pl.read_parquet(temporary, columns=KEYS)
    if not written.equals(frame.select(KEYS)):
        raise ValueError("Written inference output failed exact pair coverage validation")
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "checkpoint", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    infer(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
