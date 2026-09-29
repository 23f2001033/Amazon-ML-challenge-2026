# Amazon ML Challenge 2026: Business Entity Resolution (team Embedding)

**The task:** match noisy business records from two sources (S2, S3) to their clean reference entity (S1). It covers millions of records in the US and India, plus **France with no training labels at all**. The metric is macro F0.5 per S1 entity.

**Result:** public leaderboard **0.988574** (macro F0.5), and shortlisted in the **Top 1,000 teams** for the code-review round. The window was 25–27 Sep 2026 (72 hours, 5 uploads per day).

**Team:** Aman Kumar Maurya (lead), Reshma G, Afnan Sayyad, Ved Prakash Yadav.

## Approach in one view
1. **Canonicalisation.** Rule-based normalisation inverts the observed noise: legal forms, aliases, leetspeak, accents, Indic-script tokens and address abbreviations. Dictionaries are learned from train only.
2. **Blocking.** Region-scoped TF-IDF search with name, address, combined and fallback passes, plus learned "region-leak" expansion. About 70 candidates per S1.
3. **Learned pruning.** Two LightGBM pair scorers keep pairs with p ≥ 0.02. That leaves 4.51 candidates per S1, and `candidate_pairs.tsv` contains exactly these.
4. **Matcher.** Four fine-tuned multilingual cross-encoders (Multilingual-MiniLM, XLM-R base, bge-reranker-base, bge-reranker-v2-m3, two of them French-aware) feed a stacked second-stage LightGBM with competition and sibling features.
5. **Decision layer.**
   - Partition assignment, with thresholds tuned under test-like conditions.
   - Label-backed rules for zero-shot France: a same-address category-swap veto, and rescue of acronym and dropped-word copies.

The full methodology, with evidence for every decision, is in **[`Documentation_template.md`](Documentation_template.md)**.

## Repository layout
| Path | Contents |
|---|---|
| `Documentation_template.md` | Methodology write-up (the completed challenge template) |
| `code/business_entity_resolution/` | The pipeline: `src/`, `models/` (final LightGBM models and decision configs), `README.md` (exact reproduction steps), pinned `requirements*.txt` |
| `docs/` | Problem notes, decision log, EDA report, experiment log |
| `eda/` | EDA scripts (run in order `00`–`10`), summary outputs and figures |
| `submissions/SUBMISSIONS.md` | Every leaderboard upload with its change, validation and leaderboard score |
| `share_team/` | Scripts shared inside the team: Kaggle GPU cross-encoder scripts (`ce_kaggle/`) and the scoring and error-analysis kit (`scoring/`) |
| `check_submission.py`, `make_package.py`, `pack.sh` | Submission checks and packaging |

## Not included
- **The competition dataset.** Get it from the official challenge page. Place it as described in `code/business_entity_resolution/README.md` (§1).
- **Large intermediate files:** predictions, features, cross-encoder scores, submission TSVs and zips. The pipeline regenerates them.
- **Fine-tuned cross-encoder weights** (up to 2.2 GB). The training scripts are in `code/business_entity_resolution/src/` and `share_team/ce_kaggle/`.

## Licences
All models are MIT or Apache-2.0 and at most 568M parameters (LightGBM, Multilingual-MiniLM, XLM-R, bge-rerankers). No external data, APIs or lookups were used.
