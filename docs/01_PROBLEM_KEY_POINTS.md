# Amazon ML Challenge 2026: Problem Key Points

Sources: the problem statement PDF, the guidelines PDF, `student_resource/README.md`, `Documentation_template.md` and `utils/validate_submission.py`. This file lists every rule or fact that changes how we build the solution. Each point says why it matters.

---

## 1. The task in one line
**Business Entity Resolution:** for every **Source 1 (S1)** record, return every **Source 2 (S2)** and **Source 3 (S3)** record that refers to the same real business. The answer can be zero, one or many records.

| # | Key point | Why it matters for us |
|---|-----------|-----------------------|
| 1.1 | S1 is the **deduplicated reference** source. Its records are clean, canonical and unique. | We query from S1 into S2/S3. S1 never matches S1. |
| 1.2 | S2 and S3 are **noisy** and share **no common identifiers** with S1. | Matching can only use name, address and country text. |
| 1.3 | One S1 entity can match **0..N** records from S2 and S3 together. | This is a one-to-many linkage, not 1:1 record matching. |
| 1.4 | Only three columns carry signal: `business_name`, `business_address`, `country`. The source is given by the ID prefix. | Features are entirely string-based. |

## 2. Data facts from the statement
| # | Key point | Impact |
|---|-----------|--------|
| 2.1 | All files are **TSV**. Commas appear inside addresses and ID lists. | Always read with `sep="\t"` and no quote processing, and write TSV. |
| 2.2 | Train covers **US and India**. Test **adds France**, which never appears in train. | **Zero-shot country generalisation.** No country-specific hard-coding, filtering or one-hot. Features must be country-agnostic, or France-specific rules must be written without labels. France must appear in the submission. |
| 2.3 | Name noise: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix inconsistency, DBA/trade names, `&` vs "and", word-order swaps, typos. | Normalisation, token-order-invariant similarity, and legal-suffix-aware features. |
| 2.4 | Address noise: abbreviations (Rd/Road, St/Street), transliteration, missing components (PIN, state), landmark references ("Near SBI ATM"), municipal numbering, component reordering. | Parse addresses into order-invariant components. Use set-based similarity rather than sequence similarity. |
| 2.5 | `train_ground_truth.tsv` has one row per S1 with a comma-separated list of matches. The list is empty for singletons. | Positive pairs come from exploding the list. Everything else is a negative. |

## 3. Output and format rules (violations mean rejection)
| # | Rule | Impact |
|---|------|--------|
| 3.1 | `matching_results.tsv` has header `source1_entity_id<TAB>matched_entity_ids`. **Only this file is scored.** | This is the leaderboard file. |
| 3.2 | **Exactly one row per test S1 entity.** Missing S1 IDs mean rejection. | Always left-join onto the full test S1 list, France included. |
| 3.3 | Empty `matched_entity_ids` for no-match entities. | Singletons get an empty string, not NaN or "None". |
| 3.4 | No duplicate IDs within a list, and no duplicate S1 rows. | Deduplicate before writing. |
| 3.5 | Only S2-/S3- IDs that exist in the test set. No S1 self-matches. | Only emit IDs from the test S2/S3 files. |
| 3.6 | `candidate_pairs.tsv` (same format, column `candidate_entity_ids`) must contain the **exact set fed to the final model for inference**. Matches must be a **subset** of candidates. | It is not scored but is audited for recall ceiling and reduction ratio. The blocking stage must be clean and recorded. |
| 3.7 | Run `utils/validate_submission.py` before every upload. | Cheap insurance, since there are only 5 uploads per day. |

## 4. Evaluation metric: the most important section
| # | Key point | Impact |
|---|-----------|--------|
| 4.1 | **F-beta with beta = 0.5**: `F0.5 = 1.25·P·R / (0.25·P + R)`. Precision counts about twice as much as recall. | A false merge costs more than a missed link. The decision threshold should be higher than the F1-optimal one. |
| 4.2 | **Macro-averaged per S1 entity.** Each S1 entity contributes one score and all entities weigh the same. | A large cluster counts no more than a small one. Errors on small clusters (1–2 matches) hurt a lot. |
| 4.3 | **Singletons:** an empty prediction scores **1.0**. **Any** prediction scores **0.0**. | Deciding whether an entity has any match at all is its own sub-problem. It needs an entity-level "has-any-match" confidence, not only pair scores. |
| 4.4 | A non-singleton entity with an empty prediction scores 0 (recall = 0). | Emptying a list is costly when matches exist. Tune on the per-entity macro metric, not on pair-level F. |
| 4.5 | The threshold must be chosen to maximise **macro per-entity F0.5** on a held-out validation split. | We need our own evaluator that exactly reproduces the official metric. |
| 4.6 | Public leaderboard = a subset of test. Private leaderboard (final) = the rest. | Don't overfit the public LB. Trust local validation. |

## 5. Constraints and fair play
| # | Rule | Impact |
|---|------|--------|
| 5.1 | The final model must use an **MIT or Apache-2.0** license and have **at most 8B parameters**. | Check licenses of any pretrained encoder (for example sentence-transformers). GBDTs such as LightGBM (MIT) and XGBoost (Apache) are fine. |
| 5.2 | **No external data lookup**: no ER APIs, business registries, geocoding APIs or internet augmentation. | All normalisation is rule-based or learned from train. Generic language knowledge (state-code tables, abbreviation lists) and offline libraries are fine. Any data dictionary we build must be learned from the training data. |
| 5.3 | Code is audited and must be reproducible end to end (data → blocking → matching → output). | Scripted pipeline, fixed seeds, pinned `requirements.txt`, README. |
| 5.4 | Multiple IDs or accounts mean disqualification. Only one login is allowed at a time. | Operational rule. |

## 6. Logistics
| # | Point | Impact |
|---|-------|--------|
| 6.1 | Window: **25 Sep 2026 00:00 IST – 27 Sep 2026 23:59 IST** (3 days). | A tight schedule. Build a baseline submission early. |
| 6.2 | **Max 5 submissions per day.** | Every upload must be locally validated and locally scored first. |
| 6.3 | Keep a version history of all submissions. | Keep a `submissions/` log with a timestamp, local validation score and LB score. |
| 6.4 | Deliverables: a zip containing `output/` (both TSVs), `code/business_entity_resolution/{src, README.md, requirements.txt}`, and the filled `Documentation_template.md`. The guidelines also ask for a 1–2 page approach document and commented source code. | Keep code modular and commented from the start. Keep the methodology document updated as we go. |
| 6.5 | The methodology document must cover the methodology, the **blocking strategy**, the **model architecture and feature engineering**, and anything else relevant. | These EDA and decision docs feed straight into it. |

## 7. Official tips, with our reading
- *"Invest in a strong blocking strategy. It sets the recall upper bound."* Blocking recall is the ceiling on everything else.
- *"Explore Jaccard, Levenshtein and TF-IDF cosine."* These are the baseline feature families.
- *"Pay attention to country-specific address patterns."* This must be balanced against France being unseen.
- *"Consider the P/R trade-off. F0.5 rewards precision."* Tune the threshold on macro F0.5.
- *"Don't neglect singletons."* In train only **5.6%** of S1 are singletons (see EDA), so they matter less than the tip suggests. They still carry 1.0 each and are easy to lose to a single false positive.

## 8. Scale and compute facts (from EDA; they shape engineering)
| Split | S1 | S2 | S3 |
|-------|----|----|----|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test  | 1,732,544 | 4,887,273 | 5,082,316 |

Local machine: 16 GB RAM, 16 CPU threads, an RTX 2050 with 4 GB (CUDA is not currently usable from torch). The full cross-product (~10^13 pairs) is impossible, so blocking is mandatory. Memory is the main engineering constraint: we use Parquet, Polars and sparse top-k.
