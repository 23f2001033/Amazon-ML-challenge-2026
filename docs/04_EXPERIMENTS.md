# Experiment Tracker

The single place to see **what we tried, what worked, what didn't, and what's next**.
Rule: every change is one isolated experiment (E-id) measured on the **same** validation universe (D-020) against the current best. A change is merged only if it improves val macro F0.5 by more than noise (≥ +0.001) without hurting India or US by more than 0.001. Leaderboard uploads are reserved for merged versions or deliberate probes.

---

## Scoreboard
| Version | Val macro F0.5 | India / US (val) | Public LB | LB − val | Status |
|---|---|---|---|---|---|
| **v1** | **0.9769** | 0.9725 / 0.9797 | **0.966474** (~rank 135) | **−0.0104** | submitted 25 Sep 18:59 |
| v1b | = v1 (France-only change) | = v1 | **0.966763** | | probe, 25 Sep 20:01 |
| **v2a** (E2+E3+E4) | **0.9810 (+0.0041)** | 0.9771 / 0.9833 | **0.965516** | **−0.0155** | LB below v1b: count features shifted test probabilities down |
| **v2b** (v2a + stage-2 specialist) | **0.9824 (+0.0014, 2-fold CV)** | 0.9791 / 0.9844 | **0.967982** | −0.0144 | stage 2 transfers to LB (+0.0025 vs v2a) |
| **v2c** (v2b, thr 0.50 calibrated to v1 test match rates) | n/a | | **0.968091** (best, 25 Sep) | | calibration helps only slightly (+0.0001) |
| v3 (v2a minus ALL 7 count features) | **0.9738 (−0.0072 vs v2a)** | 0.9701 / 0.9760 | _test run overnight_ | | ambiguity counts carry real signal; LB will show whether the shift outweighs it |
| v3 + rich stage 2 (f1_ features leaked counts back) | 0.9813 (+0.0076 over v3 s1) | 0.9768 / 0.9841 | | | the gain was mostly the re-introduced count features |
| **v3b** (drop only the 5 v2a count features; keep v1's amb counts) | **0.9788** | | _test run overnight_ | | middle ground |
| v3b + base stage 2 | 0.9805 (+0.0016) | 0.9768 / 0.9827 | | | |
| **v3b + rich stage 2** | **0.9810 (+0.0022)** | 0.9775 / 0.9832 | _candidate for 26 Sep_ | | |
| target v2 | ≥ 0.982 | | ≥ 0.975 | | |
| top-10 LB (25 Sep evening) | | | 0.984–0.985 | | |

---

## v1: what it is
Canonicaliser (D-021) → region-scoped TF-IDF blocking, 25 name + 25 address + 10 fallback candidates (98.6% recall, 56/S1) → 65 features → LightGBM (13 of 19 train chunks, negatives subsampled) → partition constraint → threshold 0.75.

### What worked (keep)
| Component | Evidence |
|---|---|
| Canonicalisation incl. learned Indic dictionary | Native-script names are only 2.8% of v1 false negatives vs 6.4% of true pairs, so they are handled better than average |
| Region-scoped blocking | 98.63% recall, better than the 96.2% global kNN, and 10× faster |
| Competition features (`pre_rank_mid`, `pre_gap_mid`) | #2 and #4 by gain; lost-competition FNs are only 1.4% of FNs |
| House-number features | #3, #6 and #10 by gain |
| Partition constraint | 0 violations; FPs on records owned by another queried S1 are only 11% of FPs |
| Threshold tuned on macro F0.5 | Flat optimum 0.70–0.80 |

### What didn't work / limits
| Item | Evidence |
|---|---|
| Expected-F0.5 decision rule | 0.9763 vs threshold 0.9769; probabilities are not calibrated enough (TENTATIVE D-015) |
| Validation does not represent France | LB − val = −0.0104. If US/India transfer exactly, France scores about **0.91** (0.9665 = 0.38·0.9797 + 0.47·0.9725 + 0.15·x) |
| Training on only 13 of 19 chunks | Laptop RAM/disk crash (D-024) |

---

## v1 error analysis (val, `eda/10_error_analysis_v1.py`)
Total loss = 1 − 0.9769 = **0.0231**.

| Entity outcome | Entities | Loss pts | Share |
|---|---|---|---|
| Missed match(es) only | 28,253 | 0.01233 | **53.5%** |
| Has matches, predicted empty | 822 | 0.00418 | **18.1%** |
| Extra false match(es) only | 3,611 | 0.00396 | 17.2% |
| Singleton given a false match | 288 | 0.00146 | 6.3% |
| Both FP and FN | 720 | 0.00112 | 4.9% |

**About 72% of the loss is recall. About 28% is precision.**
- **False negatives (34,727 pairs):**
  - scored below threshold: 72% (median p 0.30)
  - never a candidate (blocking miss): 27%
  - lost competition: 1.4%
- **FN profile:** **49% have an empty S2/S3 address**, vs 13.5% of all true pairs. Only 20% share the house number (59% of all true pairs do). Region is equal in only 48% (86% of all true pairs).
- **False positives (4,841 pairs):**
  - true distractors: 61%
  - orphans of dropped S1s: 28%
  - records of another queried S1: 11%
- **FP examples** are decoys:
  - same address, house number ±1 (515/516, unit 903/904, 359 vs 359½)
  - same address, name-mutated distinctive token (Pranva→Pranoro, Ujjaiya→Ujjaiyayn, Yamuita→Yamuioro, Grow→Glow, MD→HD)

### France eyeball (test, no labels; `eda/outputs/10b_france_eyeball.txt`)
- **Accepted near the threshold, likely false merges:** generic-word substitution at the same address (VC Patrimoine→VC Sportive, Gymnastique Club→Compagnie, Art Soins→Art Sportive, MW Collège→XW Collège), and pseudo-word names at the same address (Brixlyra, VEOVANTAGE).
- **Rejected grey zone, likely misses:** exact-name records with an **empty address** (Aqua Amis SARL, OS Club SARL, Reseau Sportive S.A.S, 0.48–0.68), and acronym typos (FXQ→JFXQ, 0.45).

### Hypotheses tested and REJECTED (do not rebuild)
| ID | Hypothesis | Result |
|---|---|---|
| H-1 | "A substituted name token (both sides have an unmatched token) marks a decoy" | **False on train.** Substitution pairs are 94.3% true matches (112k pairs, p ≥ 0.3), because the noise generator also substitutes. A hard rule would destroy recall. It may still help **as a feature** combined with others. |
| H-3 (E1 premise) | "A deviation from the S1 shared by several candidates marks a decoy business" | **Direction reversed.** A shared house-number deviation is 97.1% true (65.7k pairs), a unique one is 67.8% true (44.3k). True copies agree with each other; decoys are isolated one-off records. The signal to use is **sibling support**. Name-variant consensus carries no signal (88.6% vs 89.0%). Null-address name-sibling linking is weak (44.8% vs 36.5%). |
| H-2 (E4) | "An exact compact-name key across regions recovers blocking misses (region mismatch)" | **False.** +0.0001 recall for +2 candidates/S1 (WV, OK). Region mismatch is not what hides true pairs. The key is kept in code but switched off. |

---

## v2 backlog (ranked by expected gain, each an isolated experiment)
| ID | Experiment | Targets | Why (evidence) | Expected gain (val) | Cost |
|---|---|---|---|---|---|
| **E1** | **Second-stage model with cluster/consensus features** from v1 probabilities: (a) competition in p-space (rank and gap of p among S1s claiming the record, best rival p); (b) the S1's confident-match count per source; (c) agreement of the candidate with the S1's other confident matches (house number, name, address tokens) | FN below threshold + decoy FPs | Decoys disagree with the S1's consensus; true noisy matches agree. The below-threshold FNs have median p 0.30 | +0.003 to +0.006 | medium |
| **E2** | **Null-address evidence**: exact/compact-name count in the S2/S3 pool, name uniqueness among S1 in the country, "S1 has an open slot" (no match yet from this source), name-only similarity stack | 49% of FNs; France misses | Identical-name null-address pairs are 73.6% true, but only 57.3% accepted | +0.002 to +0.004 | low |
| **E3** | **Train on all 19 chunks** + light hyperparameter tuning (leaves, min_data, rounds) | everything | v1 used 13/19 chunks | +0.0005 to +0.0015 | low (compute) |
| **E4** | **Blocking recall**: K_FALL 10→30, exact compact-name key for region-less records, city→region inference for S2/S3 | 27% of FNs are blocking misses (ceiling 0.9957) | Misses are dominated by null-address and alias names | +0.001 to +0.002 | low–medium |
| **E5** | **Leave-one-country-out validation** (train US → val India, and the reverse) | measure France-like generalisation | LB − val = −0.0104; France is the likely cause | diagnostic | medium |
| **E6** | **France-robust features**: short-acronym exact match, distinctive-token (lowest-df) similarity, generic-word-aware name similarity (IDF within block) | France FPs and FNs | France eyeball: acronym and generic-word patterns | LB +0.002 to +0.005 (France share) | medium |
| **E7** | **Per-country threshold probe on LB** using saved `test_pred.parquet` (no rerun): France threshold 0.65 / 0.85 | France decision | Direction of the France error is unknown; one upload answers it | LB diagnostic | ~0 compute, 1 upload |
| **E8** | **Calibration + expected-F** (isotonic on held-out) | decision | Expected-F lost by 0.0006 under miscalibration | +0.000 to +0.001 | low |
| **E9** | **Cross-encoder feature** (Ved): fine-tuned on our pairs, scored only on the uncertain band | grey zone | Only if E1–E4 plateau | unknown | high (GPU) |

### Findings on 25 Sep evening that re-rank the backlog
1. **v2a** (E2 empty-address/IDF features + E3 all 19 chunks + E4 fallback 20): val **0.9810** (+0.0041). India 0.9771 (+0.0046), US 0.9833 (+0.0036). Blocking ceiling 0.9962.
2. **Val vs test prediction profiles are the same** (v1 model): accepted/S1, grey-zone/S1 and empty rate all match per country, France included. The LB gap is therefore **confident errors, not uncertainty**. For France, v1b implies about 50% false merges in the accepted 0.75–0.85 band.
3. Even a perfect France caps v1 at about 0.979 LB, so the core model must also improve on US/India. Top LB is 0.9870.
4. **Grey-zone oracle (v2a val):** perfect decisions for pairs with p in 0.2–0.95 give **0.9906 (+0.0097)**, touching only **0.26 pairs/S1** (0.4% of candidates). For p in 0.05–0.98: 0.9931 (+0.0121), 0.54 pairs/S1. **The remaining gap is concentrated in a tiny set of pairs, so a heavy specialist model (stage-2 / fine-tuned cross-encoder) is affordable on test (about 0.45–0.95M pairs).**

### Day-1 closing lessons (25 Sep, 23:30)
1. **Train/test feature shift is real.** Raw-count features (name frequency in the pool, S1 ambiguity counts, IDF magnitudes) depend on split size and distractor density. v2a lowered test match rates on all countries (FR 3.28→3.16, IN 3.24→3.20, US 3.34→3.30 matches/S1) and lost LB despite +0.004 val. **v3 drops them (ER_DROP_COUNTS=1).**
2. **Stage 2 generalises**: +0.0014 val CV became +0.0025 LB.
3. **Threshold / calibration moves are small** (≤ 0.0003 each). Stop spending uploads on them.
4. **Unknown split of the 0.014 gap between France and shift.** Probe `submissions/probe_fr_empty` measures France directly: F_France ≈ (0.968091 − LB_probe)/0.15 + 0.056 (assuming a proportional France share in the public subset).

### Night of 25→26 Sep: where the remaining validation loss is (v3b, uncertain band)
- **H-4 (edit type separates decoys from noise): mostly refuted.** Only "vowel-substitution-only" pairs behave decoy-like (14.7% true vs model 0.35), and that's just 539 pairs. The band is dominated by identical names (104k pairs, 31% true) and heavily different names (91k, 37% true).
- **The model is calibrated in every bucket** (mean p ≈ true rate), so the residual val errors are mostly **inherent ambiguity**: empty-address records whose name is shared by 2–5 S1 (34% true; 4.2k FN), and same-street records with a different house number (55% true; about 2.6k FN and 1.1k FP).
- **Therefore the val→LB gap (≈0.014) is now the largest lever.** Val is near its practical ceiling for our information. Day 2 opens with the France probe (France's share of the gap) and v3b + rich stage 2 (the count-shift share).

### 26 Sep 02:00–04:00: adversarial validation and transfer-safe models
- **Adversarial validation** (can a classifier tell train-side from test pairs, uncertain band): ALL features AUC **0.967** (giveaways: nm_extra_tok_idf 41%, rare_tok_idf 13%, amb_country 9%, n_s1_for_mid 8%). Minus counts: 0.798. Minus counts and competition ranks/counts: **0.687**. Pure pairwise: 0.667. France vs train: 0.92–1.0 always (structurally different).
- **Root cause:** competition/context features (n_s1_for_mid val 10.4 vs test 18.1 vs France 49.4; pre_rank_mid 1.7 vs 2.2 vs 10.4) depend on processing-batch size and country size, which differ between val and test. **This is a validation-design artefact**, and the likely main source of the val→LB gap.
- **Test structure EDA:** test's extra records are mostly orphans (names matching no S1: US 56%→50%); decoy-like records rise only +0.7–1.9 pp. The distractor mix is not the cause.
- **v4 (transfer-safe, drops counts + ranks):** v4a 0.9722 / +S2 0.9753; **v4b** (keeps relative gaps) 0.9725 / **+S2 0.9754**. The ranks/counts are worth ≈0.006 on val. Whether they're worth it on test is measured by uploads A/B.
- **CPU cross-encoder not viable** (5 train / 85 infer pairs/s). The GPU quota is pending at AWS; a Kaggle kit is ready (`share_team/ce_kaggle`).
- **Diagnostics without wasting uploads:** country-mix submissions (`mix_countries.py`) replace the pure France probe.

### 26 Sep 12:00: upload A = 0.966111 and what it proves
- **A (v4b transfer-safe: no count or competition-rank features, calibrated per country) scored 0.966111, the lowest of all** (v2c 0.968091). Its val→LB gap shrank from 0.0144 to 0.0093, but about 0.007 of real signal was lost. **Count and competition features carry test signal. Compute them test-consistently (v5) instead of deleting them.** The transfer-safe direction is rejected.
- **Singleton check (label-free):**
  - Counting test S1s with no candidate at p ≥ 0.02 as empty, the predicted no-match rate is US 5.9%, India 6.4%, France 5.9%.
  - Val: predicted 5.6%, truth 5.7%. The train singleton rate is 5.58%.
  - **No singleton over-matching on test.**
- **Match-count profile, same v3b stage 1, matches/S1 at p ≥ 0.5:**
  - Val: US 3.37, India 3.37 (truth 3.42).
  - Test: US 3.40, **India 3.32**, France 3.43 (France has the highest grey mass, 0.10).
  - v2a's count features push all test countries down by another 0.04–0.12, which is why v2c's lower threshold helped.
  - **India under-predicts on test. France is the most uncertain.**
- Distractor density was already matched in val (81% queries), and the region-less fallback pool in val is the full train pool, so it is no easier than test.

### 26 Sep 13:00–15:00: the gap is India blocking (wide-region check)
- **Wide universe** (`eval_regions.py`): 20 unseen train regions, 633k queries, giant regions allowed.
  - v3b stage 1: val 0.9788 vs **wide 0.9698**.
  - US 0.9815 → 0.9812 (transfers). India 0.9746 → **0.9579**: Telangana **0.876**, Maharashtra 0.968, Tamil Nadu 0.976, UP 0.977.
  - US 0.981 and India 0.958 at test mix ≈ 0.968 = the LB.
- **Telangana:**
  - S2/S3 write Hyderabad addresses with the old state name "Andhra Pradesh": 18.5% of pairs, and region blocking never compares them.
  - Other learned leaks: DC → "WA" 77%, WA → DC 2.6%, Chandigarh → Punjab 11%, and 16 pairs in all (`region_leaks.py`).
- **Maharashtra:** half of the in-region misses have the exact S1 name, shared by a median of 35 pool records, and truncated addresses. The name top-25 cut is arbitrary among ties.
- **v6 blocking** (combined name+address pass + leak expansion), recall:
  - Telangana 0.803 → **0.9915**, Maharashtra 0.963 → 0.985, NC 0.992 → 0.995.
  - Val 0.9877 → **0.9916**, wide ≈0.972 → **0.9908**, big train universe 0.9920.
  - Candidates per S1: 66.5 → 69–74.
- **Stage 2 trained on val + wide** (4 region folds, out-of-fold; smoke test on the v3b base): val 0.9788 → **0.9812**, wide 0.9699 → **0.9723** (India 0.9581 → 0.9613). The gain transfers to the hard regions.
- **Cross-encoder** (Kaggle T4, 1 epoch on 2.23M hard pairs, ~51 min): val AUC **0.9947 vs stage 1 0.9945** on 994k pairs. Log-loss 0.084 vs 0.075, so it's less well calibrated. It is to be stacked in stage 2.
- **Ops lesson:** never run a second CPU-heavy job next to LightGBM training. OpenMP oversubscription stalled v6 training for over an hour.

### 26 Sep evening: stacking and decision experiments (v6 base, out-of-fold on val + wide)
| Experiment | Weighted val+wide | Verdict |
|---|---|---|
| v6r stage 2, no cross-encoder | 0.9835 | LB 0.977243 |
| **+ CE v1 (v6rc)** | **0.9870** | +0.0035 |
| + CE v2 band features (tested on the v3b base, val only) | +0.0007 | keep |
| Seed-bagged stage 2 (3 seeds) | 0.9870 | no gain, dropped |
| Expected-F0.5 set selection instead of a threshold | +0.0001 | not adopted |
- **Remaining wide loss (v6rc, 0.0129):** 59% missed matches, and 79% of the missed-but-scored pairs have an empty address. The name is typically shared by about 5 S1s; only 6–8% can be singled out by raw name or legal form, so these are ambiguous for any method.
- **France (no labels):** about 0.94–0.97 by back-calculation from the LB. It is the largest remaining lever. French canonicalisation was checked and is fine.

### 27 Sep night: France measurement and data-mining for hidden signals
- **France measured:** A (v6rc3 with France emptied) = 0.849373, together with v6rc3 = 0.983658:
  - US+India on test ≈ **0.989**, above the wide validation 0.988.
  - **France ≈ 0.953**, and ≤ 0.975 for any plausible French singleton rate.
  - France is the whole gap to 0.99.
- **v7 (wider blocking, K 40/40/40/30):** recall val 0.9916 → 0.9931, but stage 2 0.9835 → 0.9838 only. Not adopted.
- **Checked on France without labels, no bug found:**
  - canonicalisation of French names and addresses is correct;
  - French S1 twins (same name + address) are 0.03% of S1;
  - house-number decoys are accepted *less* in France than in the US.
- **Excess French matches:** France accepts more pairs with an exact address and a weak name match (name < 50: 0.148/S1 vs ~0.065). Shared S1 addresses are 3× as common (13.9%).
- **Shared-address stage-2 features** (neighbours at the S1's address, name match of the best neighbour, name similarity without the city word): val+wide 0.9879 → 0.9879, gain share ≤ 0.04%. Not adopted.
- **Hidden signals searched:**
  - S1-only attributes vs singleton: **AUC 0.501**. Singletons are random.
  - Raw-format fingerprints (identical raw address, upper case, source): no distractor fingerprint. An identical raw address means more likely *true*.
  - **Record-only attributes vs distractor: AUC 0.778 (0.7835 out-of-region).** Distractors almost always carry a full address with a house number and longer, composite names. Records without a house number are 97.4% true; empty-address records are 97.7% true. → `record_prior.py`, leak-free, tested as a stage-2 feature.
  - If the record's name equals another S1's name, an "exact address + unrelated name" pair is only 4–36% true. Already handled by the model.
- **Stage 2 relies most on CE 3's margin over the rival S1** (raw-text reranker; 2.3× the next feature). Hence the overnight **French-aware CE 3-FR / CE 4-FR**, fine-tuned on 828k French pairs with guaranteed labels built from test S1 (`make_fr_synth.py`).

### 27 Sep morning: v6fr3 = 0.984381; structural signals and test-like calibration
All numbers are out-of-fold on val + wide (US/India) for the v6fr3 stack (`s2_v6_rich_ce_ce2_ce3fr_ce4fr`), baseline 0.98796.
- **No leaks:** entity IDs are uniformly random; row order is unrelated to the owner, and siblings are scattered. Injected name numbers ("#64834", "(ID: 4904)") occur in 0.65% of records and are unrelated to the S1 ID.
- **Empty addresses are independent per record** (per-S1 counts match a binomial exactly), so there is no entity-level signal.
- **Per-source record counts follow a fixed prior** (S2: 0 / 1 / 2 / 3 / 4 = 13% / 36% / 30% / 15% / 5%; S2 and S3 independent given non-singleton).
- **Round-2 stage 2** (`s2_round2.py`): features recomputed from round-1 decisions, plus rival per-source counts, a count-prior likelihood ratio and twin consistency. Result 0.98800 (+0.00004). Not adopted.
  - Missed empty-address records are genuinely ambiguous (e.g. 6 identical "Future Exports Private Limited" S1s). The count prior picks the right owner in some cases and the wrong one in others.
- **Confident pairs (stage-1 p ≥ 0.99, which bypass stage 2):** 0.03% wrong. A perfect veto oracle gives +0.00025 on US/India, so any gain from re-scoring confident pairs must come from France.
- **France, label-free:**
  - Predicted per-source count distributions match US/India (empty 5.8%, mean S2 1.61 vs 1.63). France is not grossly over- or under-matching, so its loss is swaps.
  - Suspicious hub swaps (another S1 at the same address matches the name at least as well): 0.0064 per S1 vs 0.0014 India and 0.0006 US. That's at most about 0.004 of France F.
  - France blocking looks normal: named records with an address and no candidate, 11.6% vs 16.5% US and 9.5% India.
- **Test-like orphan density changes the best threshold.** Validation holds far more orphan-like records than test: dropped S1s (19% simulated vs about 9% US and 2% India on test), plus records whose owner lives outside the universe (on test their S1 is present). Removing them to test levels removes 30% of the band negatives and none of the positives.
  - Best threshold at test-like density: 0.575 (variant A, both kinds at test rates, +0.0003) to ≤ 0.5 (variant B, other-region owners removed, +0.0008).
  - Per country: **US 0.60** (+0.0002 to +0.0006), **India 0.50** (+0.0005 to +0.0009).
  - Retraining round 2 at test-like density adds only +0.0001 over re-thresholding.
  - → candidate **v6fr3t** (US 0.60, India 0.50, France unchanged 0.725): expected +0.0003 to +0.0006 on the leaderboard. Changes 1.1% of US and 1.4% of Indian S1s.
- **CatBoost stage-2 partner** (`s2_catboost.py`, same 132 features and folds): CatBoost alone 0.98797 vs LightGBM 0.98796. Average with LightGBM: 0.98800 plain, +0.00003 at test-like density (A 0.98926 vs 0.98923). Same verdict as seed bagging: the learners agree, and the remaining errors are in the data, not the learner. Not adopted. Test-like per-country best thresholds confirmed: US 0.65 (A) / 0.45 (B), India 0.5 (A) / 0.45 (B).

### Revised plan (supersedes the order below)
- **S2-GBDT:** specialist second-stage model on the grey zone with sibling-support features (H-3 signal), p-space competition and richer pair details. Train on v2a val predictions with 2-fold region CV.
- **S2-XENC (Ved):** cross-encoder fine-tuned on grey-zone pairs (train + val universes) using canonical plus raw text. It scores only the grey zone on test. Its score becomes a feature of S2-GBDT.
- **France:** a stricter France threshold (v1b +0.000289; v1c pending) as a stop-gap. A real fix needs discrimination: S2 models, plus possibly synthetic French pairs generated from test S1 with the known noise operations (to be discussed: transductive, no labels, no external data).

### Order of execution
1. **E7 now.** It's free: re-decide from saved predictions and upload one probe to learn the France direction.
2. **E3 + E2 + E4**, the cheap and certain ones, together as **v2a**. Measure each contribution by ablation.
3. **E1**, the biggest single lever, as **v2b**.
4. **E5 → E6** for France.
5. **E8, E9** if time remains.

### Results log (fill as experiments complete)
| ID | Date | Val F0.5 | Δ vs best | India / US | LB | Decision |
|---|---|---|---|---|---|---|
| v1 | 25 Sep | 0.9769 | — | 0.9725 / 0.9797 | 0.9665 | baseline |
| E4 blocking (WV+OK) | 25 Sep | recall 0.9933/0.9905 → fallback 15: +0.0011, 20: ≈+0.0016, 30: +0.0022 | | | | **fallback 20 adopted** (30 costs +40% compute for +0.0006 more) |
| E7 probe v1b | 25 Sep | France thr 0.85 | | unchanged | **0.966763 (+0.000289)** | **France errors near the threshold are mostly false merges. Go stricter on France** |
| E7 probe v1c | 25 Sep | France thr 0.92 | | unchanged | _pending_ | vs v1b: does stricter still help? |
| E7 probe v1d | 25 Sep | US/India thr 0.82 (France 0.85) | | val −0.0001 | _pending_ | vs v1b: does the LB prefer stricter everywhere? |
