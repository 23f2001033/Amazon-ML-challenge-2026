# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Embedding  
**Team Members:** Aman Kumar Maurya (Team Leader), Reshma G, Afnan Sayyad, Ved Prakash Yadav  
**Submission Date:** 2026-09-28 (final leaderboard submission: 2026-09-27, `v6fr3zF2`)


**Where to find the requested information:**

| Requested item | Section |
|---|---|
| Methodology used | §1 Executive Summary, §2 Methodology (2.1 Problem Analysis, 2.2 Solution Strategy) |
| Candidate Generation / Blocking Strategy | §3 Candidate Generation (canonicalisation, blocking passes, learned pruning, candidate statistics) |
| Model Architecture and Feature Engineering | §4 Matching Model (4.1 stage-1 features, 4.2 cross-encoders, 4.3–4.5 stage 2 and blend, 4.6 decision layer) |
| Any other relevant information | §5 Results & Error Analysis, §6 Conclusion, Appendix A (code and reproduction), B (compliance), C (additional results) |

---

## 1. Executive Summary
The solution is a **cascade: blocking → learned pruning → stacked matcher → decision layer**:

1. **Candidate generation.** Rule-based canonicalisation inverts the observed noise. A multi-pass, region-scoped TF-IDF search then finds about 69 blocking candidates per Source 1 (S1) entity.
2. **Learned pruning.** Two LightGBM pair scorers keep the candidates with probability ≥ 0.02 in either of them: **4.51 per S1** (7.8M pairs on test).
   - `v6`, and `v6t`, trained with test-faithful sample weights.
   - v6 alone keeps 99.56% of the true pairs the blocker found.
3. **Final matcher.** Four fine-tuned multilingual **cross-encoders** (two of them French-aware) and a second-stage LightGBM re-score the uncertain pairs using competition and sibling evidence. For US/India, the stage-2 probabilities of the two stage-1 views are averaged.
4. **Decision layer.**
   - A one-to-one (partition) assignment.
   - Thresholds tuned under test-like conditions: per country, and for US/India also per S1 candidate count.
   - Label-backed rules for zero-shot France: a veto of same-address "sister" records with a swapped category word, and acceptance of acronym and dropped-word copies that the French models under-score.

`candidate_pairs.tsv` is exactly the pruned set of step 2. The final matcher never sees any other pair.

The key ideas:
- A validation design that reproduces test conditions, including a second held-out set of 20 regions.
- Learned "region-leak" blocking. S2/S3 write Telangana addresses with the pre-2014 state name "Andhra Pradesh" and Washington DC as "WA".
- A combined name + address search pass for crowded regions.
- Cross-encoders stacked on competition features ("does another S1 claim this record more strongly?").
- For zero-shot France: French-aware cross-encoders fine-tuned on pairs built from the unlabelled French S1 records, and a label-backed veto of same-address category swaps ("Pornic Club SAS" vs "Pornic Sportive SAS"). The same pattern is only 2–6% true in labelled US/India data.

**Validation macro F0.5 (out-of-fold, labelled US/India held-out regions):**
- 0.9877 (validation regions) and 0.9880 (20 wide regions) at the simulated orphan density;
- **0.9911 in test-like conditions** (orphan records thinned to test density) with the final US/India blend and thresholds.

**Public leaderboard: 0.988574** (final submission `v6fr3zF2`).

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on 2.2M S1, 10.3M S2/S3 records and 7.6M labelled pairs:
- **Synthetic, reversible noise.** S1 is clean. S2 and S3 apply source-specific noise:
  - legal-suffix swaps, accents, leetspeak, website-style names (`latestedebuchclub.com`);
  - `formerly:` / `dba` aliases;
  - dropped or reordered address parts, perturbed house numbers, and empty addresses (about 3% of records).
- **Native scripts.** 24% of India S2 and 13% of India S3 names are in one of 9 Indic scripts. The vocabulary is closed: every test token also occurs in train.
- **Ground truth.** 5.6% of S1 are singletons. Every S2/S3 record belongs to at most one S1, and country always agrees.
- **Name ambiguity.** 40–50% of S1 names are shared (up to 567 × "Meridian"). In France, names are built as "city + generic word", e.g. 530 distinct "Bordeaux Club SAS/SARL/EURL" in one city, so the address and legal form carry the identity.
- **Planted decoys.** Near-duplicates with the same name and street but a shifted house number (1643 → 1647). 99.3% of them are unmatched distractors.
- **Region mislabels in S2/S3 (learned from train labels).**
  - 18.5% of Telangana's true matches carry the old state name "Andhra Pradesh".
  - 77% of Washington DC's matches are tagged "WA".
  - Chandigarh matches are often tagged Punjab.
  - Region-scoped blocking that ignores this loses these pairs completely.
- **Train/test structure.** Test has 5.76 S2/S3 records per S1 vs 4.68 in train, consistent with test S1 being a subsample whose other records remain as orphan distractors. Test adds France (15% of S1, zero-shot).

### 2.2 Solution Strategy
**Approach type:** blocking + learned pruning + stacked matcher (GBDT + cross-encoders) + partition-constrained assignment.

**Core innovations:**
1. **Validation that reproduces test.**
   - Whole (country, region) blocks are held out. Inside them, 81% of S1 are queries and 19% are dropped, with their S2/S3 records left in the pool as orphans. This matches test's distractor density.
   - A second held-out set of **20 regions ("wide")**, including the largest Indian states, predicts the leaderboard much better than a small validation set. It is what exposed the Telangana problem (F0.5 0.876 → 0.980 after the fix).
2. **Learned blocking extensions:** region-leak expansion and a combined name + address pass.
3. **Test-consistent context features.** Competition and ambiguity counts are computed in the same batch sizes and on an S1 set of test's size, so they mean the same in training and test.
4. **Cross-encoders on the uncertain pairs,** stacked with competition features of their own scores.
5. **Test-like decision tuning.** Validation keeps far more orphan-like records than test: dropped S1s (19% simulated vs about 9% US and 2% India measured on test) and records whose S1 lies outside the held-out universe. Thresholds are therefore tuned after thinning those records to test density.
6. **Label-backed rules for the unseen country.** A pattern is vetoed in France only when (a) labelled US/India data shows the same generator pattern is almost always false and (b) France accepts it far more often than US/India do.

---

## 3. Candidate Generation (Blocking + Learned Pruning)

- **Blocking keys used:** all passes are scoped to (country, region), with learned region-leak expansion and a region-less fallback within the country.
  - character 3-gram TF-IDF on the canonical compact name;
  - word/number TF-IDF on canonical address tokens;
  - a combined name + address TF-IDF vector.

  Learned pruning by a stage-1 LightGBM pair scorer (two views) then follows.
- **Candidate pairs generated:** about 120M blocking candidates (about 70 per S1). Pruning cuts them to **7,814,413 pairs in `candidate_pairs.tsv` (4.51 per test S1)**. Every submitted match is inside this set.
- **How we ensured true matches were not lost:**
  - Pair recall was measured at every stage on two labelled held-out region sets: blocking 0.9916 / 0.9908, and pruning keeps 99.56% of those.
  - Each observed miss pattern got a targeted fix: region mislabels → learned leak expansion; crowded identical names → the combined name + address pass.
  - Pruning keeps 99.56% of the true pairs the blocker found. The pairs it drops have stage-1 p < 0.02, far below every decision threshold (lowest 0.35).

**Canonicalisation** (`textnorm.py`, `resources.py`):
- **Names:**
  - aliases resolved (`formerly`, `dba`, `aka`), URLs and handles stripped, leetspeak repaired, accents folded;
  - legal forms canonicalised into a separate field;
  - Indic tokens translated with a dictionary learned from train pairs by position alignment;
  - spelling variants (shri/sri/shree, laxmi/lakshmi) unified.
- **Addresses:** street, saint and French abbreviations normalised (rue, ave, blvd, allée, impasse, route, chemin), PO boxes and null tokens removed.
- **Regions:** explicit region detection for US states, Indian states and codes, native-script state names, and French regions and departments, plus a city→region map learned from S1 records.
- No external data or lookups are used.

**Blocking passes** (`blocking.py`). All passes stay within the same `country` label, which is treated as an open set so France runs unchanged.

| Pass | What it compares | Neighbours kept |
|---|---|---|
| A. Name | char 3-gram TF-IDF on the compact core name, within (country, region) | top 25 |
| B. Address | word/number TF-IDF on canonical address tokens, within (country, region) | top 25 |
| C. Fallback | name pass against region-less records (empty or unparseable addresses), within the country | top 20 |
| D. Combined | name and address TF-IDF vectors concatenated, so ties among identical names are broken by address | top 25 |
| E. Region-leak expansion | queries of region R also search the regions R's true matches leak into | as A, B, D |

- **Why D:** in Maharashtra, half of the missed pairs had the exact S1 name, shared by a median of 35 pool records, plus a truncated address. D raises recall there from 0.963 to 0.985.
- **How E's leak pairs are learned:** only from train labels (`region_leaks.py`: rate ≥ 2% with ≥ 30 pairs, or rate ≥ 0.5% with ≥ 500 pairs; 16 pairs). It raises Telangana's recall from 0.803 to 0.9915.

**Learned pruning** (stage 1, `features.py`, `model.py`):
- LightGBM pair scorers (69 features, below) score every blocking candidate. Only pairs with p ≥ 0.02 go on to the final matcher.
- Two views are used: `v6` and `v6t`, which is trained with test-faithful sample weights (§4.5). A pair is kept if either view gives p ≥ 0.02.
- Pruning keeps 99.56% of the true pairs the blocker found (val and wide). The 0.44% it drops have stage-1 p < 0.02, far below the lowest decision threshold (0.35, the empty-S1 rescue). Stage 2 never re-scores them.

**Candidate statistics:**

| | Validation | Wide (20 regions) | Test |
|---|---|---|---|
| Blocking candidates per S1 | 69.1 | 73.7 | ≈ 70 (68–74 per batch) |
| Blocking pair recall | 0.9916 | 0.9908 | — |
| Candidates per S1 after v6 pruning | 4.16 | 4.14 | 4.20 (7,284,320 pairs) |
| **Final candidates per S1 (union of v6 and v6t pruning, `candidate_pairs.tsv`)** | — | — | **4.51 (7,814,413 pairs)** |
| True pairs kept by v6 pruning | 99.56% | 99.56% | — |
| Recall of the v6-pruned candidate set | 98.72% | 98.64% | — |

- **Reduction ratio** of the final candidate set against the same-country cross product (6.7 × 10¹² pairs): **0.9999988** (7.8M pairs kept).

---

## 4. Matching Model

**Features used:**
- **Name:** TF-IDF cosine; rapidfuzz ratio, token-set, token-sort and partial ratio; Jaro-Winkler and Levenshtein on the compact name; IDF-weighted token containment; legal-form equality; name-ambiguity counts.
- **Address:** TF-IDF cosine; fuzzy ratios; token Jaccard and containments; house-number equality, permutation, dropped digit and numeric gap.
- **Other:** competition ranks and margins among the S1s claiming the same record; sibling support from the S1's other confident records; four cross-encoder probabilities and their margins over the best rival S1.

**Model type:** stage-1 LightGBM (two views) → four fine-tuned multilingual cross-encoders (Multilingual-MiniLM, XLM-R base, bge-reranker-base, bge-reranker-v2-m3) → stacked stage-2 LightGBM → rule-based decision layer with partition assignment.

**Threshold selection method:** macro-F0.5 optimisation on out-of-fold predictions of held-out regions, at test-like orphan density.
- US/India: per country and per S1 candidate-count bucket, cross-fitted between the two held-out sets.
- France (no labels): label-free count-model evidence plus leaderboard feedback.

### 4.1 Stage 1: LightGBM pair scorer (MIT)
- **Features (69, all country-agnostic; the country's identity is never a feature, only within-country counts such as name ambiguity):**
  - **Name:** TF-IDF cosine; rapidfuzz ratio, token-set, token-sort and partial ratio; Jaro-Winkler and Levenshtein on the compact name; IDF-weighted token containment in both directions; legal-form equality; name ambiguity.
  - **Address:** TF-IDF cosine; fuzzy ratios; token Jaccard and asymmetric containments.
  - **House number (decoy defence):** equality, digit permutation, dropped digit, last-digit equality, log numeric gap, number-set Jaccard and containments.
  - **Context and competition:** the pair's rank and score gap among the S1's candidates, and among all S1s competing for the same record; number of competing S1s.
- **Test-consistent computation.** Context features use the same batch size as test (100k S1). Name-ambiguity counts are taken on a per-country S1 sample of test's size. An earlier attempt that dropped these features instead lost 0.002 on the leaderboard.
- **Training:**
  - 45 held-out-free regions (957k S1 queries, 28.4M presampled pairs, 3.3M positives).
  - Easy negatives (blocking rank > 15) are subsampled at 25% with weight 4, so probabilities stay calibrated.
  - 127 leaves, learning rate 0.05, up to 3,000 rounds with early stopping on an S1-level holdout.

### 4.2 Cross-encoders (MIT / Apache-2.0, all ≤ 568M parameters)

| Model | Base checkpoint | Input text | Training pairs | Scores |
|---|---|---|---|---|
| CE v1 | `microsoft/Multilingual-MiniLM-L12-H384` (117M, MIT) | canonical name ; canonical address | 2.23M (positives + 12 hardest negatives per S1, 10% of regions) | every final candidate (p ≥ 0.02) |
| CE v2 | `FacebookAI/xlm-roberta-base` (278M, MIT) | canonical | 3.04M from 45 regions (8 hardest + 2 combined/leak-only negatives) | uncertain band |
| CE 3-FR | continued from CE 3 = `BAAI/bge-reranker-base` (278M, MIT) fine-tuned on 1.3M raw US/India pairs | **raw** name \| address | 828k French pairs (below) + 450k US/India raw pairs | uncertain band |
| CE 4-FR | `BAAI/bge-reranker-v2-m3` (568M, Apache-2.0), gradient checkpointing | raw | 800k of the same mix | uncertain band |

- **French pairs without labels** (`make_fr_synth.py`), built only from the test France S1 records, which are clean reference entities:
  - positives: an S1 against a copy of itself passed through the observed noise operations;
  - negatives: two *different* S1 records at the same address, with the same core name, or sharing the first word;
  - decoys: an S1 against a house-number-shifted copy.
- No test labels exist or are used, and no external data. French hold-out AUC is 0.9997. Swapping CE 3 for CE 3-FR + CE 4-FR raised France's F0.5 by about 0.005 on the leaderboard. US/India were unchanged.

- **Training setup:** 1 epoch, binary cross-entropy, fp16, Kaggle T4 GPUs.
- **Leakage control:** no cross-encoder was trained on validation or wide regions, so their scores on both are out-of-sample.
- **Stand-alone quality:** on 994k validation pairs, CE v1 has AUC **0.9948** vs 0.9945 for stage 1, a comparable model built from completely different evidence.

### 4.3 Stage 2: specialist re-scoring of the uncertain band (LightGBM)
- **Scope:** pairs with stage-1 p in [0.02, 0.99), about 1.1 per S1. Pairs outside the band keep their stage-1 score.
- **Features:**
  - all stage-1 features;
  - **p-space competition:** best rival S1's probability for the same record, the margin over it, and counts of confident claimants;
  - **sibling support:** agreement with the S1's other confident candidates (same house number, similar name or address);
  - direct pair checks;
  - for every cross-encoder: its probability, logit, **margin over the best rival S1**, the S1's best other candidate, confident-claimant counts, and rank within the S1.
- **Training:** the out-of-sample bands of both held-out sets (validation 220k + wide 713k pairs). 4 region folds give out-of-fold scores; the final model is refit on all folds.
- **Most important features:** CE margin over the best rival, stage-1 margin over the best rival, stage-1 probability.

### 4.4 Stage 2 retrained for test conditions (`s2_tune.py final w`)
- **The shift:** 37% of the stage-2 training pairs involve orphan-like records, i.e. records whose true S1 was dropped from the held-out universe or lies outside it. On test such records are rare: about 9% US and 2% India, measured via identical-record groups.
- **The fix:** the final stage 2 is trained with those pairs down-weighted to the test density.
  - Out-of-fold, test-like: 0.98938 vs 0.98925, with India improving most.
  - The recalibrated model needs thresholds of US 0.725 and India 0.65.
  - On the leaderboard, together with the decision rules below, it gave **+0.00093 (0.987206 → 0.988137)**, more than validation predicted. The training/test shift is larger than the simulation captures.

### 4.5 Second, test-faithful stage 1 and blend (`blend_preds.py`, US/India)
- **v6t:** a second stage 1, trained like v6 but with the test-faithful weights also applied at stage 1 (`ER_S1_TESTW=1`). Records whose S1 was dropped from a training universe keep only their test density (US 0.47, India 0.105); records owned outside the universe get 0.05.
- It keeps different pairs than v6. Its 402k new val/wide and 643k new test band pairs were scored by all four cross-encoders (`export_ce_new.py`, `score_new_kaggle.py`: inference only, same checkpoints).
- **Stage 2 on v6t alone** is level with v6 (test-like 0.99101 vs 0.99096). **The mean of the two stage-2 probabilities is better: 0.99107** (US 0.99132, India 0.99078; thresholds US 0.65, India 0.675), because the two stage-1 views make different pruning errors.
- **Used for US/India only.** For France, v6t raised the count-model estimate of extra records from 0.0102 to 0.0123 per S1 with no recall gain, so France keeps the v6 stage 2 'w'.
- Leaderboard: +0.000152 (0.988316 → 0.988468), together with the final swap veto below.
- The candidate set is the union of the two pruned sets: about 4.5 pairs per S1.

### 4.6 Decision (`final_decide.py`)
- **Partition constraint:** each S2/S3 record goes only to its highest-probability S1, as in the ground truth.
- **Thresholds, tuned at test-like density.** The held-out simulation holds far more orphan-like records than test. It has 19% of S1s dropped vs about 9% (US) and 2% (India) measured on test via identical-record groups, plus records whose S1 lies outside the universe. The global threshold (0.725) that is optimal as simulated is therefore too strict. Thresholds are tuned after thinning those records to test density.
  - First upload with it: US 0.60, India 0.50, expected +0.0004. They were uploaded together with the France veto (combined +0.0028), so their own effect is not separately measured.
  - **Final (`v6fr3zF2`), US/India:** per S1 candidate-count bucket, on the blended stage-2 probabilities (see *Thresholds per S1 candidate-count bucket* below).

    | Bucket (pruned candidates of the S1) | US | India |
    |---|---|---|
    | 1 | 0.50 | 0.50 |
    | 2–3 | 0.70 | 0.65 |
    | 4–6 | 0.725 | 0.70 |
    | 7–10 | 0.675 | 0.70 |
    | 11+ | 0.65 | 0.675 |
  - **Final, France: 0.50.** It moved from 0.725 to 0.60, 0.55 and 0.50 over the uploads. Each of those uploads raised the leaderboard, although each also carried other changes. A label-free count model (observed per-source record counts fitted as a thinned generator distribution plus extra records) estimated each step as more recall at few extra false records.
- **France same-address category-swap veto.** An accepted pair is removed when:
  - the S2/S3 record sits at the S1's exact address (same house number, address token-set similarity ≥ 90);
  - the record drops an S1 word;
  - the record adds a *different real business-name word*, i.e. a word used by ≥ 20 S1 names that is not a generator noise word, not a typo, and not an abbreviation of the dropped word.

  Example: "Pornic Club SAS" → "Pornic Sportive SAS".
  - Generator noise words are learned per country as the words over-represented in S2/S3 names vs S1 names (e.g. *Groupe, Développement, Holding*).
  - **Evidence:** in labelled US/India data this pattern (a "sister business" at the same address) is **2–6% true**, and the models reject it there (0.0006–0.0009 accepted pairs per S1).
  - Training regions: 2.2% India, 3.7% US.
  - Held-out wide regions, exact final rule: 5.8% India, 5.6% US.
  - French names are "city/person + category word", so the US/India-trained models accepted it **about 100× more often** (0.078 pairs per S1, 76% of them at stage-1 p ≥ 0.99, i.e. never re-scored).
  - The per-source record counts of the affected S1s match the "extra record" hypothesis, not the "true copy" one.
  - Removing the 20,153 pairs raised the leaderboard from 0.984381 to 0.987206 (France about +0.016 F0.5).
- **France "dropped word + noise suffix" copies.** Rejected best-candidate French pairs at an exact or close address, whose record only drops S1 words and adds generator noise words, are accepted. In labelled US/India held-out data this pattern (uncertain band) is 98–99% true (0.992 at an exact address, 0.984 at a close one), and the US/India models rarely reject it (0.12 rejected per 1,000 S1). France rejected it 12.6 times per 1,000 S1, about 100× as often. The French cross-encoders under-score it because their synthetic training lacked it.
- **Empty-S1 rescue.** An S1 left empty takes its best still-unassigned candidate if p ≥ 0.35 (+0.0001 on test-like US/India held-out).
- **Acronym copies (all countries; found in a manual audit of French borderline pairs).** Rejected best-candidate pairs whose record name is the acronym of the S1 name ("Manuel Benkadi Amicale SAS" → "MBA") at an exact or close address are accepted. In labelled US/India data every such pair is true (4,853/4,853 accepted, 30/30 rejected). France rejected about 950 of them, because the French cross-encoders under-score initials. A manual audit of French examples agreed (9/9).
- **Thresholds per S1 candidate-count bucket (US/India).** An S1 with a single pruned candidate uses 0.50; S1s with 2–6 candidates use 0.65–0.725. Tuned on the val regions and scored on the wide regions, and the reverse: +0.00005, positive in all four country × direction cells. France uses one threshold, 0.50. The upload with 0.55 scored 0.988572; the final upload with 0.50, which also extended the acronym rule to US/India, scored 0.988574.
- **Final swap veto.** The category-swap veto is re-applied to the final decisions. It catches swaps accepted between the base threshold (0.725) and the final France threshold, and records whose next-best S1 took over after a veto: 927 pairs at 0.60, and 1,555 at 0.50 (measured before the acronym rule was added).
- Evaluated and not adopted: per-S1 expected-F0.5 subset selection (+0.0001); global or count-aware assignment (+0.00004); a CatBoost stage-2 partner (+0.00003).

---

## 5. Results & Error Analysis

- **F0.5 score (macro), best validation:** 0.9911 in test-like conditions (US/India held-out, out-of-fold); 0.9877 / 0.9880 at the simulated orphan density. **Public leaderboard: 0.988574.**
- **Common false positives (wrong merges):**
  - orphan records whose own S1 is absent (mostly empty-address records), attached to a same-name S1;
  - planted house-number decoys (same name and street, shifted number);
  - in France, same-address "sister" businesses with a swapped category word (now vetoed).
- **Common false negatives (missed matches):**
  - empty-address records whose name is shared by several S1s (about 5 on average), which are genuinely ambiguous;
  - records the blocker never retrieves, mostly Indian addresses truncated to "unit number, city";
  - in France, copies the French models under-score (acronyms, dropped-word copies; now rescued).

| Version | Main change | Validation | Wide | Leaderboard |
|---|---|---|---|---|
| v1 | canonicalisation + TF-IDF blocking + LightGBM | 0.9769 | — | 0.966474 |
| v2b | + stage-2 re-scoring | 0.9824 | — | 0.967982 |
| v3brc | + cross-encoder v1 in stage 2 | 0.9850 | — | 0.975330 |
| v6r | + region-leak + combined blocking, 45-region training, stage 2 on val+wide | 0.9834 | 0.9836 | 0.977243 |
| v6rc | v6r + cross-encoder v1 | 0.9867 | 0.9871 | — |
| v6rc3 | + cross-encoders v2 and 3 (raw text) | 0.9875 | 0.9880 | 0.983658 |
| v6fr3 | CE 3 → French-aware CE 3-FR + CE 4-FR | 0.9877 | 0.9880 | 0.984381 |
| v6fr3s | + test-like per-country thresholds + France category-swap veto | 0.9892 (test-like) | — | 0.987206 |
| v6fr3w | stage 2 trained with test-like weights; drop-word French copies; empty-S1 rescue | 0.9894 (test-like) | — | 0.988137 |
| v6fr3x | US/India from stage 2 'w2' (records whose owner lies outside the held-out regions, an artifact of the region split, also down-weighted; thr 0.70); France threshold 0.60 | 0.9910 (test-like B) | — | 0.988316 |
| v6fr3z | US/India: mean of stage 2 on two stage-1 views (v6 + test-faithful v6t, all four cross-encoders on the new pairs); France: final swap veto | 0.9911 (test-like B) | — | 0.988468 |
| v6fr3zbB | France acronym rescue + France threshold 0.55; US/India thresholds per candidate-count bucket | 0.9911 (test-like B) | — | 0.988572 |
| **v6fr3zF2** | **France threshold 0.50; acronym rescue in all countries** | 0.9911 (test-like B) | — | **0.988574** |

- **Remaining errors** (wide, v6rc):
  - Most of the loss is **missed matches**. 79% of missed-but-scored pairs have an **empty address**. Their S1 name is typically shared by about 5 S1s in the country, and only 6–8% can be singled out by raw name or legal form. These are ambiguous for any method.
  - Wrong matches are mainly **orphans of dropped S1s** (empty-address records whose own S1 is absent) and planted house-number decoys.
- **Leaderboard decomposition.** A diagnostic upload with every France row left empty gave the leaderboard score of US+India alone: **US/India ≈ 0.989 on test** (above validation, as expected with fewer orphans), and France ≈ 0.953 at that time. France is 15% of test S1.
- **France:** zero-shot, with no labels. Its names are generic and shared addresses are common (13.9% of S1 vs about 5%).
  - The French-aware cross-encoders lifted it to about 0.958.
  - The category-swap veto lifted it to about 0.974.
  - The test-weighted stage 2 and the dropped-word and acronym rescues lifted it to about 0.977–0.979. Only the total is measured; the estimate assumes US/India ≈ 0.990–0.991.
- **Oracle error budget** (US/India held-out, test-like; each class fixed alone with the truth):

| Error class | Ceiling |
|---|---|
| Blocking misses with an address (mostly Indian records truncated to "unit number, city") | +0.0018 |
| Ambiguous empty-address records | +0.0016 |
| Blocking misses without an address | +0.0012 |
| False positives | ≤ +0.0009 per class |

  A record-centric address-containment blocking pass targeting the truncated addresses raised wide recall only from 0.9908 to 0.9913 and did not change stage-1 F0.5, so it was not adopted.

---

## 6. Conclusion
- **Validation design mattered most.** A small validation set hid a region-labelling quirk that cost Indian regions up to 0.12 F0.5. A larger held-out set exposed it, and fixing it in blocking closed the validation-to-leaderboard gap.
- **Blocking must be learned, not just engineered.** Labels reveal where noise moves records across regions, and crowded regions need address-aware tie-breaking.
- **Text models add what features can't.** Cross-encoders reading the pair directly add +0.004–0.005, most of all in the zero-shot country. They pay off most when their scores enter competition features between S1s.
- **Zero-shot transfer needs generator-level reasoning.** The biggest France gain came from recognising a distractor family (same-address category swaps) that labelled US/India data shows is almost always false (2–6% true). US/India-trained models accept it only because French names are built differently. Each France rule (the veto, dropped-word copies, acronym copies) was admitted only with labelled US/India evidence and a clear France-vs-US/India rate difference. The France threshold, which has no labels, came from label-free count-model evidence and leaderboard feedback.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains:
- `src/`: all source code;
- `models/`: the exact stage-1 and stage-2 models and the decision configs behind the submitted files;
- `README.md`: exact end-to-end commands;
- `requirements.txt` and `requirements-gpu.txt`: pinned versions.

The entry points that reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv` (README §3):
1. `prepare.py`: raw TSV → canonicalised parquet. Dictionaries are learned from train.
2. `run_test.py` with `models/lgbm_v6.txt` and with `models/lgbm_v6t.txt`: test blocking, features and both stage-1 views. It writes `test_pred_{v6,v6t}` and the uncertain-band features.
3. Cross-encoder scores (GPU):
   - `ce_train_infer.py`, `ce2_train_infer.py`, `ce3_train_infer.py`, `ce_fr_train_infer.py`, with their training pairs from `export_ce_*.py` and `make_fr_synth.py`;
   - pairs that only v6t keeps: `export_ce_new.py` → `score_new_kaggle.py` → `export_ce_new.py merge`.
4. `s2_tune.py final w` / `final w2`, with `ER_S2_LOAD=../models`, which loads the shipped stage-2 boosters: stage-2 test probabilities for v6 (w, w2) and v6t (w2).
5. `blend_preds.py`: US/India = mean of the two w2 predictions; France = v6 'w'.
6. `final_decide.py ../models/v6fr3zF2_final.json <root>/output`: the decision layer. It reproduces the uploaded file byte-for-byte (md5 `83a25eaa473c18cceec1f48084ef4acf`).

Training from scratch (README §4):
- `run_dev.py block | features | train`: universes, blocking, stage-1 training; v6t adds `ER_S1_TESTW=1`.
- `eval_regions.py`: the wide held-out check.
- `s2_tune.py dev`: out-of-fold stage 2 and round counts.
- The analysis scripts behind every decision rule are listed in README §6.

### B. Compliance
- No external data, APIs, geocoding or lookups. All dictionaries, region maps and leak pairs are learned from the provided training data or are generic abbreviation tables.
- **Models and licences:**

| Model | Licence | Size |
|---|---|---|
| LightGBM | MIT | — |
| Multilingual-MiniLM-L12-H384 | MIT | 117M |
| XLM-RoBERTa-base | MIT | 278M |
| bge-reranker-base | MIT | 278M |
| bge-reranker-v2-m3 | Apache-2.0 | 568M |

  All are far below the 8B-parameter limit.
- **Compute:** CPU work on a 16-vCPU cloud VM used as compute only; GPU fine-tuning on Kaggle T4.

### C. Additional Results
All numbers below are test-like (orphan records thinned to test density) on the labelled held-out regions (US/India).

- **Two-view stage-2 blend** (`blend_eval.py`):

  | Stage-2 input | Macro F0.5 |
  |---|---|
  | v6 | 0.99096 |
  | v6t | 0.99101 |
  | Mean of both | **0.99107** (US 0.99132, India 0.99078) |

- **Thresholds per S1 candidate-count bucket** (`bucket_thr.py`), cross-fitted (tuned on val, scored on wide, and the reverse):

  | Direction | US | India |
  |---|---|---|
  | val → wide | +0.00004 | +0.00006 |
  | wide → val | +0.00008 | +0.00002 |

  Overall: 0.99106 → 0.99111. Single-candidate S1s want a lower threshold (no rival claimant); S1s with 2–6 candidates want a stricter one.
- **Leave-one-country-out stage 2** (`loco.py`, our proxy for zero-shot France):
  - Trained on India, scored on US: best threshold moves 0.50 → 0.80, loss at the best threshold 0.0003.
  - Trained on US, scored on India: best threshold moves 0.65 → 0.50, loss 0.0006.
  - So transfer loses little at the right threshold, but the right threshold moves unpredictably. That is why France's threshold was set from label-free evidence and leaderboard feedback.
- **France pattern audit** (`fr_cells2.py`): accepted and rejected French pairs grouped by name-edit type × address relation, compared with the labelled US/India truth rate of the same cell.
  - Cells used for rules: category swaps (2–6% true: vetoed); drop + noise-word copies (98–99% true: accepted); acronyms (100% true: accepted).
  - Cells deliberately left alone: same name with the house number dropped (45–67% true).
- **Measured and not adopted:**
  - CatBoost stage-2 partner (+0.00003);
  - seed bagging (+0);
  - larger stage-2 LightGBM (±0);
  - zero-shot LLM judge on uncertain pairs (≤ 8B, Apache-2.0; AUC 0.51 vs 0.84 for stage 2);
  - containment / record-centric blocking (recall +0.0005, F unchanged);
  - hub-address and house-shift vetoes (US/India analogues 88–99% true);
  - count-prior second round (+0.00004).

