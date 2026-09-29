You are an expert in entity resolution and competition machine learning. Please help me close a specific gap in a live competition. Read everything below. Then give me concrete, testable ideas, ranked by expected leaderboard gain, that can be built and run in ≤ 10 hours with the resources listed. For each idea, say *why* it would gain on the leaderboard specifically, and how to validate it before spending an upload. Be critical: many "obvious" ideas are already measured as zero-gain (section 7).

# 1. Competition
- **Task:** Amazon ML Challenge 2026, Business Entity Resolution. Given business records from three sources (S1 = clean reference; S2, S3 = noisy), find for every S1 record all S2/S3 records of the same real business.
  - An S1 entity may have 0 or many matches.
  - Each S2/S3 record belongs to at most one S1 (a partition).
  - Fields: `entity_id`, `business_name`, `business_address`, `country`.
- **Train:** US + India, with ground truth. 2.2M S1, 11.3M S2+S3, 7.6M labelled pairs.
- **Test:** US 663k S1, India 810k, **France 259k (15%, never seen in training)**. Pool: 9.97M S2+S3.
- **Metric:** macro F0.5 per S1 entity, averaged over all test S1.
  - Singletons score 1 if the prediction is empty, else 0.
  - A non-singleton with an empty prediction scores 0.
  - The public leaderboard is a random subset of test.
- **Submission:** `matching_results.tsv`, one row per test S1. `candidate_pairs.tsv` (the exact set the final model runs inference over) is also reviewed; a smaller candidate set per S1 ranks higher in the final evaluation.
- **Rules:** no external data, APIs, geocoding or lookups. The final model must be MIT/Apache and ≤ 8B parameters.
- **Deadline:** today, ~14 hours left. **4 leaderboard uploads left.**
- **Leaderboard:**
  - Top 3 are **0.9915, 0.9910, 0.9909**; rank 13 ≈ 0.990.
  - **We are at 0.984381 (rank ~300).**
  - Target: **≥ 0.992**.

# 2. Measured split of our score (important)
- We uploaded our best model with every France row emptied (score 0.849373) next to the full version (0.983658). Hence:
  - **US + India on test ≈ 0.989**, slightly *above* our held-out validation (0.988).
  - **France ≈ 0.953 → 0.958 now**, bounded ≤ 0.975 for any plausible French singleton rate.
- So **France (15% weight) costs ~0.005**. **US+India at 0.989 vs the top teams' ~0.992 costs ~0.0025.** We must fix both.
- Test has far **fewer "orphans"** (records whose own S1 is absent from test) than our validation simulation: ≈ 9% US, 2% India, 2.5% France vs the simulated 19%. That explains US+India test > validation. Re-tuning thresholds for this changes nothing; the optimum stays 0.70.

# 3. Data facts (EDA)
- **S1 is clean.** S2/S3 noise:
  - legal-suffix swaps/drops, case changes, accents, leetspeak, typos;
  - website-style names (`acme.com`), `formerly`/`dba` aliases;
  - native Indic scripts: 24% of India S2 names, in 9 scripts, a closed vocabulary;
  - dropped or reordered address parts, perturbed house numbers (leading zeros, ranges), empty addresses (~3%);
  - "renamed" records with a pseudo-word name at the exact address ("urban marketing" → "nylaonyx"; 67% of such exact-address + unrelated-name pairs are true in train).
- **5.6% of S1 are singletons.** S1-only features cannot predict singletons (AUC 0.50).
- **Planted decoys:** same name and street, shifted house number (1643→1647). 99.3% are distractors.
- **Name ambiguity:** 40–50% of S1 names are shared (up to 567 S1s share the core name "meridian").
- **Record-only attributes separate true records from distractors with AUC 0.78.** Distractors almost always have a full address with a house number and longer composite names. Records without a house number are 97% true.
- **Identical "twin" records** (same canonical name + address + house), 22.6% of the pool, belong to the same S1 99.7% of the time.
- **Records of one entity don't share raw name spellings.** Matched records with an identical raw name belong to the same S1 only 39% of the time. For an empty-address record, a raw-name twin that has an address points to its true S1 only 56% of the time.
- **S2 and S3 record counts per entity are nearly independent.**
- **Region mislabels in S2/S3** (learned from train labels; we fixed them with a "leak expansion"):
  - Telangana addresses written as "Andhra Pradesh" in 18.5% of pairs;
  - Washington DC → "WA" in 77%.
- **France** (unlabelled; everything below is observed on test):
  - Names are "city + generic word + legal form" ("Bordeaux Club SARL"; 530 S1s share the core name "Bordeaux Club", 486 share "Nantes Club").
  - Only 3 regions.
  - **13.9% of French S1 share an address with another S1** (US/India ~5%). There are "hub" addresses where dozens of associations sit at one building, each named e.g. "X Comité" / "Y Amicale".
  - French noise: street abbreviations (R./AV/BD/Imp), "N°"/"#" prefixes, department names ("Gironde", "Nord") instead of regions, legal forms SAS/SARL/EURL/SCI/SA/SASU/EI swapped, "& Fils" / "Cie" / "Groupe" / "(France)" / "& Associés" suffixes, pseudo-word renames.
  - Our canonicaliser handles all of these correctly (verified).
  - The test pool has 5.53 records per S1 in France vs 5.76–5.82 in US/India.
  - We predict 3.32–3.38 matches per S1 in France vs 3.36–3.37 in US/India.

# 4. Our pipeline (current best, 0.984381)
1. **Canonicalisation:** names → core/compact/legal/native/alias/web fields, with an Indic→English token dictionary learned from train pairs. Addresses → tokens, house number, number set, region detection (US states, Indian states/codes, French regions/departments, city→region map).
2. **Blocking,** region-scoped, top-K per pass:
   - char-3gram TF-IDF on names, K=25;
   - word TF-IDF on addresses, K=25;
   - a combined name+address pass, K=25;
   - a region-less fallback, K=20;
   - a learned region-leak expansion.

   Result: ~69 candidates/S1, recall 0.9916 (val) / 0.9908 (20 held-out "wide" regions).
3. **Stage 1:** LightGBM pair scorer (69 features: name/address fuzzy + TF-IDF, IDF-weighted containment, house-number features, legal form, ambiguity counts computed test-consistently, competition ranks among the S1's candidates and among S1s competing for a record).
   - Trained on 45 regions (957k S1 queries, 28M pairs).
   - Validation 0.9806.
   - **Learned pruning:** pairs with p ≥ 0.02 are kept (4.2/S1, 99.56% of true pairs) and form `candidate_pairs.tsv`.
4. **Cross-encoders on the uncertain band** (0.02 ≤ p < 0.99, ~1.1 pairs/S1):
   - CE v1: Multilingual-MiniLM-L12, cleaned text, scores all p ≥ 0.02 pairs.
   - CE v2: XLM-R base.
   - **CE 3-FR / CE 4-FR:** bge-reranker-base / bge-reranker-v2-m3 on **raw text**, fine-tuned on US/India pairs **plus 828k French pairs with guaranteed labels built from the test France S1 records**:
     - S1 vs its synthetic noisy copy = match;
     - two different S1 at the same address / same name / same first word = non-match;
     - shifted house = decoy.
   - French synthetic hold-out AUC is 0.9997. On real US/India band pairs they're about as good as the original CE 3 (band AUC ~0.94).
5. **Stage 2:** LightGBM re-scores the band.
   - Features: stage-1 features; p-space competition (best rival S1's p for the same record, margin); sibling support (agreement with the S1's confident records); per-CE score, logit, **margin over the best rival S1** (the most important features), confident-claimant counts, rank in S1.
   - Trained on the out-of-sample bands of two held-out sets (val 220k + wide 713k pairs), 4 region folds.
   - Out-of-fold macro F0.5: val 0.9877, wide 0.9880.
6. **Decision:** partition (each record to its highest-p S1), global threshold 0.725.

# 5. Leaderboard history (what moved it)
| Version | Change | LB |
|---|---|---|
| v1 | TF-IDF blocking + LightGBM | 0.9665 |
| v2b | + stage-2 band re-scoring | 0.9680 (+0.0015) |
| A | removed count/competition features ("transfer-safe") | 0.9661 (worse) |
| v3brc | + cross-encoder v1 in stage 2 | 0.9753 (+0.007) |
| v6r | India blocking fix (Telangana→"Andhra Pradesh" leak, combined pass), 5× training regions, no CE | 0.9772 |
| v6rc3 | v6 + CE v1/v2/3 | 0.98366 |
| **v6fr3** | CE 3 replaced by French-aware CE 3-FR + CE 4-FR | **0.98438** (France +0.005) |

# 6. Where the remaining loss is (held-out val+wide regions, 634k S1, US+India, labelled; v6rc stack at 0.9871, total loss 0.0129)
- **Missed matches dominate.** S1s missing ≥ 1 record cost 0.0076; S1s with everything missed cost 0.0029. False positives cost 0.0018 (+0.0004 on singletons).
- **Missed pairs, 70.5k:**
  - **50k scored but below threshold.** 79% have an **empty address**; 58% have the same compact name as the S1. Median p is 0.22.
  - **20k not retrieved by blocking** (43% empty address).
- **Empty-address misses are mostly ambiguous.** The record's name is shared by a median of 5 S1s in the country; only 6–8% can be singled out by raw name or legal form.
- **Wrong pairs, 6.2k:**
  - 3.7k orphans of dropped S1s (76% empty address, median p 0.89);
  - 2.0k distractors or planted decoys;
  - 0.5k records belonging to another S1.

# 7. Tried and measured as ~zero gain (don't re-suggest without a new angle)
- Threshold / per-country threshold tweaks (±0.0003).
- Expected-F0.5 per-S1 set selection (+0.0001).
- Seed-bagged stage 2 (+0).
- Wider blocking K=40 (+0.0003).
- More cross-encoders of the same view (MiniLM retrained on more data: +0).
- Shared-address stage-2 features (neighbours at the S1's address, best-neighbour name match, name similarity without the city word): +0.
- Record-prior feature (record-only distractor model, AUC 0.78): +0.
- France-only veto/add rules driven by the French cross-encoders: touch <1% of French pairs.
- Removing count/competition features (−0.002 on the leaderboard).

# 8. Resources
- **CPU:** a 16-vCPU / 61 GB Linux VM with all intermediate data (candidates, features, band predictions, all CE scores for val/wide/test). A full stage-2 retrain plus test takes ~30 min; a full stage-1 rebuild takes ~4 h.
- **GPU:** Kaggle T4×2, one committed run at a time plus one interactive, ~10 GPU-hours left. A base-size cross-encoder scores ~1,000 pairs/s; the large model ~450 pairs/s.
- **Time and uploads:** ~14 hours, 4 uploads today (one should be reserved as a final safety upload).

# 9. What I need from you
1. **The most likely reason the top teams reach ~0.992 on US+India when we're at 0.989 with this pipeline.** What structural signal or method are we missing? Consider:
   - how the generator creates records (true noisy copies vs distractors vs orphans);
   - global or cluster-level assignment instead of pairwise-then-greedy;
   - the empty-address ambiguity;
   - how records of one entity relate to each other.
2. **The fastest way to lift France from ~0.958 to ~0.99 without French labels.** Consider hub addresses with many same-pattern names, and zero-shot transfer from US/India. Is there a principled, label-free way to validate a France change before uploading?
3. **For each idea:** expected gain, implementation steps (compatible with the pipeline above), runtime, and a validation plan.

Be specific (features, algorithms, thresholds). Assume we can implement anything in Python (polars, LightGBM, PyTorch/transformers).
