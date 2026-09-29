# Decision Log

Every decision we take is recorded here with its **reason** and the **evidence** behind it. Entries are append-only. If a decision is reversed, add a new entry that references the old one instead of editing history.

Format: `D-### | date | area | status (ACTIVE / SUPERSEDED / TENTATIVE)`

---

### D-001 | 2026-09-25 | Tooling | ACTIVE
**Decision:** Convert all raw TSVs to Parquet (zstd) once (`eda/00_convert_to_parquet.py`) and use **Polars** for data work.
**Reason:** About 2.4 GB of TSV and 24M rows on a 16 GB RAM machine. Parquet loads in about 1 s instead of about 30 s per file, and Polars uses far less memory than pandas for string-heavy frames.
**Evidence:** Every file converted with `rows == physical lines - 1`, so no rows were split or merged by parsing.

### D-002 | 2026-09-25 | Parsing | ACTIVE
**Decision:** Read TSVs with **quoting disabled** (`quote_char=None`), all columns as strings, and empty fields as null.
**Reason:** Names contain `"` and `'` characters (for example `Callahan's`). Quote handling could merge rows or strip characters. The row-count check (D-001) confirms the parse is exact.

### D-003 | 2026-09-25 | Environment | ACTIVE
**Decision:** Add `polars`, `unidecode` (GPL-2+, used only as an offline text utility and not as a model) and `sparse_dot_topn` (Apache-2.0) to the environment.
**Reason:** Polars for scale. Unidecode to fold accents and romanise Indic scripts for similarity. `sparse_dot_topn` gives fast top-k sparse cosine for TF-IDF blocking.
**Note:** The license rule (MIT or Apache-2.0) applies to the **model**. Utility-library licenses will still be listed in the final README. If there is any doubt about unidecode's GPL license, D-012's learned dictionary removes the need for it on Indic scripts. Accent folding can be done with stdlib `unicodedata`.

### D-004 | 2026-09-25 | Validation | ACTIVE
**Decision:** Build validation by **holding out S1 entities** (with all their GT pairs). Search against the **full train S2+S3 pool**, not only the held-out entities' matches.
**Reason:** The metric is per-S1-entity macro F0.5. A held-out entity must face the same distractor density as in test. A pool that holds only the fold's matches would inflate precision.
**Evidence:** 26% of S2/S3 records match no S1 (distractors). In test the S2+S3/S1 ratio is 5.76 vs 4.68 in train, so distractors are, if anything, denser in test (see EDA §3).

### D-005 | 2026-09-25 | Leakage | ACTIVE
**Decision:** Do **not** use entity-ID numbers or file row order as features.
**Reason/Evidence:** EDA §9: the correlation between S1 and match ID numbers is 0.0001, and between S1 and match row positions it is 0.0001. The 15.9k coincidental overlaps of ID numbers match chance (≈2.2M × 7.6M / 1e9). There is no leak, and using IDs would also be unjustifiable in the audit.

### D-006 | 2026-09-25 | Blocking | ACTIVE
**Decision:** Treat **country as a hard blocking key**. Only compare records with the same `country` label, and handle country as an open string set.
**Reason/Evidence:** 100% of the 7,638,365 positive pairs have matching country labels (EDA §2). This cuts comparisons roughly in half with zero recall loss. Treating it as an open string set means France is handled automatically.

### D-007 | 2026-09-25 | Post-processing | ACTIVE
**Decision:** Enforce **each S2/S3 record is assigned to at most one S1**: if several S1s claim the same record, keep only the highest-scoring claim.
**Reason/Evidence:** In the train GT, **0** S2/S3 IDs appear under more than one S1 (EDA §2). The GT is a partition. This constraint removes false positives for free, which matters under F0.5.

### D-008 | 2026-09-25 | Modeling focus | ACTIVE
**Decision:** The match decision must rely heavily on the **address**, not only the name.
**Evidence:** 40% of train S1 records share their normalised name with at least one other S1 in the same country (up to 253 entities per name, for example "Meridian"). The pair (core name, first address part) is almost unique: only 0.2% duplicates (EDA §7).

### D-009 | 2026-09-25 | Blocking | ACTIVE
**Decision:** Do **not** rely on ZIP or PIN codes for blocking.
**Evidence:** Postal codes are almost absent. A 6-digit PIN pattern appears in 0.3% of India S1 addresses. The 10% "zip5" hits in US addresses are actually 5-digit house numbers (EDA §4).

### D-010 | 2026-09-25 | Validation | ACTIVE
**Decision:** The validation harness must **simulate test's higher distractor density**. Drop about 20% of held-out S1 entities from the query set but keep their S2/S3 records in the search pool as orphans.
**Reason/Evidence:** Test has 5.76 S2+S3 records per S1 vs 4.68 in train, uniformly across US, India and France (EDA §3). If matches per S1 stay at 3.46, distractors per S1 rise from about 1.2 to about 2.3. Without this simulation, local precision and tuned thresholds would be too optimistic for the leaderboard.

### D-011 | 2026-09-25 | Blocking | ACTIVE
**Decision:** Blocking is a **union of cheap, complementary keys, scoped by geography**, not a global TF-IDF kNN:
(1) canonical-name exact keys within country; (2) house number + street token + city keys; (3) rare-name-token inverted index within state/city (skipping high-frequency tokens); (4) char-n-gram TF-IDF top-k inside state/city blocks for the fuzzy tail. Target: ≥99% recall on train at about 20–40 candidates per S1. The candidate set must be audit-clean, because `candidate_pairs.tsv` = exactly what the model scores.
**Evidence:** Global char-3gram TF-IDF top-50 gives 96.2% recall (name 69.5% alone, address 87.6% alone). It runs at about 16 queries/s, which means **30+ hours** for 1.7M test S1 (EDA §8). Name-only neighbourhoods are flooded by same-name entities (§7). The misses are native-script names, null addresses and alias names, and each needs a dedicated key.

### D-012 | 2026-09-25 | Normalisation | ACTIVE
**Decision:** Resolve native-script (Indic) names with a **token dictionary learned from train pairs** by position alignment. Each native token maps to a set of English variants. Spelling variants (laxmi/lakshmi, jay/jai, shree/sri/shri) are canonicalised on both sides. unidecode is only a fallback.
**Evidence:** The native vocabulary is closed: 1,518 tokens in train and in test, with **100% of test occurrences covered by train**. Position alignment gives 1,347 tokens at **96.6% weighted purity** (EDA §6). Native-script names are 24% of India S2 and 13% of India S3, and 34.5% of blocking misses. This uses train data only, so it complies with the external-data rule.

### D-013 | 2026-09-25 | Features | ACTIVE
**Decision:** Engineer **fine-grained house-number features**: equality after stripping zeros, subsequence/substring, numeric gap |a−b|, same digit count, digit-multiset equality (permutation), range containment (1357-1361), and number-set containment. Treat these as the main precision lever.
**Evidence:** Planted decoys are near-duplicates with the same name, street, city and unit but a shifted house number (1643→1647, 4845→4854). 99.3% of them are distractors (EDA §10). True pairs perturb numbers differently (leading zeros, dropped digit, ranges, prefixed plot numbers). Address-number Jaccard is 0.84 for true pairs vs 0.09 for hard negatives.

### D-014 | 2026-09-25 | Generalisation | ACTIVE
**Decision:** Features must be **country-agnostic** (similarities, ranks, containment, ambiguity counts). Country is never a model feature, only a block key. Add French normalisation (street types, bis/ter, accents, region↔department) from generic language knowledge. Measure generalisation with **leave-one-country-out** validation (train US → validate India, and the reverse) as the proxy for France.
**Evidence:** France is 15% of test S1 and absent from train. Its noise rates match US and India (EDA §12), so the generator is shared, but its geography and naming differ.

### D-015 | 2026-09-25 | Decision rule | TENTATIVE
**Decision:** Replace a single global threshold with **per-entity expected-F0.5 subset selection**. For each S1: sort candidates by calibrated probability and evaluate the expected F0.5 of the top-k for k = 0..n (k = 0 has expected score P(no match)). Choose the best k. The fallback and baseline is a global threshold tuned on macro F0.5.
**Reason:** The metric is per-entity macro F0.5. For a 3-match entity one false positive costs about 0.21 and one miss costs about 0.09 (asymmetry of about 2.3×). The best cut depends on each entity's probability profile. It will be confirmed on validation against the threshold baseline.

### D-016 | 2026-09-25 | Address similarity | ACTIVE
**Decision:** Use **asymmetric/containment** address measures (share of the shorter record's tokens and numbers present in the longer, partial ratios) alongside symmetric ones. Strip null-like tokens (`null`, `<NULL>`, `N/A`) before comparing. Model null addresses explicitly with a flag and don't impute them.
**Evidence:** 55% of India S2 and 39% of India S3 true matches drop address parts. 3–4% of addresses contain null-like tokens and 3–5% are fully null (EDA §4–5).

### D-017 | 2026-09-25 | Features | ACTIVE
**Decision:** Add a **name-ambiguity feature**: the count of S1 entities in the same country and block sharing the core name, plus IDF-weighted name overlap. This lets the model require stronger address evidence for generic names.
**Evidence:** 49.5% of S1 share their core name with another S1, up to 567 ("Meridian"). The pair (core name, first address part) is almost unique, with 0.2% duplicates (EDA §7).

### D-018 | 2026-09-25 | Compute | ACTIVE
**Decision:** Develop and iterate on the laptop with ~10% samples. Run full-scale blocking, feature building and training on an **AWS memory-optimised CPU instance** (r-family, ~128 GB, spot where possible; $140 budget). Use the **Kaggle GPU** only for optional neural components (for example a multilingual encoder as an extra feature). Never use AWS AI or data services (for example Amazon Location Service geocoding) on the data.
**Reason:** The bottleneck is RAM and CPU for sparse and string operations (the laptop had ~4 GB free during EDA), not GPU. The fair-play rules ban geocoding and external lookups.

### D-019 | 2026-09-25 | Features | ACTIVE
**Decision:** Use the parsed **US state** as a strong feature and soft block key. Records whose state can't be parsed (null addresses) fall back to name keys.
**Evidence:** When parseable on both sides, the state agrees in 99.8% of true pairs vs 49.5% of hard negatives. About 5% of matches have no parseable state (EDA §5, §10).

### D-020 | 2026-09-25 | Validation | ACTIVE (refines D-004, D-010)
**Decision:** Build validation and training sets as **region universes**. Whole (country, region) blocks go to VAL (~8% of each country's S1) or TRAIN (~16%). A block may not push a universe past 1.5× its target, so no giant state dominates. Within a universe, 81% of S1 are queries. The other 19% are dropped but their S2/S3 records stay in the pool as orphan distractors.
**Reason:** Blocking is region-scoped, so a universe that contains every S1 of its regions reproduces test conditions exactly. S1 entities compete for the same records (the partition constraint works as in test), and 81% keeps distractors at about 2.3 per S1 (solving 1.215 + 3.46·(1−f) = 2.3·f gives f ≈ 0.81). Holding out whole regions also prevents geographic leakage between train and val.

### D-021 | 2026-09-25 | Normalisation | ACTIVE
**Decision:** The canonicaliser (`src/textnorm.py`) inverts each catalogued noise operation:
- alias removal (`formerly`, `dba`, `d/b/a`, `aka`, `t/a`: keep the part after)
- pipe/URL parts
- website and handle names reduced to a compact stem
- leetspeak digits inside words
- accent folding
- collapsed initials (`L.L.C.` → llc)
- legal-suffix canonicalisation; legal forms are kept as a separate field and excluded from the core name
- honorifics removed from the core
- PO boxes and null tokens stripped from addresses
- street, saint and French abbreviations
- explicit region detection (US states, India states and codes, French regions and departments)

Resources learned from data (`src/resources.py`): the Indic name dictionary (1,353 tokens), spelling-variant groups (laxmi/lakshmi, jai/jay, shree/sri/shri), native-script state names (16), and city→region maps from S1.
**Evidence:** Region detected for 100% of train S1 and 96.6% of train S2. The remaining 3.4% ≈ the null-address rate.

### D-022 | 2026-09-25 | Model choice | ACTIVE
**Decision:** The final matcher is **LightGBM** (MIT license, far below the 8B-parameter cap). **No LLM.** A ≤150M-parameter multilingual encoder (MIT or Apache, license checked on the model card) may be added later as an extra feature scored only on uncertain pairs, on the Kaggle GPU.
**Reason:** About 50M test pair decisions make LLM inference infeasible (a 7–8B model manages ~10–50 pairs/s on a T4). The noise is rule-based, and the one language-model-shaped problem (transliteration) is already solved by the learned dictionary. Llama and Gemma licenses are not MIT or Apache. scikit-learn (BSD) is used only for TF-IDF preprocessing, never as the final model.

### D-023 | 2026-09-25 | Results: baseline v1 | ACTIVE
**Setup:** canonicaliser (D-021) → region-scoped blocking (K_name=25, K_addr=25, K_fallback=10) → 65 features → LightGBM (trained on 13 of the planned feature parts, 3.3M pairs, easy negatives subsampled at 25% with weight 4) → partition constraint + threshold.
**Validation** (region universe, 196,781 queries, distractor density simulated as in D-020):
| Metric | Value |
|---|---|
| Blocking pair recall | **98.63%** (val) / 98.53% (train); 56.5 candidates per S1 |
| Blocking-oracle ceiling (macro F0.5) | 0.9957 |
| **Macro F0.5, threshold 0.75** | **0.9769** (flat from 0.70 to 0.80: robust) |
| Macro F0.5, expected-F rule | 0.9763 |
| India / US | 0.9725 / 0.9797 |
| Singletons / non-singletons | 0.9738 / 0.9771 |

**Decision:** use the **threshold 0.75** rule for submission v1. Expected-F is slightly behind, most likely because of calibration after negative subsampling. Keep D-015 TENTATIVE and revisit after calibration (isotonic on held-out data).
**Top features by gain:** pre (blocking cosine sum), **pre_rank_mid** (competition among S1s for the same record), house_s_in_cnums, pre_gap_mid, ad_tset, **house_gap_log**, pre_rank_s1, nm_canon_ratio, amb_country, house_len_eq. This confirms the EDA: the partition/competition and house-number decoy defence (D-007, D-013) carry the model.

### D-024 | 2026-09-25 | Engineering | ACTIVE
**Decision:** Everything runs memory- and disk-bounded. Features are built in 10k-S1 chunks and streamed to parquet parts. Val and test are scored on the fly by region batch, keeping only (s1, mid, p), with p ≥ 0.02 for test. Raw-TSV parquet copies are deleted after preparation.
**Reason:** The laptop has about 6–7 GB of usable RAM (other apps use ~9 GB) and the C: drive is 98–99% full (3–5 GB free). Unbounded runs crashed with MemoryError and disk-full errors, and when the page file can't grow, swapping slows feature building from ~80k to ~9k pairs/s.

### D-025 | 2026-09-25 | Submission hygiene | ACTIVE
**Decision:** Before every upload:
1. Run the official validator in strict mode (`--check-ids`).
2. Run `check_submission.py`. It adds byte-level format checks, per-source ID existence, one row per test S1, matches ⊆ candidates, same-country matches, the partition check, and zip layout and content checks.
3. Re-derive the matches from the saved scores to confirm the decision step is deterministic (explicit (p, s1) tie-break).

The final package ships the exact v1 LightGBM model and config (`code/business_entity_resolution/models/`), so reviewers can regenerate the submitted outputs exactly.
**Reason:** There are only 5 uploads per day. Reproducibility is audited ("anyone should be able to regenerate both output files"), and v1 was trained on 13 of 19 chunks, so a retrain alone would not reproduce it bit-for-bit.

### D-026 | 2026-09-25 | Process | ACTIVE
**Decision:** From v2 on, every change is an isolated, ablated experiment tracked in `docs/04_EXPERIMENTS.md`. A change is merged only if it gains ≥ +0.001 val macro F0.5 without regressing India or US by more than 0.001. Leaderboard uploads are used only for merged versions or deliberate probes (e.g. the France threshold, E7).
**Reason:** v1 scored 0.9665 on the LB (val 0.9769). The field is packed in the 0.96–0.985 band, so gains must be attributable, and LB feedback is scarce (5 uploads per day).
**Evidence driving v2:** 72% of val loss is recall. 49% of false negatives have an empty address. 27% of FN pairs are blocking misses. FPs are house-number and name-mutation decoys. France is estimated at about 0.91 on the LB. H-1 (substitution = decoy) was rejected on train data.

### D-027 | 2026-09-25 | Compute | ACTIVE (supersedes the compute part of D-018)
**Decision:** All heavy runs move to AWS `m6i.4xlarge` (16 vCPU, 61 GB, ap-south-1), driven over SSH from the laptop. Jobs run in `tmux` so they survive disconnects. Only final TSVs and models are downloaded.
**Evidence:** The v2a test run took 52 min on AWS vs about 3 h on the laptop, which also crashed repeatedly on RAM/disk. Server preparation outputs match the laptop exactly (region coverage identical; dataset bytes and lines identical).

### D-028 | 2026-09-25 | Features | ACTIVE
**Decision:** v3 removes raw-count features whose scale depends on split size (amb_country, amb_region, pool_cnt_c, pool_cnt_s, rare_tok_idf, nm_idf_total1, nm_extra_tok_idf). Scale-free ratios are kept (IDF-weighted containment, rarest-token hit).
**Evidence:** v2a: val +0.0041 but LB −0.0013 vs v1b, with lower predicted match rates on test in every country. v2b → v2c (restoring match rates) moved the LB up slightly. Both point to downward-shifted test probabilities.

### D-029 | 2026-09-25 | Diagnostics | ACTIVE
**Decision:** Spend one day-2 upload on a France-empty probe (v2c with France rows blanked) to measure France's leaderboard F0.5 directly. This decides whether effort goes to France (e.g. synthetic French training pairs from test S1 using the catalogued noise operations; no labels, no external data) or to the US/India model.
**Reason:** Each LB upload is scarce, and the 0.014 gap has two candidate causes that need opposite work.

### D-030 | 2026-09-26 | Validation | ACTIVE (D-029 is superseded: the France probe is no longer needed)
**Decision:** Add a second held-out check, the **wide universe**: 20 more train regions (25% of each country's S1, giant regions allowed, same 81% query / orphan simulation; `eval_regions.py`). Every candidate is judged on val **and** wide. Wide is the leaderboard proxy.
**Evidence:**
- v3b stage 1 scores val 0.9788 but **wide 0.9698**.
- US transfers perfectly (0.9815 vs 0.9812). India drops from 0.9746 to **0.9579**: Telangana 0.876, Maharashtra 0.968.
- Mixing US 0.981 and India 0.958 at test proportions gives ≈0.968, which **is the leaderboard**. The gap is India, not France or feature shift.
- Upload A (0.966111) had already shown that deleting the count and competition features loses more than it gains.

### D-031 | 2026-09-26 | Blocking | ACTIVE
**Decision (v6 blocking, `ER_COMB=1 ER_LEAKS=1`):**
1. **Region-leak expansion.** Queries of region R also search the pool regions that R's true matches leak into. The pairs are learned from train labels (`region_leaks.py`: rate ≥ 2% with ≥ 30 pairs, or ≥ 0.5% with ≥ 500 pairs; 16 pairs).
2. **Combined pass.** TF-IDF name and address vectors are concatenated, top-25, so ties among identical names are broken by address.
**Evidence:**
- S2/S3 write Telangana addresses with the old state name "Andhra Pradesh": 18.5% of Telangana pairs, never compared before. Washington DC → "WA": 77%.
- In Maharashtra, half of the in-region misses have the exact S1 name, shared by a median of 35 pool records, and a truncated address.
- Blocking recall: **Telangana 0.803 → 0.9915**, Maharashtra 0.963 → 0.985, NC 0.992 → 0.995. Candidates per S1: +4 in normal regions, +57 in Telangana.

### D-032 | 2026-09-26 | Training data | ACTIVE
**Decision:** v6 trains stage 1 on **all regions except val and wide**: 957k queries in 45 regions, about 5× before. Negatives are presampled while writing (`ER_PRESAMPLE=1`) to bound disk use. It keeps the v5 test-consistent feature computation (batch 100k, ambiguity counts on an S1 sample of test's size).
**Reason:** The old training universe held 3 Indian states. Test India spans 34 states and several native scripts.

### D-033 | 2026-09-26 | Candidate file (problem-statement update) | ACTIVE
**Decision:** `candidate_pairs.tsv` is the set the final matcher runs inference over. That is the blocking candidates kept by the stage-1 pruning model (p ≥ 0.02). `run_stage2.py test` writes it next to `matching_results.tsv`.
**Reason:**
- The update says candidate generation counts toward the final ranking: a smaller candidate set per S1 ranks higher, and the file must be "whatever your model actually runs inference over… the last stage".
- The cross-encoder, stage 2 and the assignment only ever see pairs with stage-1 p ≥ 0.02.
- The pipeline is a 3-step cascade: TF-IDF multi-pass blocking (69/S1) → learned pruning, a LightGBM pair scorer (**4.16/S1**) → final matcher (cross-encoder + stage-2 GBDT + one-to-one assignment).
**Evidence (v6):** pruning keeps **99.56%** of the true pairs the blocker found (val 4.16/S1, wide 4.14/S1). Matching results are unchanged, because pairs below 0.02 are never matched (threshold 0.675).

## D-034: per-country thresholds tuned at test-like orphan density (27 Sep)
**Decision:** US 0.60, India 0.50, France 0.725, instead of one global 0.725.
**Evidence:** the held-out universes hold far more orphan-like records than test.
- 19% of S1 are dropped in the simulation vs ~9% US / ~2% India measured on test.
- Records whose owner lies outside the universe add more.

After thinning them to test density, the best thresholds are US 0.6–0.65 (variant A) / 0.45 (B) and India 0.5 / 0.45. Chosen values are robust across both variants. Leaderboard: about +0.0004 (inside v6fr3s).

## D-035: France same-address category-swap veto (27 Sep)
**Decision:** remove accepted French pairs at an exact address where the record drops an S1 word and adds a different real name word ("Pornic Club SAS" -> "Pornic Sportive SAS"), excluding generator noise words, typos and abbreviations (`final_decide.py`).
**Evidence:**
- Labelled US/India: 2.2–3.7% true (sister-business distractor).
- Accepted per S1: France 0.078 vs US/India 0.0006–0.0009.
- Swap structure is random across categories (the distractor signature, not the synonym-noise one).
- Per-source counts of affected S1s fit the extra-record model (L1 0.035 vs 0.135).

**Result:** 20,153 pairs removed. Leaderboard 0.984381 -> 0.987206 (France about +0.016).

## D-036: not adopted on 27 Sep (measured)
- round-2 stage 2 with count prior / twins: +0.00004;
- CatBoost stage-2 partner: +0.00003;
- record-centric containment blocking (v6n): wide recall +0.0005, stage-1 F unchanged;
- hub-rename / house-shift vetoes: US/India analogues 88–99% true;
- stricter threshold for ambiguous empty-address records: hurts at test-like density;
- the count test as a general estimator: fails validation on labelled groups.
