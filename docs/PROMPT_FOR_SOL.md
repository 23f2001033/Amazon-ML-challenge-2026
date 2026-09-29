This is an update to the brief I gave you earlier (Amazon ML Challenge 2026, Business Entity Resolution; sections 1–9). The rules and the pipeline are the same. Since then we moved from **0.984381 to 0.988316**, rank ~100. Top 3 are 0.991829 / … / 0.99117.

**About 5.5 hours and 3 uploads are left.** I need your sharpest plan to reach ≥ 0.990, ideally ≥ 0.9912.

Read everything first. Many obvious ideas are already measured (section 4). Don't re-suggest those without a new angle.

# 1. Where we stand
| Version | Change | LB |
|---|---|---|
| v6fr3 | (end of the earlier brief) | 0.984381 |
| v6fr3s | France "category-swap" veto + per-country thresholds tuned at test-like orphan density | 0.987206 (+0.00283) |
| v6fr3w | Stage 2 retrained with test-like sample weights, plus two small decision rules (below) | 0.988137 (+0.00093) |
| v6fr3x | US/India from stage-2 variant w2 (below); France threshold 0.725 → 0.60 | **0.988316** (+0.00018) |

**Score split.** LB = 0.8503 × F(US+India) + 0.1497 × F(France), by S1 share.
- Only the older v6rc3 was measured directly: an upload with France emptied gave US+India ≈ 0.989.
- Current estimate: **US+India ≈ 0.990–0.991, France ≈ 0.975–0.979.** Only the sum is measured.
- **To reach 0.990 we need +0.0017.** That is France +0.011 alone, US+India +0.002 alone, or a mix.
- For the top 3 we need +0.0029.

# 2. What worked since the brief, and why
1. **France category-swap veto (+0.0028 LB).**
   - **Pattern:** the accepted pair has the same address, and the record drops an S1 name word and adds a *different real category word*. Example: "Pornic Club SAS" → "Pornic Sportive SAS". Generator noise words, typos and abbreviations are excluded.
   - **Evidence:**
     - The same pattern in labelled US/India pairs is only 2.2–3.7% true.
     - France accepts it 0.078 times per S1 vs 0.0006–0.0009 for US/India.
     - Per-source record counts of the affected S1s fit the "extra distractor" model.
   - **Effect:** 20,153 French pairs removed.
   - **General method that worked:** find a generator pattern that labelled US/India data shows is false, and that France accepts far more often per S1. Every France-only change moves the LB by exactly 0.15 × ΔF(France).
2. **Test-faithful orphan density.**
   - Our held-out universes hold far more "orphan-like" records than test:
     - S1 deliberately dropped from the universe: 19% simulated vs ~9% US and ~2% India on test, measured via identical-record groups;
     - records whose owner lies outside the universe (on test their S1 is present).
   - Thinning them to test density removes ~30% of the band negatives and none of the positives. That moves the best thresholds.
3. **Stage 2 with test-like sample weights (w).**
   - 37% of band training pairs are orphan-like; they're down-weighted to test density.
   - Test-like val: +0.00013. LB: +0.00093 together with two small rules:
     - accept 3,888 French "dropped word + noise suffix" copies at an exact or close address (98–99% true in US/India);
     - "empty-S1 rescue": an S1 with no prediction takes its best unassigned candidate if p ≥ 0.35 (1,746 S1s).
   - **The LB moved about 5× more than validation predicted**, so the train/test shift is bigger than our simulation captures.
4. **w2:** records owned outside the universe weighted 0.05. Test-like val (variant B) 0.99097 vs 0.99062 for w. Used for US/India in v6fr3x; France uses the w model at threshold 0.60.

# 3. In flight now (the only candidate being built)
- **Stage 1 retrained with the same test-faithful weights (v6t):** +0.00077 test-like at stage-1 level.
- But 32% of its uncertain-band pairs are new and have no cross-encoder scores, so stage 2 on v6t scores **0.99094 vs 0.99097. No gain yet.**
- The four CEs are now scoring the new pairs on Kaggle (val/wide now, test after ~19:30). Then stage 2 is re-run. It will be uploaded only if val-B beats 0.99097 by ≥ 0.0002.
- Candidate ~21:30 IST at the earliest. **If you think this path is a waste, say so and why.**

# 4. Measured and rejected since the brief
- **LLM judge** (≤ 8B instruct model, zero-shot, 6,000 labelled val band pairs): AUC **0.51** vs stage-2 0.84. Combined: no gain.
- **Learner changes:** CatBoost stage-2 partner +0.00003; bigger stage 2 (more leaves/rounds) ±0; stacking/ensembles of stage-2 variants ±0. The learners agree; the remaining errors are in the data.
- **Round-2 stage 2** (features from round-1 decisions: rival per-source counts, count-prior likelihood ratio, twin consistency): +0.00004.
- **Blocking:** containment pass and record-centric fallback give wide recall +0.0005, but stage-2 F is unchanged.
- **Vetoes whose US/India analogues are 88–99% true:** hub-address renames, house-number shifts, typo swaps.
- **Stricter threshold for ambiguous empty-address records:** hurts at test-like density.
- **Count model:** observed per-source counts = Binomial(true count, r) + Poisson(extra), with the true-count prior from train.
  - Validated for aggregate recall, but it underestimates false positives, and group-level use fails validation on labelled groups.
  - For France it estimates recall **r ≈ 0.944 at threshold 0.725 → 0.948 at 0.60**, for about +0.0017 extra records/S1. The LB rewarded that trade (+0.00018). Taken at face value, France still loses more to missed records than to false ones, but this estimator is weak on FP.
- **Confident pairs (stage-1 p ≥ 0.99, which bypass stage 2):** 0.03% wrong in US/India. A perfect veto oracle gives only +0.00025.
- **No leaks:** entity IDs and row order are random. Empty addresses are independent per record. Per-source counts follow a fixed prior.
- **France stage-2 variants:** w2 and a France-specific reweighting (wfr) showed no label-free evidence of beating w on France.

Section 6 of the earlier brief (loss anatomy) still holds for US/India. Missed matches dominate; 79% of scored-but-missed pairs have an empty address, and the name is shared by ~5 S1s.

# 5. Resources for the remaining time
- **CPU VM:** 16 vCPU / 61 GB.
  - A decision-layer change on saved test predictions takes minutes.
  - A stage-2 retrain plus test takes ~40 min.
  - A stage-1 rebuild (~4 h) is no longer possible.
- **Saved artefacts:**
  - stage-1 p for every candidate pair (p ≥ 0.02 kept, 4.2/S1) on val, wide and test;
  - stage-2 p on the band;
  - all four CE scores on the band;
  - CE 3-FR / CE 4-FR scores on the 802k confident French pairs;
  - all pair features;
  - full labels for val/wide (US/India).
- **GPU:** Kaggle T4 × 2. Busy ~1.5 h with the scoring above. A base CE scores ~1,000 pairs/s, the large one ~250–450 pairs/s.
- **Rules:** no external data, APIs, geocoding or lookups; final model MIT/Apache ≤ 8B; no paid LLM APIs; I can't give you machine access.

# 6. What I need from you
1. **Diagnosis.** Where are the ~0.0035 between us (0.9883) and the top (0.9918) most likely lost: France precision, France recall, US/India empty-address ambiguity, or something structural we've never looked at? Rank your hypotheses by the evidence above, and name the one measurement on our saved artefacts that would confirm or kill each.
2. **Moves that fit in ≤ 3.5 hours with the artefacts above,** ranked by expected LB gain. For each:
   - the exact rule, feature or algorithm, and which country;
   - the expected gain, using LB = 0.85 ΔF(US+India) + 0.15 ΔF(France);
   - the validation before upload: the labelled US/India analogue plus per-S1 rate comparison for France rules, and test-like variant B for US/India.
3. **France patterns to check next,** in the style of the category-swap veto. These are concrete signatures in accepted or rejected French pairs to measure against their US/India analogue, e.g. which record/S1 word-edit shapes, address relations or hub structures. Include the recall side: patterns of *rejected* French pairs that US/India data says are true.
4. **Upload plan:** how to spend the 3 remaining uploads, and what to keep as the final submission.

Answer as a numbered list, highest expected value first, specific and testable. No generic advice (more data, bigger models, ensembling, threshold tuning).
