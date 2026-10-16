# PREREG — P2: PLASTIC-regime forward-transfer geometry (canonical LLM-LoRA)

**Locked 2026-07-26.** Transfer-geometry thread, phase P2. Lambda (ML-CF lane). Canonical plastic regime = B reshapes the backbone via LoRA. Reuses face1c_p4 harness helpers. 2 datasets (clinc150, banking77) for cross-dataset replication.

## Why plastic (from P1c)
Frozen regime EXHAUSTED: reliable forward-transfer interaction defeats a 14-descriptor battery (P0/P1/P1b/P1c), but frozen shares ONE backbone -> feature-reuse geometry degenerate + LogME undefined. The canonical second-law hunt REQUIRES B to reshape the backbone so CKA/subspace/LogME/eff-rank become source-dependent. Substrate = LLM-LoRA because the coupling papers we rebut (Geometry Conflict 2605.09608, O-LoRA 2310.14152, TSV 2412.00081) are LLM-LoRA, and it reuses the TESTED p4 harness.

## Design
Model pythia-1.4b (arch-agnostic all-linear LoRA), NCAP target caps (1-vs-rest vs shared NEG), SEEDS multi.
Per (seed s, source j): LoRA-FT base on j's texts -> reshaped backbone -> extract mean-pooled layer-mid acts for every target i (= phi_j(A_i)) + neg. (This is exactly face1c_p4's cache.)
- **FT(i<-j)** = fewshot_AUC(target i | phi_j features) - fewshot_AUC(target i | BASE features). Forward transfer through the j-reshaped backbone (does learning j make i easier). i != j.
- Reliable FT via seed-averaging; split-half reliability; row(target)/col(source)/interaction decomposition (P1b method).

## Descriptors (source-dependent, per (i,j))
- COS = cos(d_i, d_j) in BASE features  [forgetting cosine; DISSOCIATION check]
- CKA_drift = linearCKA(base(A_i), phi_j(A_i))         [how much source j moved target i's features]
- CKA_reuse = linearCKA(phi_j(A_i), phi_j(A_j))         [shared structure in reshaped space]
- SUBOVL = mean cos^2 principal-angles(top-k PCs phi_j(A_i), top-k PCs phi_j(A_j))
- LOGME = LogME(target-i labels {i vs neg} | phi_j(A_i) features)   [the MUST-BEAT, You et al 2021]
- EFFRANK = effective rank (RankMe entropy) of phi_j(A_i)           [feature-reuse richness]

## Pre-registered QUESTIONS + decision bands (commit BEFORE numbers)
**Q1 (dissociation persistence):** rho(COS, reliable-FT) per dataset. PERSISTS if boot 95% CI inside (-0.30,+0.30) both datasets (transfer still cosine-blind in plastic). BREAKS if CI reaches +/-0.30 (coupling appears once features are plastic -- itself an interesting result reconciling with the coupling papers).
**Q2 (feature-reuse law):** for each of {CKA_drift, CKA_reuse, SUBOVL, LOGME, EFFRANK}, rho(X, reliable-FT) AND rho(X, interaction-residual). HIT = |rho|>=0.30, boot CI excl 0, SAME-SIGN replication BOTH datasets, perm_p<0.01, and (for the residual target) genuine interaction. Report which descriptor + whether it BEATS LogME.
  - HIT on LOGME only -> "plastic transfer is label-evidence-governed" (confirms lit baseline, modest novelty).
  - HIT on a representation-geometry descriptor (CKA/SUBOVL/EFFRANK) that beats LogME -> THE SECOND LAW (feature-reuse geometry governs transfer) = the flagship result.
  - NULL (no descriptor clears bar) -> plastic transfer ALSO geometrically opaque -> the strongest possible negative (transfer resists geometry in BOTH frozen and plastic regimes); thread's law-hunt closes honestly, contrarian dissociation stands.

## Cost / ops discipline
- SMOKE=1 first (1 tiny cell, NCAP=3, 2 seeds, 40 steps) end-to-end on Lambda BEFORE real spend (~$0.5).
- Full: 2 cells (clinc150, banking77) x 3 seeds x NCAP sources. a10-first. WATCHDOG generous.
- **Per-source incremental JSONL cache** on the instance (`out/plastic_rows.jsonl`) -> a watchdog kill preserves every completed (seed,source). Launcher pulls JSONL + JSON. (H.265 lesson: incremental writes; size watchdog to workload.)
- Falsifier-first; every outcome (persist/break x hit/null) decision-relevant. USER writes, Claude probes.
