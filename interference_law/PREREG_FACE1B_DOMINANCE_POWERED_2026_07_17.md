# PREREG — Face-1b: POWERED adjudicator for a-priori pecking-order predictability (Lambda arm)

**Date:** 2026-07-17 · lock = sha256(this file)[:8] · Extends the SCREEN `VERDICT_FACE1_DOMINANCE_APRIORI` (full-null at n=12, single seed).
**Venue:** Lambda ML-CF lane, a10-first (resnet18/128px is workload-light). ~1–1.5 GPU-hr, ~$1–3. Fire once, always-terminate.

## Why (from the screen)
The screen found φ (dominance potential) NOT a-priori predictable — cardinal LOO-R² −0.17, ordinal rank-corr +0.08 — but flagged the honest
confound: n=12 and φ is measurement-noisy (split-half ρ 0.51 → attenuation biases the ordinal test toward null). This arm DE-CONFOUNDS that by
(i) n=16 caps, (ii) **3 seeds seed-averaged to de-noise φ**, (iii) a SECOND dataset (DTD) for generalization, (iv) a pooled n≤32 test.

## Design
Reuses `probe_cyclic.build_F` + `.hodge` (φ machinery that DEFINED the pecking order) and `probe_dirlaw_arch` feature/FT path — shipped, not
re-coded. For dataset ∈ {CIFAR, DTD}, 16 caps, shared neg pool:
- Base predictors (frozen ResNet18, computed ONCE, a-priori): dprime, spread, munorm, wnorm, iso — same defs as `probe_fragility`.
- For seed ∈ {0,1,2}: build full 16×16 directed F (full_ft 300 steps per perturber col); φ = hodge(F); fragility_i (row mean), disrupt_i (col mean).
- **φ_mean = mean over 3 seeds** (de-noised). Reliability = pairwise cross-seed Spearman of φ (report; expect > screen's 0.51).

## Pre-committed (IDENTICAL to screen — no drift)
- Trio {dprime, wnorm, spread}, fixed. Primary predictor = **d′**. Others Bonferroni-99.2% exploratory.
- Ridge λ=1.0 standardized. Bootstrap 10k ρ-CI; permutation 10k ρ-p; permutation-null 1k under LOO-R².
- **CARDINAL pass:** LOO-R²(φ_mean; trio) ≥ max(0.30, perm-null95) AND ρ(d′,φ_mean) 95% CI excludes 0.
- **ORDINAL pass (answers the title):** Spearman(LOO-pred φ_mean, true) ≥ 0.50 AND perm-p < 0.05.
- Evaluated per-dataset (CIFAR = primary, matches screen; DTD = generalization) AND **POOLED (n≤32)**.

## Priors (stated to lose)
- Same as screen: cardinal FAIL likely; ordinal — now with de-noised φ and n up to 32, I upgrade to **~50/50** (the screen's ordinal null may have
  been pure attenuation). Falsifier that proves the screen's headline wrong = ordinal pass on CIFAR or pooled.

## Pre-committed HEADLINES
- ORDINAL pass (CIFAR or pooled): **"Screen OVERTURNED — the pecking order IS a-priori computable (ordinally), the n=12 null was attenuation."**
- CARDINAL pass too: magnitude predictable.
- BOTH fail with de-noised φ (cross-seed ρ clearly > 0.51) at n≤32: **"CONFIRMED — the order is genuinely emergent from interaction, not static geometry"** (the screen null was real, not a power artifact).
- If cross-seed reliability stays ~0.5 even seed-averaged: φ itself is intrinsically noisy → report as an UPSTREAM measurement limit, not a clean null.

## Output
Remote `out/face1b_dominance_apriori.json` → pulled to `interference_law/out/`. Then `VERDICT_FACE1B_*`. Cost logged to `lambda_cost_ledger.csv`.
