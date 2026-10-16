# PRE-REGISTRATION — iNaturalist Deep-Tree (v0.3). LOCKED before the CONFIRMATORY run.

**Date:** 2026-06-13. Motivation: CIFAR-100 is a 2-level tree (Form-1 hyperbolic NULL, fair-eval
confirmed). iNat is 6–7 levels — the substrate where hyperbolic geometry + stacked tiers SHOULD pay
if they ever do. The CIFAR pre-reg gates DO NOT port automatically; re-anchored below.

## SCOUT vs CONFIRMATORY (critical distinction)
- The first run (`deeptree_inat.py`, 2-tier frozen-ResNet18 → euc/hyp z_trans, raw fit-pools
  {50,200,1000,4000}, multi-level readout) is an **EXPLORATORY SCOUT**, NOT confirmatory. Purpose:
  validate the iNat pipeline, surface iNat's per-class size distribution (to anchor the gate below),
  and give a rough geometry read. **Its fit-pool numbers MUST NOT be reported as "Axis-1 on iNat"**
  — raw counts conflate few-shot with which classes happened to be sampled (iNat species sizes vary
  wildly). A hyp-win in the scout is a LEAD to confirm, not a result.

## Re-anchored gates (CONFIRMATORY run only)
1. **Fit-pool = balanced k-per-target-level-class** (k ∈ {1,2,5,10,20}), NOT raw count. This is the
   true analog of CIFAR's "~10 examples per superclass." Few-shot is defined per taxonomic node, so
   variable species sizes can't be tuned against. The crossover gate (hyp−euc ≥ +2% top-1, 95% CI
   excludes 0, paired, 3 seeds) applies to THIS k. "Axis-1 confirmed on iNat" REQUIRES a clean low-k
   crossover under this definition.
2. **Tier depth: start at 3, not 6.** First confirmatory architecture = 3-tier stacked bottleneck
   **class → order → family** (the Arm-E stacked design, now necessary). Don't debug architecture +
   substrate simultaneously. Full 6-level (kingdom→species) only after the 3-tier is clean.

## Sequence (locked)
SCOUT (running) → use its class-size data to finalize balanced-k sampling → build 3-tier stacked +
balanced-k harness → CONFIRMATORY run (hyperbolic vs Euclidean at each tier interface, balanced-k
transfer to novel species). Geometry question and stacked-tier question tested IN THAT ORDER.

## What would CONFIRM / REFUTE on iNat
- **CONFIRM hyperbolic-needs-depth:** hyp beats euc by ≥+2% (CI>0) at low balanced-k, AND the
  advantage GROWS as the target level goes higher (deeper abstraction) — the depth-dependence CIFAR
  couldn't show. Would mean Form-1's CIFAR null was a shallow-tree artifact, not a geometry verdict.
- **REFUTE:** hyp ≤ euc even on the 6-level tree → hyperbolic doesn't help for this kind of tier
  abstraction, full stop (a strong, general negative).
