# VERDICT — Thread 1: promote G5 from separation to load-bearing, complete the ladder. 2026-06-22.

Two-part thread (user-directed): (1) re-fire the OOM'd opt-1.3b geometry cell on a big GPU; (2) run the RMD-SRC
random-grouping null to test whether the family partition specifically PAYS per geometry feature — discharging the banked
caution "a class-separation SE is not a prediction-claim SE". Plus the depth-side mediation (option B division of labor).

## Part 1 — opt-1.3b cell recovered
OOM was cross-seed memory fragmentation (completed seed 0 at 40GB, died seed 1), NOT a single-seed peak. Fix = three
`torch.cuda.empty_cache()` calls (after each optimizer free + end of each seed), zero dynamics change. Re-ran on the
available 40GB a100_sxm4 (no H100 capacity): all 3 seeds completed. **opt1_3b G5 = 0.439** (rand 0.125), G3 0.242, G4
0.309, G6 0.180 — OPT-leaning, an independent replication of the ML-agent's 0.430. Ladder now complete: OPT {125m .344,
350m .410, 1.3b .439} vs Pythia {160m .525, 410m .505, 1b .490}.

## Part 2 — RMD-SRC family-grouping null (6 cells, no imputation)
Faithful to B_PARSIMONY_SPEC (per-unit vs group-shared OLS, held-out Gaussian-ll delta, K=200 random groupings ->
perm_p; PAYS_STRICT iff base in {PAYS,OUTPERFORMS,MATCHES} AND perm_p<=0.10). Unit=model, group=family, y=each Gi
(intercept-only), rows=per-carrier values. Script `rmdsrc_geom.py`, result `gateA/rmdsrc_geom.json`.

| feature | base | perm_p | verdict | robust to hostile 1.3b imputation? |
|---|---|---|---|---|
| **G5** deposit↔surge | PAYS | **0.015** | **PAYS_STRICT** | **YES** — paid strict even under PY-high clone (p=.025) |
| **G4** surge↔stiff   | PAYS | **0.015** | **PAYS_STRICT** | NO — flips to COSTS under PY-high imputation; holds only because real 1.3b is OPT-leaning |
| G2 spectral decay | PAYS | 0.129 | PAYS_BUT_NOT_SPECIFIC | n/a (never family-specific; precision-fragile) |
| G1, G3, G6 | COSTS | — | COSTS (size-driven) | n/a |

**Conclusion:** G5 is the rock-solid family-specific geometry feature — the family PARTITION specifically pays (beats
random) AND survives the worst-case imputation of the structural hole. This promotes G5 from "OPT/Pythia differ at ~12×
bootstrap-SE" (separation magnitude) to "family is the load-bearing organizing partition for G5" (parsimony+permutation
footing). G4 is a secondary family feature but contingent on the real 1.3b being OPT-like; G5 ≫ G4 in robustness.

## Part 3 — depth-side mediation (my half of the two-mechanism B-D model)
`mediation_depth.py` / `gateA/mediation_depth.json`. Valid depths d12/d18/d24 (d6 VOID, surgery degenerated).
- **B-D builds monotonically with depth: −0.039 → +0.014 → +0.055** (swing 0.094) — protection is depth-built.
- **G5 is blind to the depth component** — NOT a correlation claim (n=3, G5 spuriously Spearman'd +1.0). The argument is
  variance-partition + sign: G5 swings 0.043 across depth vs 0.180 across family (~4× less responsive) AND its depth
  wiggle is WRONG-SIGNED vs its family law (family: G5↓ as B-D↑; depth: G5↑ as B-D↑). It structurally cannot mediate the
  0.094 depth swing. fc1's clock swings 0.66 (funnel), right-signed, monotone → the credible depth carrier.
- **HONEST LIMIT:** per-carrier `rho(resid_BD, depth_fraction)` is +0.8–1.0 for EVERY carrier type (fc1 +1.0, fc2 +0.70,
  attn +0.83–.97), a UNIVERSAL depth gradient — does NOT isolate fc1 at the per-carrier level. Claim = "depth component
  exists + G5 can't see it" (solid) and "fc1's clock is its model-level signature" (consistent w/ SPIRAL), NOT
  "fc1 exclusively carries it."

## Two-mechanism partial mediation (merge with ML-agent's family/size G5->B-D)
**B-D = α·G5(family/size) + β·depth-built-structure(depth) + ε** — orthogonal mediators. G5 mediates the family/size
component (its home turf, swings 0.18, right-signed); the depth component is G5-invisible and loads on depth-built
structure. ML-agent runs α (G5->B-D, 6 size cells; needs pending opt-1.3b + pythia-1b B-D); I supplied β (depth side).

## Open / follow-on
- "G5 PREDICTS B-D" (not just is-family-organized) needs per-cell dispositive B-D across the ladder. B-D-sign grouping in
  RMD-SRC NOT run — per-cell B-D currently only at ~350M band + the 3 depth cells + opt125m/pythia160m new cells.
- G4's family-specificity should be confirmed/denied once a second large-OPT or perturbed-1.3b cell exists.
