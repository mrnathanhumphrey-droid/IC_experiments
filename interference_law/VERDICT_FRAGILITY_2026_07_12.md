# VERDICT — NEW-1a: what is fragility? (PREREG a36cf7eb, locked 2026-07-12). Landed, $0 local 94s, 40 victims × 8 perturbers, non-saturated (median fragility 0.184).

## RESULT: PARTIAL. Fragility has a real dominant structural correlate — base discriminability MARGIN protects (ρ(d′,fragility)=−0.51, 1a-P3 ✅). But the full base-structure model does NOT generalize (LOO-R²=0.08, 1a-P1 ✗), and my lead mechanism (isotropic sensitivity) was NOT fairly tested — the random-perturbation instrument was DEGENERATE (moved margins by 0.2%). So: margin is a genuine handle; the sensitivity mechanism is untested, not refuted.

## Univariate structural predictors of fragility (n=40)
  predictor   ρ(·, fragility)   reading
  dprime      −0.511            ← STRONGEST. Better-separated capabilities are more ROBUST. (1a-P3 CONFIRMED, ≤−0.30)
  wnorm       −0.362            larger deployed-head norm → more robust
  spread      +0.306            more within-class feature dispersion → more fragile
  iso         +0.227            crowding (mean cos to other caps) → more fragile (1a-P4 near-miss, threshold +0.30)
  sens        −0.127            DEGENERATE instrument (see below) — noise, not a real test
  munorm      +0.120            weak

## Predictions
  1a-P1 predictable   multivariate LOO-CV R² = 0.080 (in-sample 0.369)   ✗ FAILED — but overfit + one dead predictor (below).
  1a-P2 sensitivity   ρ(sens,fragility) = −0.127                          ✗ but UNTESTED — degenerate instrument, null uninformative.
  1a-P3 margin        ρ(dprime,fragility) = −0.511                        ✅ CONFIRMED — the real finding.
  1a-P4 crowding      ρ(iso,fragility) = +0.227                           ◐ near-miss (right sign, below +0.30).

## The instrument failure (honest — 1a-P2 is untested, not refuted)
sens_i was defined as random-perturbation margin sensitivity with ε = 1%·‖θ0‖ = 0.85. Diagnostic: mean|Δmargin|/margin = 0.002 — the
perturbation moved the readout margins by 0.2%, i.e. FLOAT NOISE. Two causes: (a) ε=0.85 is ~14× smaller than real task travel
(‖Δθ‖≈12 ≈ 14%·‖θ0‖); (b) an ISOTROPIC-random direction in 11M-dim is nearly orthogonal to the few feature-relevant directions, so even
at realistic magnitude it barely perturbs features vs a gradient-STRUCTURED perturbation. So the isotropic-sensitivity hypothesis has NOT
been fairly tested — the null (ρ−0.13) reflects a dead instrument. Fixing ε and using a structured/Jacobian sensitivity is a NEW prereg
(1a'), not a silent re-run under this lock (sens's operationalization was part of what was registered).

## Why 1a-P1 failed, and why "not predictable" OVERSTATES it
LOO-R²=0.08 vs in-sample 0.37 = classic overfit: 6 collinear predictors (dprime/wnorm/munorm intercorrelated) + one dead predictor
(sens) on n=40 → the multivariate fit does not generalize (standardized coefs all ~0.02, signs scrambled by collinearity). But d′ ALONE
carries ρ=−0.51 (~26% of rank-variance) → fragility IS partially structural (margin-driven), not idiosyncratic noise. The registered
6-predictor LOO failed; the phenomenon (margin protects) is real. A parsimonious model (d′ + spread) would very likely clear the 0.20
"predictable" floor — deferred to 1a'.

## What stands
- **MARGIN PROTECTS**: the single best base-measurable correlate of robustness is discriminability d′ (ρ−0.51). Well-separated
  capabilities resist destruction; dispersed/crowded ones are fragile (spread +0.31, iso +0.23). This is a real, cheap, mechanistic handle
  on which capabilities will forget — measurable BEFORE any perturbation, from the base geometry alone.
- Fragility is PARTIALLY but not richly predicted by these base features (best single ρ 0.51; multivariate doesn't generalize as built).
- The isotropic-sensitivity mechanism is OPEN (bad instrument), the most important loose end.

## Follow-up (1a', NEW prereg if pursued — NOT done here)
(1) A REAL sensitivity instrument: perturb along gradient-STRUCTURED directions at realistic ‖Δθ‖≈12 (or compute ‖∂margin_i/∂θ‖, the
local Jacobian norm) — the fair test of whether fragility = sensitivity. (2) Parsimonious model (d′ + spread) with LOO. (3) More
victims/seeds for LOO stability. (4) Does margin-protects hold with a MATCHED-magnitude perturber set (removes the NEW-1 ‖Δθ‖ compression)?

## Caveats
Single seed; ResNet18; 40 victims / 8 perturbers; deployed binary heads. Base AUC saturates at 1.0 → d′ (not AUC) used, good. sens
instrument dead (named). LOO overfit (named). The margin-protects finding (ρ−0.51) is the robust, banked result; everything else is
partial or open.
