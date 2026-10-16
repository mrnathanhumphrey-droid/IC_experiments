# PREREG — NEW-1a: WHAT IS FRAGILITY? A mechanistic theory of what makes a capability easy to destroy.

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune, deployed frozen-head metric. $0 local. Reuses probe_asymmetry / probe_gradient_alignment machinery.

## The question NEW-1 created
NEW-1 (lock ebdbba5a) proved the directed forgetting matrix is asymmetric and that VICTIM FRAGILITY (fragile_i = mean over perturbers of
forget(i←·)) is the DOMINANT axis (ΔR²=0.23, > cosine 0.07). But fragility is a black box — we measured it, we don't know what it IS.
This probe regresses fragility on BASE-MEASURABLE structural properties of each capability (no perturbation needed to compute them) to
find the mechanism: what makes capability i easy to destroy?

## Design (fragility with regression power)
VICTIMS = classes 0..39 (40 capabilities), each with a deployed binary head w_i (class_i vs neg), FROZEN. PERTURBERS = 8 fixed binary
tasks (classes [50,55,60,65,70,75,80,85] vs neg), full-FT. NEG = [90..99]. fragile_i = mean over the 8 perturbers of forget(i←p) =
AUC_i(base) − AUC_i(after). n=40 victims → real regression power. Disjoint victim/perturber/neg pools.

## Base-measurable structural predictors (computed at θ0, per victim i)
- dprime_i   : deployed-head discriminability = margin/within-class-noise along w_i (base separation quality)
- spread_i   : within-class feature dispersion = mean per-dimension std of class-i features
- iso_i      : capability-space CROWDING = mean cos(d_i, d_k) over other victims k (high = embedded in a dense region)
- munorm_i   : ‖feature centroid of i‖
- wnorm_i    : ‖deployed head weight w_i‖
- sens_i     : RANDOM-perturbation margin sensitivity (the mechanistic LEAD) = mean over R random unit weight-directions of
               |margin_i(θ0 + ε·u) − margin_i(θ0)| / ε. Isotropic local sensitivity of i's readout to GENERIC backbone movement
               (uses RANDOM perturbations, NOT the task perturbers → non-circular test of "fragile to anything").

## PRE-REGISTERED PREDICTIONS
1a-P1 — FRAGILITY IS STRUCTURALLY PREDICTABLE. Multivariate linear regression of fragile_i on the 6 standardized predictors gives
    LEAVE-ONE-OUT CV R² ≥ 0.40. Fragility is not idiosyncratic noise — it is determined by base structure. FALSIFIED if LOO-R² < 0.20
    (fragility is not structurally predictable → it lives in the perturbation-specific interaction, not the capability).
1a-P2 — THE MECHANISM IS ISOTROPIC SENSITIVITY (my lead). sens_i is the single strongest predictor: ρ(sens_i, fragile_i) ≥ +0.50, and
    it is the top |ρ| among the 6. → fragility = how sensitive a capability's readout margin is to GENERIC backbone movement; fragile
    capabilities are fragile to ANY perturbation, not just task-aligned ones (isotropic, a local-geometry property). FALSIFIED if
    ρ(sens_i, fragile_i) < +0.30 → fragility is NOT generic sensitivity; it is structural (separation/crowding) instead.
1a-P3 — MARGIN PROTECTS. dprime_i negatively predicts fragility: ρ(dprime_i, fragile_i) ≤ −0.30. Better-separated capabilities (larger
    margin-to-noise) are more robust to destruction.
1a-P4 — CROWDING HURTS. iso_i (mean cos to other capabilities) positively predicts fragility: ρ(iso_i, fragile_i) ≥ +0.30. Capabilities
    embedded in DENSE regions of capability space are more fragile (more directions collide with them). [Uncertain sign; registered.]

## Outcome grid (locked)
- 1a-P1 + 1a-P2: fragility = ISOTROPIC SENSITIVITY — a clean local-geometry mechanism, portable and predictable from a cheap random-
  perturbation probe. Best outcome: fragility gets a mechanistic definition + a cheap estimator.
- 1a-P1 holds, 1a-P2 fails (structural predictors win instead): fragility is about SEPARATION/CROWDING (margin + capability-space
  density), not generic sensitivity → a different, also-mechanistic theory (fragility is a property of where the capability sits).
- 1a-P1 fails (LOO-R² < 0.20): fragility is NOT a capability property — it lives in the victim×perturber interaction → NEW-1's row-effect
  was partly an artifact of the specific perturber set; report and pivot to the interaction structure.

## Non-saturation
fragile_i median expected in partial band (NEW-1 gave 0.206 with the same perturber style). Require median fragile_i ∈ [0.03, 0.45]; if
saturated (some victims → chance) note and, if widespread, lower perturber FT strength. sens_i: choose ε so the mean |Δmargin| is
measurable (not float noise, not a huge nonlinear jump) — target ε giving mean |Δmargin|/margin ≈ 0.05–0.3; R=6 random directions.

## Config (locked)
build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT, wd 1e-4, perturber FT 300/5e-4 (as NEW-1 gate-passed). 40 victims, 8 perturbers,
NEG 90-99, PER_CLASS 100, seed 0. sens_i: R=6 random weight directions, ε auto-scaled to ‖θ0‖ (report the scale). Metrics = univariate
Spearman + multivariate LOO-CV R² + standardized coefficients. Single seed for the structure. No production system, no network, pure local CIFAR.
