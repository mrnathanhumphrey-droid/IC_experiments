# PREREG — Thread #7b: UNIFICATION, w_A-projected re-test. Was #7 falsified, or mis-projected?

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune (plastic backbone), deployed frozen-A-head metric. $0 local GPU. Reuses probe_unification.py machinery.

## Why this exists (and why it is NOT a goalpost-move)
#7 (lock 18693b8d) FALSIFIED the unification AS REGISTERED: on-axis margin collapse along the centroid-difference axis d_A gave
ρ(onaxis,forget_old) = −0.26 (predicted ≥ +0.50), wrong sign. Post-hoc I identified the prime suspect: the DEPLOYED head decides
along its LOGISTIC weight w_A, NOT the centroid-difference d_A. AUC is rank/scale-invariant, so margin shrink along d_A need not drop
the head's AUC (which lives on w_A). #7's projection axis was mis-specified for its own metric. This is a NEW prereg with a NEW
predictor (project onto w_A), NOT a re-score of #7 — the falsification of #7-as-registered stands regardless of #7b's outcome.

## The circularity hazard (stated up front, and how the design defeats it)
Drift of the score distribution ALONG w_A trivially relates to the w_A-head's AUC (margin-along-w is the first moment of that
distribution). So ρ(ONAXIS_w, forget_old) being positive is PARTLY DEFINITIONAL and is NOT, by itself, evidence for the unification.
The falsifiable, non-definitional content is the CONTRAST and the LOOP:
  (i)  DECOMPOSITION: on-axis (along w_A) movement forgets; OFF-axis movement is HARMLESS. If off-axis drift were also predictive,
       the story would collapse to "any movement breaks the head" (= ‖Δθ‖ magnitude, no direction). The gap on−off is the content.
  (ii) LOOP CLOSURE: the BASE-space cosine cosAB (a property of the B-task, independent of A's post-drift) predicts the SIGN of the
       on-axis drag. cosAB→ONAXIS_w is not definitional — it links the plastic on-axis mechanism back to the frozen −cos coordinate.

## Definitions (per A-class a ∈ SUB_A, per B-task b; base θ0, B-FT θ0+Δθ; deployed head w_A trained at base, FROZEN)
- ŵ_A = normalize(w_A)            deployed logistic decision axis (BASE), unit
- forget_old = AUC(w_A;F_A0,F_neg0) − AUC(w_A;F_A1,F_neg1)     DEPLOYED capability loss (metric of record, IDENTICAL to #7,
                                                                already confirmed non-saturated median 0.094 in [0.05,0.45])
- margin(F_A,F_neg) = <mean F_A − mean F_neg, ŵ_A>            centroid separation along the deployed axis
- ONAXIS_w = margin(F_A0,F_neg0) − margin(F_A1,F_neg1)        SIGNED margin collapse along w_A (positive = shrank → forgetting)
- dprime(F_A,F_neg) = margin / sqrt((var(F_A·ŵ_A)+var(F_neg·ŵ_A))/2)   noise-normalized discriminability along w_A
- DDPRIME = dprime0 − dprime1                                 secondary (AUC≈Φ(d'/√2), tracks AUC drop more faithfully than raw margin)
- ΔF_A = mean(F_A1) − mean(F_A0)
- OFFAXIS_w = ‖ΔF_A − <ΔF_A, ŵ_A> ŵ_A‖                        A-drift ORTHOGONAL to the deployed axis (harmless-hypothesis component)
- ONAXIS_absA = |<ΔF_A, ŵ_A>|                                 A-only on-axis drift magnitude (paired with OFFAXIS_w, same ΔF_A)
- DTHETA = ‖Δθ‖ (per-b)  ;  cosAB = <d_A, d_B> (base space, d_A = centroid-diff as in #1/#7)

## PRE-REGISTERED PREDICTIONS
P1b (necessary, partly definitional). ρ(ONAXIS_w, forget_old) ≥ +0.50. Signed margin collapse along the DEPLOYED axis governs
    forgetting. (If this fails even along w_A, the dichotomy is irreducible — magnitude governs regardless of projection.)
P2b (THE non-definitional content — decomposition). ON-axis strongly beats OFF-axis on the SAME ΔF_A:
    ρ(ONAXIS_absA, forget_old) − ρ(OFFAXIS_w, forget_old) ≥ +0.30, AND ρ(OFFAXIS_w, forget_old) < +0.30.
    Movement off the decision axis is harmless; only on-axis movement forgets → forgetting is DIRECTIONAL, not mere ‖Δθ‖ travel.
P3b (recovers what ‖Δθ‖ missed). Within-B (fixed b, fixed ‖Δθ‖), ONAXIS_w still predicts WHICH A-classes forget:
    within-B ρ(ONAXIS_w, forget_old) ≥ +0.40. ‖Δθ‖ is constant within-B so cannot; the on-axis projection can → it carries signal ‖Δθ‖ lacks.
P4b (loop closure, non-definitional crown). ρ(cosAB, ONAXIS_w) ≤ −0.30: anti-parallel B drags A along −w_A. Links plastic on-axis
    drag to the frozen −cos law from a fully plastic backbone.

## Outcome grid (locked)
- P1b & P2b & P4b hold: UNIFICATION RESCUED — #7's null was a projection artifact; forget = signed on-axis (w_A) drift; ‖Δθ‖ won #1
  only as the magnitude proxy of this directional quantity. Dichotomy dissolves into ONE law across frozen↔plastic. Top outcome.
- P1b holds but P2b FAILS (off-axis also strong): the on-axis result is tautological (any drift moves the score dist) → unification
  UNPROVEN; dichotomy stands. Honest.
- P1b FAILS (even along w_A): dichotomy is IRREDUCIBLE physics — plastic forgetting is magnitude-governed regardless of projection
  axis; #7 + #7b together CLOSE the unification dream cleanly (a strong, publishable negative).
- P4b sign wrong/null: plastic on-axis drag is NOT the frozen −cos mechanism → at best an analogy, not a bridge.

## Config (locked)
Identical to #7: build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT, FT_STEPS=500 FT_LR=3e-4, SUB_A=[0,6,12,18,24,30,36,42],
N_B=15 (full), NEG 90-99, B_POOL 50-89, 5 cls/B-task, seed 0. forget_old metric UNCHANGED from #7 (non-saturation already confirmed →
skip re-gate, run full directly). Only NEW: projection axis = ŵ_A (deployed head weight) + d' + on/off decomposition. Metric = Spearman
(Pearson alongside). No production system, no network, pure local CIFAR. Single seed for trend; 3-seed confirm only if P1b lands borderline.
