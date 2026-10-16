# PREREG — Thread #7: THE UNIFICATION. Is the frozen/plastic dichotomy ONE law seen in two projections?

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, to be SHA256-locked before any run.
Substrate: CIFAR-100 / ResNet18, FULL fine-tune (plastic backbone), $0 local GPU. Reuses probe_gradient_alignment (build_full,
full_ft, theta_vec, train_head) + probe_representational_law (features, ddir, prep, auc, load_banks, spearman, NEG/B_POOL).

## The crack this probes
Two laws that don't talk to each other:
  FROZEN readout : forget ∝ −cos(d_A,d_B)         [direction governs; 4-op law; ρ −0.64…−0.75]
  PLASTIC backbone: forget ∝ ‖Δθ‖                  [magnitude governs; #6b feature/Δθ/GRADIENT cosines all ρ≈0]
We named this a dichotomy. HYPOTHESIS: it is not two mechanisms — it is ONE, and the plastic-regime cosine nulled only because it
was measured in the full 11M-dim WEIGHT space (concentration of measure → everything near-orthogonal → real range, zero signal).
The damage to A can only flow through the directions that move A's OUTPUTS: A's functional tangent space. Measured THERE, direction
must re-emerge. The bridge quantity is the change in A's own centroid margin along its discriminative axis d_A — which, to first order,
IS the weight update projected through A's Jacobian (J_A·Δθ), obtainable WITHOUT forming J_A: just measure how A's features moved.

## The pivotal metric change (why this is not a re-run of #1/#6b)
#1 and #6b measured FRESH-probe forgetting (refit an A-head on the new features → information/separability loss, direction-agnostic
by construction). This probe measures the DEPLOYED-capability metric: the A-head trained at BASE is FROZEN, then evaluated on the
post-B-FT features. A frozen head fails specifically when features drift ALONG its decision axis — so this metric is where direction
of drift decides survival. Both metrics are logged; predictions are about the OLD-head (deployed) metric.

## Definitions (per A-class a, per B-task b; base backbone θ0, B-fine-tuned backbone θ0+Δθ)
- d_A = ddir(F_A0, F_neg0)             unit discriminative direction, BASE feature space
- w_A = frozen logistic A-vs-neg head trained at base (its weights never updated after)
- forget_old(a,b) = AUC(w_A; F_A0,F_neg0) − AUC(w_A; F_A1,F_neg1)      ← DEPLOYED capability loss (primary)
- forget_fresh(a,b)= probe_auc(F_A0,F_neg0) − probe_auc(F_A1,F_neg1)    ← information loss (continuity w/ #1,#6b)
- g0 = mean(F_A0) − mean(F_neg0);  g1 = mean(F_A1) − mean(F_neg1)       centroid-margin vectors
- ONAXIS  = <g0, d̂_A> − <g1, d̂_A>     on-axis margin COLLAPSE (positive = margin shrank along the decision axis)
- ΔF_A = mean(F_A1) − mean(F_A0)
- TOTAL = ‖ΔF_A‖                        total A-manifold drift magnitude
- OFFAXIS = ‖ΔF_A − <ΔF_A,d̂_A> d̂_A‖    drift ORTHOGONAL to the decision axis (harmless-hypothesis component)
- DTHETA = ‖Δθ‖  (per-b, shared over a)  replicates #1's crowned magnitude
- cosAB = <d_A, d_B>  (base space)       the ORIGINAL frozen-law coordinate

## PRE-REGISTERED PREDICTIONS (falsifiable, directional)
P1 — UNIFICATION (primary). ONAXIS margin-collapse predicts forget_old with STRONG positive Spearman, ρ(ONAXIS, forget_old) ≥ +0.50.
     Direction — which was NULL in raw θ-space (#6b ρ≈0) — RE-EMERGES once projected into A's functional tangent space.
P2 — HARMLESS OFF-AXIS (non-circularity guard). OFFAXIS drift predicts forget_old WEAKLY: |ρ(OFFAXIS, forget_old)| < ρ(ONAXIS,·),
     and ideally < 0.30. If total movement per se drove forgetting this would be false → this is what rules out the tautology
     "features moved so the probe broke." On-axis movement is the story; off-axis travel is (predicted) harmless.
P3 — MAGNITUDE IS THE SHADOW. DTHETA and TOTAL predict forget_old POSITIVELY (replicating #1's ‖Δθ‖ ρ≈+0.65) but WEAKER than ONAXIS.
     Interpretation: ‖Δθ‖ won #1 because it is the leaked magnitude of a directional quantity; decomposing recovers the signal it missed.
P4 — LOOP CLOSURE (the crown). The MECHANISM of the frozen −cos law is on-axis drag: anti-parallel B-training pulls A's centroid along
     −d_A. Prediction: ρ(cosAB, ONAXIS) < 0 (more anti-parallel B → more on-axis collapse) AND ρ(cosAB, forget_old) < 0, reproducing
     the frozen law's SIGN from a fully plastic backbone. This is the bridge: the same −cos, now mechanistically explained.

## Outcome grid (stated in advance)
- P1 holds (ρ≥+0.50) & P2 holds (off-axis weak): DICHOTOMY RETIRED — one law, forget = on-axis margin collapse; frozen is the
  J_A=identity special case. Highest-value outcome.
- P1 holds, P2 FAILS (off-axis also strong): metric is partly tautological (any drift breaks a frozen head) — REPORT as such, fall
  back to fresh-head or a margin-normalized metric; unification unproven.
- P1 FAILS (ρ<+0.50): the dichotomy is REAL physics — frozen and plastic forgetting are genuinely different mechanisms. Also a
  publishable, honest negative; kills the "one law" dream cleanly.
- P4 sign wrong/null: the plastic backbone does NOT reproduce the frozen −cos → the two regimes are only analogically related.

## Non-saturation guard (the recurring trap)
MODE=gate first (3 B-tasks): require median forget_old in the PARTIAL band [0.05, 0.45]. The OLD head is more fragile than a refit
head → expect LARGER forgetting than #6b's ~0.10; if it saturates (median >0.45, head collapses to chance) RETUNE FT_STEPS/FT_LR
DOWN before the full 8A×15B curve. Sign is only measurable in the partial regime — do not read ρ off a saturated run.

## Config (locked)
build_full ResNet18 IMAGENET1K_V1, fc=Identity. FULL fine-tune. FT_STEPS=500, FT_LR=3e-4 (as #6b gate-passed), tune DOWN if gate
saturates. SUB_A = [0,6,12,18,24,30,36,42] (8 A-classes). N_B: 3 gate / 15 full. NEG=90-99, B_POOL=50-89, 5 classes/B-task.
Seed 0 (single-seed for the trend; a 3-seed confirm only if P1 lands borderline). Metric of record: Spearman; Pearson reported alongside.
No production system touched; no network; pure local CIFAR.
