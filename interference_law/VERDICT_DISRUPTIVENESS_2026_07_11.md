# VERDICT — Thread #1: what makes a task DISRUPTIVE? (PREREG bec29a92, locked 2026-07-11). Landed, $0 local GPU, n_B=24.

## RESULT: the predictor of representational forgetting is a MAGNITUDE — weight-space distance traveled ‖Δθ‖ (ρ=+0.65) — NOT a direction (cosine ~0, #6b) and NOT the feature drift I hypothesized (‖Δφ‖ ρ=+0.15, WRONG). Direction was dethroned; magnitude is crowned.

## The predictor table (B-intrinsic property vs disruptiveness = mean re-probe forgetting over 12 A-classes)
  predictor                  Spearman ρ    note
  ‖Δθ‖ (param update norm)     +0.653      ← STRONGEST. how far the fine-tune moved the weights.
  b_acc (B difficulty)         +0.398      pearson −0.408 (SIGN-INCONSISTENT) → unreliable, discard
  ‖Δφ‖ (feature drift, neutral)+0.152      ← my pre-registered HYPOTHESIS, FAILED
  drift_rank (participation)   −0.123      weak
  ‖g_B‖ (base grad norm)       −0.085      weak
median disruptiveness = 0.079 (partial-forgetting band, valid measurement)

## Reading it
- **The new king is ‖Δθ‖** (ρ +0.65, pearson +0.66, consistent; n=24 → ~3σ). Under FIXED steps/LR, B-tasks differ in how FAR they drag
  the weights from the pretrained init, and that weight-space travel distance predicts how much they damage prior capabilities.
  Representational forgetting ≈ HOW FAR the model moves, not WHICH WAY relative to the prior task.
- **My hypothesis was wrong (honest).** I pre-registered feature drift ‖Δφ‖ (on a neutral set) as the likely king; it came 3rd at
  ρ +0.15. Param-space distance beat feature-space distance ~4×. Notable DISSOCIATION: the weight movement that damages the A-classes
  is NOT well-captured by feature movement on a NEUTRAL held-out set — the damage is somewhat localized, not globally visible on
  neutral inputs. (‖Δθ‖ integrates the whole trajectory; neutral-‖Δφ‖ samples the wrong region.)
- b_acc's ρ/pearson sign flip (+0.40 / −0.41) = non-monotone/outlier-driven → discarded. grad-norm and drift-rank are ~null.

## THE DICHOTOMY (this completes the representational chapter)
- **READOUT level (frozen backbone):** forgetting is DIRECTION-governed — forget ∝ −cos(d_A,d_B), the calibrated 4-op law (#1–#5),
  quantitative and modality-general.
- **REPRESENTATION level (plastic backbone):** direction is DETHRONED (#6b: feature/Δθ/gradient cosines all ρ≈0); forgetting is
  MAGNITUDE-governed — forget ∝ ‖Δθ‖ (weight-space travel), DIRECTION-AGNOSTIC (#1, this).
Two regimes, TWO governing variables: a frozen readout cares WHICH WAY the new task points; a plastic representation cares only HOW
FAR it moves. The cosine law and the ‖Δθ‖ law are the two faces of interference across the frozen/plastic boundary.

## Caveats (honest)
n_B=24 (ρ +0.65 ~3σ, real but MODERATE — ~42% of variance; disruptiveness is ‖Δθ‖ + residual idiosyncrasy, not fully explained).
Single seed, ResNet18, full-FT at one strength (partial regime confirmed). ‖Δθ‖ under fixed steps/LR ≈ "how hard B pulls" — not a
trivial "more steps" artifact (steps fixed), but it does conflate task-hardness with init-distance. The ‖Δθ‖-vs-‖Δφ‖ dissociation
deserves a follow-up (measure drift ON the A-manifold — but that risks circularity; the neutral-set choice was the clean one).
Deferred: does ‖Δθ‖ restricted to specific LAYERS predict better (where does the damaging travel happen)?
