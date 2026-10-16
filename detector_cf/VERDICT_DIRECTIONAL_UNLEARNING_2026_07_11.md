# VERDICT — Thread #3: directional unlearning (PREREG 0d4c05d3, locked 2026-07-11). Landed, $0 local CPU.

## RESULT: unlearning COLLATERAL follows the SAME cosine law as continual-learning FORGETTING.
You can surgically erase a capability iff it is ANTI-PARALLEL to what you keep; a capability ALIGNED with what you keep cannot be
cleanly excised. Same discriminative-direction cosine that governs forgetting (#1) governs unlearnability.

## Setup
Universal FAKE-vs-REAL head (frozen backbone, 128-d) over swap/diffusion/celebdf/dfdc. Erase ONE paradigm, measure collateral on the
other three. DIRECTIONAL = remove head's response to û = (d_target orthogonalized to the OTHER paradigms' directions). NAIVE = relabel
target→real + finetune. RANDOM = matched-magnitude removal along a random direction (control).

## Two targets, opposite outcomes — the law in action
### Erase DIFFUSION (anti-parallel to the family, cos ≈ −0.9): CLEAN surgery
  base:        swap 0.656  diffusion 0.904  celebdf 0.908  dfdc 0.934
  directional: swap 0.666  diffusion 0.204  celebdf 0.870  dfdc 0.894   (diffusion CRUSHED −0.70; others drop only ~0.04; swap untouched)
  → diffusion is anti-parallel to everything it neighbours, so removing its detection SPARES/──helps the family. Cleanly excisable.

### Erase SWAP (aligned with the family celebdf/dfdc, cos +0.6/+0.8): NO clean surgery
  base:        swap 0.656  diffusion 0.904  celebdf 0.908  dfdc 0.934
  directional: swap 0.477  diffusion 0.885  celebdf 0.842  dfdc 0.853   (swap only to 0.48, and family dragged −0.066/−0.081)
  naive:       swap 0.590  ...            (barely erases — swap rides the family's shared direction)
  → swap shares its direction with the family, so ANY method that removes swap also damages celebdf/dfdc. Entangled = inseparable.

## THE LAW: collateral ∝ +cos(target, protected)  (the unlearning dual of forget ∝ −cos)
- Family collateral when erasing DIFFUSION (cos −0.9 to family): ~0.04.  When erasing SWAP (cos +0.7 to family): ~0.07–0.08.
- A capability's SURGICAL UNLEARNABILITY is set by its cosine to what you protect: anti-parallel → clean removal; aligned → entangled,
  no clean removal. This is the SAME cosine that predicts forgetting magnitude in #1 — one geometry governs both learning and unlearning.
- Practical upshot: you can cleanly delete an ISOLATED/anti-parallel capability (e.g. a distinct generator family) from a detector's
  head; you CANNOT cleanly delete one that shares its direction with capabilities you must keep (the swap-family) — the route-arm
  entanglement, now in the unlearning domain.

## Pre-registered PREDICTION — partially met, honestly corrected
- Pre-reg predicted DIRECTIONAL would have the HIGHEST specificity. FOR AN ENTANGLED TARGET (swap) THIS FAILED: specificity ranked
  RANDOM 7.83 > NAIVE 5.96 > DIRECTIONAL 3.23. **Cause: the specificity ratio (swap_drop / other_drop) REWARDS NO-OPS** — RANDOM barely
  touches anything (drop 0.047) so its ratio is high while it accomplishes nothing. The metric is flawed for weak interventions;
  the honest read is raw erase-vs-collateral: DIRECTIONAL is the ONLY method that actually pushes swap to chance (0.48 vs naive 0.59).
- FOR AN ISOLATED TARGET (diffusion) the picture is clean: directional AND naive both excise it (0.20/0.19) with tiny collateral
  (specificity 30/51), random does nothing (−0.85). Isolated capabilities are removable by either method; the DIFFERENTIATOR is
  entanglement, not method.

## Caveats (honest)
Head-level only (frozen backbone) — erases the linear readout's response, not a representational capability. Filename-hash split
(not identity-disjoint) → absolute AUCs inflated; deltas/collateral-structure are the claim. The pre-registered SPECIFICITY metric is
flawed (rewards no-ops) — superseded by raw erase+collateral; noted, not hidden. û depends on the protected-set choice. Single seed.
Not a certified-removal system — a mechanism demo linking unlearning to the arc's directional geometry.
