# PRE-REG — Thread #3: DIRECTIONAL UNLEARNING — surgically erase one paradigm's detection via the orthogonal-complement of its direction. 2026-07-11.

SHA256-lock before the run. Turns the arc's mechanism OFFENSIVE: instead of fighting directional interference, USE it to
surgically delete a capability. If forgetting is directional (the whole arc), then a target capability should be erasable by
removing exactly the head's response to that capability's UNIQUE direction — sparing capabilities whose directions are orthogonal.
A clean $0 machine-unlearning demonstration in the warm detector sandbox.

## HARD ISOLATION (unchanged)
Same detector sandbox. Frozen `prod_models/prod_swap_seed0.pt` (read-only). Crops globbed read-only from `D:/corpus/face_swap_corpus`.
Writes only ./out/. NEVER touches the live prod ensemble / served detector. $0 LOCAL CPU. Reuses probe_detector_seq machinery.

## Setup
- Frozen backbone → 128-d features. Train ONE universal head H: FAKE (any of swap/diffusion/celebdf/dfdc) vs REAL (shared ffpp/real).
- **Target to erase = SWAP detection** (make swaps pass as real: swap AUC → 0.5), while PRESERVING diffusion/celebdf/dfdc detection.
- Per-paradigm discriminative directions d_p = normalize(mean(feat_p) − mean(feat_real)) (train split).

## Three unlearning methods (all start from the SAME H; measured on the SAME test sets)
1. **DIRECTIONAL (the proposed surgical method, closed-form, $0):** û = normalize(d_swap − proj of d_swap onto span{d_diffusion,
   d_celebdf, d_dfdc}) = the part of swap's direction UNIQUE to swap (orthogonal to the others). Remove the head's response to it:
   w' = w − (w·û)·û. No retraining. Predicted: swap detection collapses (its unique signal is gone), others intact (their
   directions untouched).
2. **NAIVE (baseline unlearning):** relabel swap examples as REAL (label 0) and continue-train H a few steps. Standard "just fine-tune
   it away." Predicted: erases swap BUT collateral-damages celebdf/dfdc (they share direction with swap, cos > 0) — less specific.
3. **RANDOM (mandatory control):** remove the head's response to a RANDOM unit direction r̂ of matched removed-magnitude
   (|w·r̂| ≈ |w·û|): w' = w − (w·r̂)·r̂. Predicted: neither erases swap nor spares others in an aimed way (diffuse/no effect).

## Pre-registered read (locked before the run)
- **SURGICAL SUCCESS:** DIRECTIONAL drives swap AUC to ≤ 0.60 (near chance) while keeping diffusion/celebdf/dfdc AUC within 0.05 of
  their pre-unlearning values. **Specificity = (swap AUC drop) / (mean other-paradigm AUC drop) is HIGHEST for DIRECTIONAL**, clearly
  above NAIVE and RANDOM.
- **NAIVE is less specific:** NAIVE erases swap but drops celebdf/dfdc by > 0.05 (shared-direction collateral) → lower specificity.
- **RANDOM is the floor:** RANDOM neither erases swap (AUC stays high) nor is aimed → specificity ≈ 1 (undifferentiated) or worse.
- **NULL / failure to report honestly:** if DIRECTIONAL also craters the other paradigms (swap's unique direction isn't actually
  separable from theirs) → surgical unlearning fails, report the collateral. If removing û does NOT erase swap (swap detection lives
  in the SHARED direction, not the unique one) → the capability isn't in the orthogonal complement; report and interpret (swap may be
  inseparable from the swap-family, echoing the route-arm finding that the family is only moderately aligned).

## Caveats (honest)
Head-level only (frozen backbone) — this erases the linear READOUT's swap response, not a representational capability. Filename-hash
split (not identity-disjoint) → absolute AUCs inflated; the DELTAS + specificity are the claim. Single seed. The orthogonal-complement
û depends on which "others" we protect (diffusion/celebdf/dfdc) — a different protected-set changes û (reported). This is a mechanism
demo, not a deployable unlearning system (no certified-removal guarantees).

## Ops
`probe_directional_unlearn.py` reusing probe_detector_seq (load_backbone/extract/Head/auc/discriminative_dir/train_head). Closed-form
DIRECTIONAL + few-step NAIVE + RANDOM control; per-paradigm AUC before/after each; specificity. py_compile + --smoke before full.
$0 local CPU, seconds. Output `out/directional_unlearn.json`.

## Status
PREREG written. Next: SHA256-lock → build → --smoke → full → read specificity (DIRECTIONAL vs NAIVE vs RANDOM) → bank.
