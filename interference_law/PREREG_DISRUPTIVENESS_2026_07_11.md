# PRE-REG — Thread #1(fun): what makes a task DISRUPTIVE? Finding the NEW predictor of representational forgetting after cosine was dethroned. 2026-07-11.

SHA256-lock before running. #6b proved NO direction-alignment cosine (feature/Δθ/gradient) predicts representational forgetting —
BUT forgetting varies ~2–3× across B-tasks. So a B-INTRINSIC property (measured WITHOUT reference to any A) governs disruptiveness.
This hunts that predictor. "We killed the old king (cosine); who is the new one?"

## Setup (reuse #6b full-FT machinery)
CIFAR-100 + ResNet18 full fine-tune (400 steps / lr 1e-3 = #6b's partial-forgetting regime). Class allocation (disjoint):
A-classes 0–11 (measure forgetting on), neg-pool 90–99 (re-probe), B-pool 50–89 (sample B-tasks), NEUTRAL drift-probe set 25–40
(for ‖Δφ‖ — disjoint from A/neg/B to avoid circularity). N_B ≈ 24 B-tasks (each = 5 sampled B-pool classes).

## Outcome & candidate predictors
- **DISRUPTIVENESS(B)** = mean over the 12 A-classes of re-probe forgetting (acc_A0 − acc_A1). One number per B-task.
- **Candidate B-intrinsic predictors (all measured on B or on the NEUTRAL set, never on A):**
  1. **FEATURE DRIFT ‖Δφ‖** = mean over the NEUTRAL probe set of ‖φ_B(x) − φ_base(x)‖ / ‖φ_base(x)‖ — how much the representation
     moved (the hypothesized new king).
  2. **DRIFT RANK** = effective rank (participation ratio) of the Δφ matrix on the neutral set — is the drift concentrated (low rank)
     or broad (high rank)?
  3. **PARAM UPDATE ‖Δθ‖** = L2 of (θ_after − θ_base).
  4. **GRAD NORM ‖g_B‖** = L2 of the base B-loss gradient.
  5. **B DIFFICULTY** = B final train accuracy (and loss drop) — do harder B move more?

## Pre-registered read (locked before running)
- **THE NEW PREDICTOR:** rank the 5 candidates by |Spearman ρ| with disruptiveness. Report the winner + the full table.
- **HYPOTHESIS (feature drift):** ‖Δφ‖ on the NEUTRAL set predicts disruptiveness with ρ > 0.5 → "representational forgetting ≈
  how far B moves the representation, roughly uniformly across capabilities" — the new law that REPLACES cosine. If ρ > 0.7, strong.
- **RANK story:** if DRIFT RANK adds predictive power beyond ‖Δφ‖ (broad drift forgets more than concentrated drift of equal norm) →
  forgetting is about BROAD reorganization (ties to Probe-18's broad-subspace). Report the partial correlation.
- **NULL:** if NO B-intrinsic property predicts disruptiveness (all |ρ|<0.3) → disruptiveness is not B-intrinsic either → forgetting
  is A×B-pair-idiosyncratic in a way neither the cosine NOR B-properties capture (deep null; report honestly).
- **CIRCULARITY GUARD:** ‖Δφ‖ measured on the NEUTRAL set (classes 25–40), disjoint from the A-classes forgetting is measured on →
  a correlation is NOT mechanically forced.

## Confounds (locked)
1. Circularity: neutral drift-probe set disjoint from A/neg/B (above).
2. Saturation: reuse #6b's partial-forgetting config (400/1e-3); report median forgetting in-band.
3. n=24 B-tasks is modest → report ρ with the understanding that only a STRONG predictor (|ρ|>0.5) is a claim; small |ρ| = suggestive.
4. Single seed for the B fine-tunes (the cross-B TREND is the claim); note.

## Compute / ops
GPU: 24 full-FTs (ResNet18) + feature extraction. Est ~8–12 min. Run AFTER the phase-diagram background job frees the GPU (avoid
contention/OOM). `probe_disruptiveness.py` (reuse probe_gradient_alignment full_ft/features/grad_vec/theta_vec). py_compile + tiny
smoke now; full run once GPU is free. Output `out/disruptiveness.json`.

## Status
PREREG written. Next: SHA256-lock → build → tiny smoke → (wait for phase diagram to finish) → full run → read the new-predictor
table → verdict + bank.
