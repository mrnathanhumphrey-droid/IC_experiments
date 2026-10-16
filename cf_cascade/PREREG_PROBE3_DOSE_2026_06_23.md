# PRE-REG — Probe 3: childhood dose-response. 2026-06-23. LOCK below.

**Question (pre-committed):** does deposit STRENGTH (childhood training amount) map monotonically onto map
load-bearingness — does protection switch on / shift signature as the deposit grows?

**Band (pre-committed):** OPT-350m, Pythia-410m ONLY (live band; dead at 1B — do NOT run at 1B).

**Matched dispositive regime:** lr_child 3e-5, lr_depth 5e-5 / 250 steps, held-out = subj, N=30 paired seeds,
V = top-128 task-A Fisher at w_A. Childhood stream = {agnews, dbpedia, yahoo, newscat, emotion} (same as DSI), FIXED
across doses (only the step COUNT varies).

**Dose ladder (pre-committed, LOCKED):** childhood steps D ∈ {100, 200, 400, 800}. D=400 is the dispositive anchor.

**Arms per dose:** A (no deposit, dose-independent baseline) / B_D (full deposit at dose D) / Dproj_D (deposit with
childhood gradient projected out of V at dose D). Plus SWAP at the dispositive dose D=400 (curvature-matched random
deposit, in-V magnitude matched) — deposit-specificity reference.

**Reads per dose (endpoint, per seed → mean ± SE):** retA, retB_D, retD_D, **B−A**, **B−D**, mapfrac_B_D
(‖P_V(deposit)‖²/‖deposit‖²), A_forget_D (severity), childhood_lossdrop_D.

**Reproduce-anchor (GATES the read):** at D=400, B−A must reproduce the banked OPT +0.089 / Pythia +0.073 within 0.03,
else harness invalid.

**HARD-STOP (pre-committed):** a dose D is flagged INVALID (excluded from the monotone read) if its A_forget leaves the
matched-band by >0.05 vs the D=400 anchor — a dose effect must not be a regime-severity effect. Report the flag, do not
silently drop.

**Pre-committed outcomes:**
- **Monotone shift** → B−D (and signature) rises/transitions smoothly with dose: load-bearingness is graded in deposit
  strength.
- **Threshold** → B−D flips on at a single dose: load-bearingness switches on at a strength.
- **Flat** → B−D dose-independent across {100…800}: deposit magnitude irrelevant to load-bearingness type (it is set by
  architecture, not strength).

**Swap read:** at D=400, real B−A ≫ swap B−A → deposit-specific (consistent with the banked full-V swap-gate); real ≈
swap → generic curvature.

**Floor discipline:** B−A / B−D are within-run paired (same seeds) → the ~0.066 run-to-run floor largely cancels in the
dose-to-dose contrast; demote absolute single-dose levels. Bank the dose TREND, not a single dose's magnitude.

Harness `cloud/probe3_dose_lambda.py` + launcher `cloud/lambda_launch_generic.py probe3`. CELL ∈ {opt350m, pythia410m}.
