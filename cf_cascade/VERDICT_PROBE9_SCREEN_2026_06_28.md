# VERDICT — Probe-9 screen: Ω is NOT dialable by childhood-stream choice. 2026-06-28.

Pre-reg `PREREG_PROBE9_AIMING_AXIS_v2_MEASURE9` (LOCK 6e269459), MODE=screen. Abort-gate (`ladder_healthy`: Ω span ≥0.15,
no degenerate). Results `gateA/probe9_screen_{opt350m,pythia410m}.json`. M_Ω=12, screen_lr=3e-5.

## Result — ABORT-GATE FIRED (the measure-9 fan-out was NOT run; ~$120 saved)
Ω = cos²θ_min(childhood-displacement subspace, phase-2-from-w_A subspace), per candidate childhood stream:
- **Pythia:** all 9 streams cluster cos²θ_min **0.064–0.088** → span **0.025** (< 0.15). `ladder_healthy=False`, no degenerate.
- **OPT:** trending tighter/lower, **0.022–0.033** on the candidates measured → span ~0.011. (Screen completed/terminated.)

Different natural childhood streams produce **the same, low** geometric overlap with the phase-2 interference. **You cannot
sweep Ω by swapping the childhood stream.** The locked abort-gate caught this on a ~$10 screen before the fan-out.

## What it means (informative, not a bug)
1. **Childhood reorganization is ~orthogonal to the phase-2 interference** (Ω ~2–9%), and **clustered** — invariant to which
   natural stream is used.
2. **Reconciles with v4 (specificity S>0) by reframing the mechanism:** the childhood *displacement* does NOT align with the
   phase-2 direction (low Ω), yet the childhood *protects* against that direction (v4). So "what the childhood moves" ≠
   "what it protects against" — reinforcing reorganization-not-pinning, and showing the **aiming in v4 is NOT via
   displacement-overlap.** The Ω-as-childhood↔phase-2-displacement-overlap quantity is real but (a) not stream-dialable and
   (b) possibly not the axis that governs the aiming.
3. **Methodological:** the abort-gate (pre-committed ladder-health) did its job — same discipline that VOID-soft'd Probe-8
   v1/v2. Do NOT re-screen the same streams or fan out measure-9; the lever is wrong, not the run unlucky.

## Decision (user-approved: c then a)
- **(c) THIS verdict** banks the negative/reframe result.
- **(a) Re-lock with the PHASE-2 lever** (`PREREG_PROBE9b_PHASE2_LEVER`): Ω is set by the *interference* direction = the
  phase-2 task. Vary phase-2 (fixed childhood) to span Ω; DV = protection P(d) = forget_NoCH(d)−forget_CH(d); σ_sem(d) =
  task-A↔phase-2 similarity = **the Ramasesh axis**, now pitted head-to-head with Ω in partial correlations. This is a
  STRONGER geometric-vs-semantic test than the (moot) childhood-lever version.

Anchors unaffected and STAND: Anchor A (Librarian free-lane, `VERDICT_ANCHOR_LIBRARIAN_OMEGA`), Anchor B (Deepfake
antagonistic/suppression-only, `VERDICT_ANCHOR_DEEPFAKE_SEEDROBUST`). The Ω axis keeps two substrate endpoints; the
continuous sweep moves to the phase-2 lever (Probe-9b).
