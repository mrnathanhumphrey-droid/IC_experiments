# PRE-REG — Probe-6: displacement-absorption. 2026-06-25.

Status: LOCKED on write (see `PREREG_PROBE6_DISPLACEMENT_LOCK.json`). No GPU until explicit go.
First-class citizen of the post-floor regime: N≥100, paired within-run differentials, fp32,
per-step heartbeat + smart killswitch, GPU never a10.

## Motivation (what we already know)
- **Probe-5 (DEBRIS):** the forgetting damage is **coordinate displacement inside V** — phase-2 pushes the
  task-A-critical coordinates away from w_A; that displacement IS the forgetting (snapping them back = +0.10
  recovery, ~16σ). The surviving displaced deposit is debris; the original w_A coords carry task-A.
- **DSI (BANKED):** load-bearingness = the childhood *absorbing the gradient surge into V* (OPT), inertness =
  deflection (Pythia).
- **Missing bridge:** DSI measured absorption at the *gradient* level. Probe-5 defined damage at the
  *coordinate-displacement* level. **Nobody has measured whether the absorption actually reduces the
  displacement.** That is this probe.

## Hypothesis
**H1:** A load-bearing childhood reduces the end-state in-V displacement of the task-A coordinates caused by
phase-2 — in OPT (load-bearing) but not Pythia (inert).
**H2 (mediation):** Within OPT, the per-seed displacement-reduction explains the per-seed retention-protection
(positive correlation) — i.e. absorption→less-displacement→more-retention is one causal chain, not two
coincidences.

## Design (paired, per seed, per architecture cell)
Cascade: w_A (task-A deposit) → [childhood?] → phase-2 (task-B / depth) → w_end. NO phase-3, NO task-A retrain
(this is a displacement+retention read on the end-state, not a relearning test).

Two arms sharing the same seed and the same w_A checkpoint:
- **NoCH:** load w_A → phase-2(250). → measure D_noch, ret_noch.
- **CH:**   load w_A → childhood(400) → phase-2(250). → measure D_ch, ret_ch.

Per-carrier displacement (the Probe-5 damage quantity):
  **D = mean_carrier || P_V( w_end − w_A ) ||**,  V = top-128 task-A Fisher directions, same carrier set.
Retention: ret = SST-2 (task-A) accuracy at w_end (same eval set / posid,negid as prior probes).

Paired contrasts (floor cancels):
- **ΔD = mean_seed( D_noch − D_ch )**   (>0 ⇒ childhood reduced displacement = absorption)
- **Δret = mean_seed( ret_ch − ret_noch )**  (>0 ⇒ childhood protected = load-bearingness)
- **r = Pearson( D_noch−D_ch , ret_ch−ret_noch )** across seeds within a cell (mediation)

Secondary (decomposition, reported not gated): in-V vs out-of-V displacement
  ||P_V(Δ)|| and ||P_⊥(Δ)|| — confirms the protection is specifically the in-V part.

## Regime (locked — matches dispositive / Probe-5)
- lr_child 3e-5, 400 steps; lr_depth 5e-5, 250 steps. Held-out subject stream for phase-2. fp32.
- V = top-128 task-A Fisher curvature directions; carriers = standard carrier set per cell
  (OPT: q/k/v/out_proj, fc1, fc2; Pythia: analogous).
- **N = 100 seeds**, paired (same seed drives NoCH and CH).
- Cells: **opt350m** (facebook/opt-350m), **pythia410m** (EleutherAI/pythia-410m).

## Decision rule (pre-committed)
Per cell, with SE from the paired per-seed distribution (N=100 ⇒ SE≈σ/10):
- **ABSORPTION CONFIRMED (expected OPT):** ΔD > 0 at ≥2σ AND Δret > 0 at ≥2σ AND r > 0.
- **INERT (expected Pythia):** ΔD within ±2σ of 0 AND Δret within ±2σ of 0.
- **FULL DISSOCIATION:** OPT = absorption-confirmed AND Pythia = inert.
- **SURPRISE / REOPEN:** Δret > 0 at ≥2σ but ΔD ≈ 0 (childhood protects WITHOUT reducing displacement) →
  damage is NOT (only) in-V displacement; reopens the Probe-5 interpretation. This is the falsifier worth fearing.
- **NULL:** ΔD ≈ 0 AND Δret ≈ 0 in OPT → load-bearingness did not reproduce here; investigate regime drift.

## Validity gates (VOID if violated)
- w_A reproduce-anchor: OPT 0.823 ± 0.02, Pythia 0.753 ± 0.02 (matches floor battery / Probe-5). VOID if off.
- Phase-2 must actually erode (ret_noch < w_A acc by a sane margin); if ret_noch ≈ w_A, depth wasn't dispositive → VOID.
- Datasets load (SST-2 task-A; subj phase-2) — HARD-STOP on any dataset-load failure (no fabricated run).

## Ops (binding)
- GPU_PREF excludes a10: `gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_gh200,gpu_1x_h100_sxm5`. If only a10 free, WAIT.
- Per-step heartbeat `[hb]` every 25 steps; `[seed]` DONE line with running ΔD, Δret each seed.
- Smart killswitch armed on the printed instance id at fire. Read progress OFF the machine — never fabricate.
- Single cell per instance; launcher self-terminates on land. Confirm "picked ..." line before walking away.

## Cost
Lighter than Probe-5 (2 arms, no phase-3 retrain): est ~childhood + 2×phase-2 per seed × 100 ≈ ~3–4h/cell on
gh200/a100, ~$8–15/cell. Two cells.
