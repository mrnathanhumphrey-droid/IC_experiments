# PRE-REG — Probe-7b: non-destructive-childhood confirmatory (Probe-7 re-run). 2026-06-26.

Status: LOCKED on write (`PREREG_PROBE7B_NONDESTRUCTIVE_LOCK.json`). No GPU until explicit go.
Post-floor regime: N≥100, paired within-run differentials, fp32, per-step heartbeat + smart killswitch, never a10.

## Why
Probe-7 (`VERDICT_PROBE7...`, pending) confound-killed Probe-6 and found the protection is ~95% **reorganization**
(Δrobust +0.289/+10.1σ) and ~1% suppression (ΔD2 +0.0176/+2.7σ) → MIXED, reorg-dominant. BUT its pre-committed validity
gate **tripped**: `childhood_nondestructive=False` — childhood dropped task-A 0.823→0.743 (−0.080 > the locked −0.05
tolerance). Per our own discipline that result is VOID-soft. Probe-7b re-runs the identical mechanism measurement with a
childhood dose calibrated to be **non-destructive**, so the headline (reorganization) is dispositive — OR surfaces that
protection is entangled with mild destructiveness, which is itself a finding.

## Stage 1 — CALIBRATION (locked grid + locked selection rule)
Childhood-only, cheap. Grid over the childhood learning rate (the dose knob), steps fixed at 400:
  **G = [1e-5, 1.5e-5, 2e-5, 3e-5]**  (3e-5 = Probe-5/6/7 value, known destructive here).
K_CAL = 12 seeds (seeds 0–11). For each lr in G: load w_A → childhood(lr, 400) → eval task-A (SST-2, 300-ex set).
Record r_cal[lr] = mean post-childhood task-A retention over the 12 seeds.

**Selection rule (LOCKED, deterministic):**
  chosen_lr = max{ lr ∈ G : r_cal[lr] ≥ w_A_acc − 0.05 }   (the STRONGEST childhood that is still non-destructive;
  load-bearingness scales with dose, so take the largest dose under the non-destructive constraint).
  If NO lr qualifies → chosen_lr = min(G) and set flag `no_nondestructive_point=True` (report the full r_cal curve;
  Stage-2 still runs for the record but is marked VOID-soft + this becomes a "protection entangled with perturbation"
  result).

## Stage 2 — MAIN (identical to Probe-7, with chosen_lr)
Paired off w_A, N=100 seeds (0–99). V = top-128 task-A Fisher at w_A (the Probe-5 damage subspace), SAME both arms.
- **NoCH:** w_A → phase-2(250).  pre-phase-2 anchor = w_A.
- **CH:**   w_A → childhood(chosen_lr, 400) → [anchor = w_postCH] → phase-2(250).
- D2 = mean_carrier ||P_V(w_end − w_anchor)||;  forget = ret_anchor − ret_end;  robust = forget / max(D2, ε).
- Contrasts: **ΔD2** (suppression), **Δforget** (protection), **Δrobust** (reorganization); **r_ch** = corr(D2_ch, forget_ch).

## Decision rule (pre-committed), per cell, SE from paired per-seed dist (N=100 ⇒ SE≈σ/10)
First, the gate that failed before **must now pass**: `childhood_nondestructive` = (mean ret_anchor_ch ≥ w_A_acc − 0.05).
If it still fails on the full N=100 (calibration drift), mark VOID-soft. Given it passes:
- **REORGANIZATION (confirmatory target):** Δforget > 0 (≥2σ) AND Δrobust > 0 (≥2σ) AND ΔD2 ≤ 0 (or within ±2σ of 0).
  Reorganization confirmed with a clean non-destructive childhood → DISPOSITIVE.
- **SUPPRESSION:** ΔD2 > 0 (≥2σ) AND Δrobust within ±2σ of 0.
- **MIXED:** ΔD2 > 0 (≥2σ) AND Δrobust > 0 (≥2σ); report the counterfactual split (suppression-alone vs reorg-alone Δforget).
- **ENTANGLEMENT / NULL:** Δforget within ±2σ of 0 at the non-destructive dose → protection required the destructive dose;
  the load-bearing effect is entangled with mild task-A perturbation. Real finding, report as such.

## Validity gates (VOID if violated)
- w_A reproduce-anchor: OPT 0.823 ± 0.02 (Pythia 0.753 ± 0.02). VOID if off.
- Phase-2 must erode in NoCH: forget_noch > 0 by a sane margin.
- Datasets load (SST-2; subj phase-2; childhood streams) — HARD-STOP (VOID) on any load failure.
- `childhood_nondestructive` reported every run; gate as above.

## Regime (locked — identical to Probe-7 except childhood lr is calibration-selected)
steps_child 400; lr_depth 5e-5 / phase-2 250; held-out subj; V = top-128 task-A Fisher at w_A; standard carriers; fp32;
N=100 paired (same seed → NoCH & CH); cells **opt350m** (primary), **pythia410m** (deferred). ε=1e-6.

## Ops (binding)
GPU_PREF excludes a10; if only a10 free, WAIT. Per-step `[hb]`; Stage-1 prints `[cal] lr=.. r_cal=..`; `[seed]` DONE with
running ΔD2/Δforget/Δrobust. Smart killswitch armed on the printed id. Read OFF the machine — never fabricate. Single cell
per instance; launcher self-terminates on land.

## Cost
Stage-1 calibration ~20 min (48 childhood runs) + Stage-2 ~2.5h ≈ **~3h/cell on a100/gh200, ~$15/cell**.
