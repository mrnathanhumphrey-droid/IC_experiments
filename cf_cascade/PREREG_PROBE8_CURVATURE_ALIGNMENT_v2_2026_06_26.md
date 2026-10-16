# PRE-REG — Probe-8 v2: targeted vs global curvature alignment (BOUNDED estimator). 2026-06-26.

Status: LOCKED on write (`PREREG_PROBE8_CURVATURE_ALIGNMENT_v2_LOCK.json`). No GPU until explicit go.
**Supersedes** `PREREG_PROBE8_CURVATURE_ALIGNMENT_2026_06_26.md` (LOCK b33ab105).

## Why v2 (estimator bug, fixed before any valid result)
v1 fired (OPT, a100) and was KILLED at seed ~20 because its metric was numerically degenerate, NOT because of any finding.
v1 used **R = rise_phase2 / rise_rand** on **raw cross-entropy** loss-rises. Observed: the random-direction CE rise was tiny
and noise-dominated — frequently ≈0 or NEGATIVE — so the ratio exploded to 1e6–1e7 and `log R` clamped to ±floor; the
phase-2 CE rise also exploded (model breaks → CE→8). `delta_logR` (+26) was pure artifact. **No results were used to choose
the v2 metric** — this is fixing a broken instrument, not post-hoc selection. v1 lock b33ab105 is VOID (estimator bug).

## The fix: bounded accuracy-drops, differences not ratios
Everything else (design, calibration, regime, cells) is identical to b33ab105. Only the per-perturbation readout and the
contrast statistic change:
- Readout = **accuracy drop** (bounded), not CE loss. For a perturbation Δ applied to anchor a:
  drop(Δ) = acc_A(a) − acc_A(a + Δ).
- **dphase2,c = drop(Δ_V,c)** — drop along the realized phase-2 in-V displacement.
- **drand,c = mean_{K_RAND=5} drop(R)** — R random in V, per-carrier ‖R_n‖ = ‖Δ_V,c,n‖ (magnitude-matched).
- **aniso_c = drand,c − dphase2,c** — anisotropy: >0 ⇒ the phase-2 direction is LESS damaging than a random direction of
  equal size (task-A is relatively robust to where phase-2 actually goes).
No ratios, no logs, no division — all quantities bounded in [−1, 1].

## Metrics & hypotheses (paired, per cell)
- **Targeting (novel):** Δaniso = mean_seed( aniso_CH − aniso_NoCH ). >0 ⇒ childhood makes the phase-2 direction relatively
  SAFER than random — direction-specific (targeted) protection beyond generic flattening.
- **Global flatness (literature):** g = mean_seed( drand_NoCH − drand_CH ). >0 ⇒ childhood reduces damage even in a RANDOM
  direction (generic basin widening).
- **Baseline anisotropy:** aniso_NoCH (does phase-2 already flow where task-A is robust, before any childhood?).

## Decision rule (pre-committed), per cell, SE from paired per-seed dist (N=100 ⇒ SE≈σ/10)
- **TARGETED (further-from-lit):** Δaniso > 0 (≥2σ).
- **GLOBAL-ONLY (matches literature):** g > 0 (≥2σ) AND Δaniso within ±2σ of 0.
- **BOTH:** g > 0 (≥2σ) AND Δaniso > 0 (≥2σ).
- **NULL:** neither ≥2σ.
- **ARCHITECTURE-DEPENDENT TARGETING (headline if it lands):** Δaniso>0 in OPT AND Δaniso≈0 in Pythia.

## Validity gates (VOID if violated)
- w_A reproduce-anchor: OPT 0.823 ± 0.02 (Pythia 0.753 ± 0.02).
- childhood_nondestructive at chosen_lr (post-childhood task-A ≥ w_A−0.05); flag + VOID-soft if not (as Probe-7b).
- **Perturbation sensitivity:** mean drand > 0.01 (a random in-V move of the realized phase-2 magnitude must actually dent
  task-A); if drand ≈ 0 the magnitude is too small to read curvature → flag `insensitive=True`, VOID-soft.
- Datasets load — HARD-STOP (VOID) on failure.

## Regime (locked — identical to b33ab105 except readout)
Stage-1 calibration: lr_child grid [1e-5,1.5e-5,2e-5,3e-5]×12 seeds, chosen_lr = max lr with post-childhood task-A ≥
w_A−0.05. Stage-2: steps_child 400 @ chosen_lr; phase-2 250 @ lr 5e-5; V = top-128 task-A Fisher @ w_A; standard carriers;
fp32; **K_RAND=5**; **N=100** paired; cells **opt350m** (primary) + **pythia410m** (cross-arch).

## Ops (binding)
GPU_PREF excludes a10; watcher available if capacity-blocked. Per-step `[hb]`; `[cal]`; `[seed]` DONE with running Δaniso,
g, aniso_NoCH/CH. Smart killswitch armed. Read OFF the machine — never fabricate. Single cell/instance; self-terminate.

## Cost
≈ v1 (~3h/cell, ~$15/cell), two cells. v1 wasted ~$1 (killed at seed 20 on the estimator bug).
