# PRE-REG — Probe-7: suppression vs reorganization (Probe-6 confound-kill). 2026-06-25.

Status: LOCKED on write (`PREREG_PROBE7_REORG_LOCK.json`). No GPU until explicit go.
Post-floor regime: N≥100, paired within-run differentials, fp32, per-step heartbeat + smart killswitch, never a10.

## Motivation
Probe-6 (`VERDICT_PROBE6_DISPLACEMENT_2026_06_25.md`) falsified "absorption = pinning": childhood PROTECTS task-A
(Δret +0.088, +7.6σ) but ends FARTHER from w_A in V (ΔD −0.129) with NEGATIVE per-seed correlation (r −0.322; more
in-V displacement → more protection). Two open problems Probe-7 closes:
1. **Step-count confound:** Probe-6's D measured displacement from w_A, but CH trains 400 extra (childhood) steps, so its
   absolute displacement is inflated by step count. We cannot attribute the larger movement to phase-2.
2. **Mechanism unresolved:** is protection (a) **suppression** — childhood makes *phase-2 itself* move less in V — or
   (b) **reorganization** — the *same/more* phase-2 movement does *less damage per unit* (robust/relocated solution)?

## Fix: measure PHASE-2-ONLY displacement, from each arm's pre-phase-2 anchor
Paired arms off the same w_A:
- **NoCH:** w_A → phase-2(250).  pre-phase-2 anchor = **w_A**.
- **CH:**   w_A → childhood(400) → phase-2(250).  pre-phase-2 anchor = **w_postchildhood** (snapshot before phase-2).

Per arm, with V = top-128 task-A Fisher directions **at w_A** (the Probe-5 damage subspace, fixed for both arms):
- **D2 = mean_carrier || P_V( w_end − w_anchor ) ||**  — displacement caused by phase-2 ALONE (step-count confound gone).
- **ret_anchor** = SST-2 acc at the pre-phase-2 anchor (NoCH: = w_A acc; CH: post-childhood acc).
- **ret_end** = SST-2 acc at w_end.
- **forget = ret_anchor − ret_end**  — task-A forgetting caused by phase-2.
- **robust = forget / max(D2, ε)**  — forgetting per unit of phase-2 in-V displacement (the reorganization quantity).

## Hypotheses
- **H_suppression:** childhood reduces phase-2's in-V movement → ΔD2 = D2_noch − D2_ch > 0, and this accounts for the
  protection (Δrobust ≈ 0).
- **H_reorg (favored by Probe-6):** childhood does NOT (mainly) suppress phase-2 movement (ΔD2 ≈ 0 or < 0) but makes each
  unit of movement less damaging → Δrobust = robust_noch − robust_ch > 0; the protection survives at equal displacement.

## Contrasts (paired, floor-robust)
- **ΔD2 = mean_seed( D2_noch − D2_ch )**  (>0 ⇒ suppression of phase-2 in-V movement)
- **Δforget = mean_seed( forget_noch − forget_ch )**  (>0 ⇒ childhood reduces phase-2 forgetting — the protection)
- **Δrobust = mean_seed( robust_noch − robust_ch )**  (>0 ⇒ reorganization: less damage per unit movement)
- **r_ch = corr( D2_ch_seed , forget_ch_seed )** within CH (reorganization ⇒ weak coupling: movement stops predicting damage)

## Decision rule (pre-committed), per cell, SE from paired per-seed dist (N=100 ⇒ SE≈σ/10)
- **REORGANIZATION:** Δforget > 0 (≥2σ) AND Δrobust > 0 (≥2σ) AND ΔD2 ≤ 0 (or within ±2σ of 0). Childhood protects at
  equal-or-greater phase-2 movement → confirms Probe-6's relocation read, confound removed.
- **SUPPRESSION:** ΔD2 > 0 (≥2σ) AND Δrobust within ±2σ of 0. Childhood protects by making phase-2 move less in V.
- **MIXED:** ΔD2 > 0 (≥2σ) AND Δrobust > 0 (≥2σ). Both contribute; report the split.
- **NULL / VOID-soft:** Δforget within ±2σ of 0 → protection did not reproduce; investigate regime (do not over-read).

## Validity gates (VOID if violated)
- w_A reproduce-anchor: OPT 0.823 ± 0.02 (Pythia 0.753 ± 0.02). VOID if off.
- **Childhood non-destructive:** CH post-childhood ret_anchor ≥ w_A_acc − 0.05; if childhood itself wrecked task-A the
  probe is meaningless → record + flag (childhood_nondestructive=False) and treat result as VOID-soft.
- Phase-2 must erode in NoCH: forget_noch > 0 by a sane margin.
- Datasets load (SST-2 task-A; subj phase-2; childhood streams) — HARD-STOP (VOID) on any load failure.

## Regime (locked — identical to Probe-5/6)
lr_child 3e-5 / 400; lr_depth 5e-5 / 250; held-out subj phase-2; V = top-128 task-A Fisher at w_A; standard carrier set;
fp32; **N = 100 seeds** paired (same seed drives NoCH and CH); cells **opt350m**, **pythia410m**.

## Ops (binding)
GPU_PREF excludes a10 (`gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_gh200,gpu_1x_h100_sxm5`); if only a10 free, WAIT. Per-step
`[hb]` every 25 steps; `[seed]` DONE with running ΔD2, Δforget, Δrobust. Smart killswitch armed on the printed id at fire.
Read progress OFF the machine — never fabricate. Single cell per instance; launcher self-terminates on land.

## Cost
~ Probe-6 + 1 extra eval/arm (ret_anchor): est ~3–4h/cell on a100/gh200, ~$10–15/cell.

## Note (out of scope, possible Probe-7b)
Direct relocation evidence — compute the task-A Fisher subspace at w_postchildhood (V_ch) and its overlap with V_A per
seed — is deferred to keep this a single clean question. Add only if Probe-7 lands REORGANIZATION and we want to show the
subspace literally moved.
