# PRE-REG amendment — Probe-11 v3: gentler-childhood re-run for BLOOM + OPT-1.3b. 2026-07-07.

SHA256-lock before GPU. Amends Probe-11 v2 (`PREREG_PROBE11v2_REALIZEDALT_3CELL`, LOCK 332e7029) / Probe-12
(`PREREG_PROBE12_SCALE_OPT13B`, LOCK ef4d761e). v2 fixed the alttask saturation, but the mix5 childhood was TOO STRONG to be
non-destructive on the new cells: BLOOM cleared the gate by only 0.002 (retA 0.782 vs 0.780), OPT-1.3b FAILED it (retA 0.832
vs 0.877, VOID-soft) — even at the gentlest grid dose 5e-6. Their S values (BLOOM −0.030, OPT-1.3b −0.037) are therefore on
damaged/edge baselines and uninterpretable. This re-runs ONLY those two cells with a childhood dose that can be
non-destructive.

## The only change
LR_GRID extended DOWN (env override): **[1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 1e-5]** (was [5e-6…3e-5]). STEPS_CHILD stays 400
(childhood "amount" held constant for comparability with OPT-350m). Margin calibration (K_CAL=20) picks the highest
non-destructive lr. Everything else = Probe-11 v2 (realized-alt, N=100, S=N(phase2)−N(alttask), per-cell anchor).

## Cells
bloom560m (anchor 0.83, a100), opt13b (anchor 0.93, h100 mem-safe).

## Pre-registered read + the under-dosing risk (named explicitly)
- If a non-destructive dose is found (no_nd=False, retA ≥ thr with margin) AND N_phase2 is meaningfully > 0 at that dose →
  the S sign is a VALID classification: S>0 SPECIFIC, S≈0 GENERIC, S<0 anti-specific/mis-aimed.
- **UNDER-DOSING caveat:** if the non-destructive dose is so gentle that N_phase2 ≈ 0 (childhood does ~nothing), then S≈0 is
  NOT "GENERIC" — it's "this cell's childhood cannot protect without being destructive." Report N_phase2 magnitude at the
  chosen dose; do NOT read a null-from-underdosing as a clean classification.
- If even 1e-6 is destructive (no_nd=True) → the cell's childhood cannot be made non-destructive at 400 steps: a real limit,
  report as such (not a classification).

## Ops
Harness `probe11_realizedalt_lambda.py` (LR_GRID now env-overridable) via env LR_GRID. Launcher: probe11 <cell> with
LR_GRID pass-through. Early gate per cell: no_nd flag + chosen_lr + first-seed retA (non-destructive?) + N_phase2 sign/size
(under-dosed?). GPU_PREF excludes a10; killswitch; hang-guard on the run-start (the v2 Pythia launcher-hang lesson).

## Status
NOT fired. Next: lock → env-override grid → fire BLOOM (a100) + OPT-1.3b (h100) → early-gate (non-destructive + not under-dosed).
