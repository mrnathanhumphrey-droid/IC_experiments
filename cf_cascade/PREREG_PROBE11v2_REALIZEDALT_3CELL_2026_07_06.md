# PRE-REG — Probe-11 v2: specificity with a REALIZED-path alttask, 3-cell (OPT/Pythia/BLOOM). 2026-07-06.

SHA256-lock before GPU. Supersedes Probe-11 v1 (`PREREG_PROBE11_BLOOM_SPECIFICITY`, LOCK 974b50c7), which was VOID-soft on
BLOOM: the 8v4 one-shot-gradient alttask direction SATURATED (dalttask_noch = 0.320 ± 5.6e-18, zero variance → floored
accuracy, N_alttask/S uninterpretable) AND the childhood was destructive (retA 0.768 < 0.780, no non-destructive lr). Both are
metric/dose problems, not a BLOOM result. v2 fixes both, and re-baselines OPT+Pythia on the identical hardened metric so the
3-cell SPECIFIC/GENERIC comparison is clean. This also delivers the "realized-path alttask" hardening the 8v4 brief flagged as
needed before print.

## Two fixes (the only changes vs 8v4 LOCK 0112d69b)
1. **Realized-path alttask.** Replace the one-shot averaged TREC gradient with a REALIZED direction: from the anchor, train
   TREC for STEPS_ALT=250 (= phase-2's STEPS_P2), take its in-V displacement dVa, unit-normalize per carrier, apply at magp
   (phase-2's per-carrier magnitude). Now phase-2 and alttask are constructed symmetrically (both realized 250-step in-V
   directions, magnitude-matched) → removes the saturation and the construction asymmetry.
2. **Extended-down lr grid.** LR_GRID = [5e-6, 7e-6, 1e-5, 1.5e-5, 2e-5, 3e-5] (was [1e-5…3e-5]). Margin rule (K_CAL=20,
   margin 0.015) picks the highest non-destructive lr. UNCHANGED for OPT (still 1e-5) and Pythia (still 3e-5) — those already
   clear at their old picks and the rule takes the max clearing lr; only BLOOM moves to a gentler non-destructive dose.

Everything else = 8v4 verbatim: task-A=SST-2; phase-2=subj (realized 250); childhood=mix5; V=top-128 task-A Fisher per carrier;
N=100 paired; S = N(phase2) − N(alttask); N(d)=dprop_NoCH−dprop_CH; N_randtop16 control. Carriers: OPT {q/k/v/out_proj,fc1,fc2},
Pythia {qkv,h_to_4h,4h_to_h}, BLOOM {self_attention.query_key_value, mlp.dense_h_to_4h, mlp.dense_4h_to_h} (Pythia-analog).

## Cells + anchors
opt350m (0.823), pythia410m (0.753), bloom560m (0.83; w_A built 2026-07-06, ckpt bloom560m_sst2_wA.pt). ANCHOR_TOL 0.02.

## Pre-registered read (same map, now on the clean metric)
- **SPECIFIC**: S > 0 (≥2σ) AND N_phase2 > N_randtop16 (≥2σ).
- **GENERIC**: S within ±2σ of 0 AND N_phase2 ≈ N_alttask > N_randtop16.
- **INTERMEDIATE**: significant but small/partial S.
- **Validation predictions (pre-committed):** OPT should REMAIN SPECIFIC under the realized-alt metric (S>0) — if the fix
  flips OPT, the original 8v4 specificity was construction-dependent (important, report loudly). Pythia should stay GENERIC.
  BLOOM classified for the first time (weak prior GENERIC from Megatron-style structure; open).
- **Validity gates:** anchor in-band; **alttask NOT saturated** (dalttask_noch SE > 0, i.e. real variance — the v1 failure
  mode); bite (mean dp2_n & dalt_n > 0.03); childhood non-destructive (retA ≥ w_A − 0.05, else VOID-soft).

## Verdict
3-point map on the load-bearing↔inert axis, on a hardened metric. Turns the n=2 capstone into n=3 (the #1 review gap) IF the
gates pass on all cells. If BLOOM is still destructive even at 5e-6, report that BLOOM's childhood cannot be made
non-destructive at this composition (a real limit, not a classification).

## Ops
Harness `cloud/probe11_realizedalt_lambda.py` (8v4 spine; realized-alt measure; extended grid; 3 cells). Launcher: probe11
<cell>. GPU_PREF excludes a10; killswitch. EARLY GATE per cell on the first `[seed]` line: confirm dalt_n has variance
(saturation fixed) + retA non-destructive. Fire all 3 cells (parallel). ~$45–55, ~5–6 h wall.

## Status
NOT built/fired. Next: lock → build harness → fire 3 cells → early-gate (dalt_n variance + BLOOM retA) → read S per cell.
