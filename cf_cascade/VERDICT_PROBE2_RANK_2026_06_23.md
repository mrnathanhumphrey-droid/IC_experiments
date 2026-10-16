# VERDICT — Probe 2 clean rank ablation (confound-killed). 2026-06-23. Single agent.

**What it settles:** does map load-bearingness live in the tiny task-A curvature CORE (K_E90 ≈ 14 OPT / ≈ 9 Pythia,
`basisinv_*.json`) or in the broad V shell? The vacuums piggyback couldn't answer it — its B−D_k read projected the
childhood GRADIENT out per K, so childhood learning (lossdrop) was NOT matched (6.66→15.41), confounding rank with
learning-amount. This probe runs childhood ONCE per seed and truncates the deposit to top-k V-dims re-normalized to full
in-V magnitude → lossdrop identical across k → confound dead by construction. Pre-reg `PREREG_PROBE2_RANK_CLEAN_2026_06_23.md`.
Harness `cloud/probe2_rank_lambda.py`. Both gh200 cells landed 2026-06-23, self-terminated clean.

## Confound confirmed dead (the point of the probe)
`childhood_lossdrop` flat across all k: OPT 6.68 ± (SE tiny), Pythia 8.78 ± 0.03. Single childhood/seed → identical
regardless of which k we read. The vacuums confound (which ran lossdrop 6.66→15.41) is gone.

## Anchors held (reproduce gate)
B−A at k=128 (= arm B, full deposit): OPT +0.080 (exp +0.089), Pythia +0.062 (exp +0.073) — both within the 0.03/floor
band → ANCHOR_OK both. retA: OPT 0.581, Pythia 0.557.

## The rank ladder (Bk − A; magnitude-matched; within-run paired across k → floor-robust)
| k | OPT Bk−A | Pythia Bk−A |
|---|---|---|
| 0 (out-of-V only = arm D) | +0.034 ±0.016 | +0.006 ±0.019 |
| core (14 / 9) | +0.021 ±0.027 | +0.032 ±0.017 |
| 32 | +0.058 ±0.022 | +0.040 ±0.019 |
| 64 | **+0.104 ±0.018** | +0.049 ±0.019 |
| 128 (full = arm B) | +0.080 ±0.020 | +0.062 ±0.017 |

## HEADLINE — load-bearingness lives in the broad V SHELL, NOT the tiny K_E core (BOTH families)
- **Pythia (inert):** protection climbs smoothly/monotonically with rank; the K_E≈9 core carries only ~half (+0.032 of
  +0.062); still climbing at 128 → DISTRIBUTED across the shell.
- **OPT (load-bearing):** the K_E≈14 core is a weak, high-variance DIP (+0.021, SE 0.027, below even k=0); protection
  PEAKS at ~k64 (+0.104) and saturates before 128 (k64 ≈ k128, overshoot ~1 SE — suggestive, not hard) → carried by an
  INTERMEDIATE shell (~top-32–64 Fisher dims), not the core.
- **basisinv enrichment is INCIDENTAL to protection:** the deposit concentrates ~10× in the curvature core, but its
  protective power comes from the broader shell. The "tiny core (basisinv) vs load-in-tail (vacuums)" tension dissolves —
  different objects (where the deposit SITS vs where load-bearingness COMES FROM), now confound-free.

## THE FLIP — clean Probe 2 overturns the vacuums "Pythia protection is out-of-V" read
Vacuums had Pythia D_K−A flat ~0.08 across all K → "K-invariant / out-of-V." That was the lossdrop confound. Clean k=0
(natural full childhood, in-V deposit removed post-hoc) → Pythia out-of-V deposit alone protects **+0.006 ≈ nothing**.
Pythia protection is IN-V and distributed; the vacuums flatness was an artifact of the unmatched-childhood arm. This is
exactly why we killed the confound — the answer changed.

## Cross-arch texture (clean)
- **Out-of-V contribution:** OPT k0 +0.034 (~40% of full protection lives out-of-V) vs Pythia k0 +0.006 (~0%). OPT has
  real out-of-V protection; Pythia has essentially none.
- **Core deposit-specificity (swap at core rank, matched magnitude):** Pythia core IS deposit-specific (real +0.032 ≫
  swap +0.006); OPT core is NOT (real +0.021 ≈/< swap +0.031). OPT's deposit-specificity — banked at full-V in the DSI
  swap-gate — emerges only with more rank, not at the K_E core.
- **Shape:** OPT non-monotonic (intermediate peak); Pythia monotone. Load-bearing OPT concentrates its protection in a
  mid-rank shell and overshoots the full deposit when energy is packed there; inert Pythia spreads thin and never
  concentrates.

## Caveats / discipline
- Ladder comparisons are within-run paired (same childhood/seed across k) → the ~0.066 run-to-run floor largely cancels;
  anchors reproduced; do not over-read absolute single-k magnitudes.
- Magnitude-match renorm at very low k amplifies few directions (possible OOD) — visible as OPT's high-variance k14 dip;
  read the core point as "insufficient + unstable," not a precise level.
- The OPT k64>k128 overshoot is ~1 SE: treat as "saturates by ~64," not a hard "peak above full."

## Threads now consolidated by this result
1. DSI swap-gate (banked): absorb-is-load-bearing, deposit-specific at full-V. → Probe 2: that specificity needs rank
   (not the core); OPT load-bearing shell ≈ where absorption concentrates.
2. basisinv (banked): tiny curvature core, deposit ~10× enriched there. → Probe 2: core ≠ protection source (incidental).
3. vacuums (now superseded for the rank read): Pythia "out-of-V" was confounded → FLIPPED to in-V distributed.
4. scale de-confound / collision-not-exodus (banked): deposit stays curvature-locked at scale — consistent with the
   deposit sitting in the core while protection lives in the shell.
