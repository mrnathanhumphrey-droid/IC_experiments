# PRE-REG — Probe-9 v2 (MEASURE-9 amendment). 2026-06-28.

Status: to be SHA256-LOCKED (`PREREG_PROBE9_AIMING_AXIS_v2_MEASURE9_LOCK.json`). **Supersedes** v1
(`PREREG_PROBE9_AIMING_AXIS_2026_06_28.md`, LOCK 11fe73b3). Everything in v1 carries UNCHANGED except the
ladder/selection, the prediction n-power, and cost, below. The harness, Ω definition, σ_sem, S statistic, per-stream
recalibration, verdict map, and re-decomposition set are inherited verbatim from v1.

## Amendment (one change, reason recorded)
**v1:** Stage-0 screen 9 candidates → down-select 5 spanning Ω → Stage-1 measure those 5.
**v2:** Stage-1 measures **EVERY candidate that clears the non-destructive gate (all 9 by default) — NO down-select.** The
screen becomes a pure **abort-gate** (ladder health only: does Ω span ≥0.15, any degenerate stream cos²θ_min>0.98 or
<1e-4); it does NOT pick the ladder.

**Why (recorded in chain):** the headline output is **Ω\* with a CI** — the crossover that turns Steele's static law +
the flat-minima refutation into a *regime law*. Five points underpower it twice: (a) Spearman at n=5 needs ρ≥0.90 for
p<0.05, whereas the locked ρ≥0.70 only becomes a real significance test at n≥7 (n=9 critical value 0.60); (b) locating Ω\*
with a tight CI on 5 points is hopeless. Measuring all 9 × 2 cells is ~18 instances vs 10 (~$50–60 more) for a
well-resolved S(Ω) curve and a defensible Ω\*. Highest-return spend in the probe. No hand-picking from the screen →
removes the exact discretion the locked rule existed to remove; with measure-all there is no down-select to bias.

## Updated predictions (n ≈ 9 streams per cell)
- **P1 (geometric):** OPT Spearman **ρ(S,Ω) ≥ 0.70** across the measured ladder (n=9 critical for p<0.05 is 0.60, so 0.70 is
  a genuine significance test now) AND high-Ω streams' S ≥ 3σ paired. **Ω\* with 95% CI** from the full 9-point S(Ω) fit
  (bootstrap over seeds, 1000 resamples; Ω\* = interpolated Ω where the fit's lower CI crosses 0).
- **P2 (fork):** Pythia flat (|ρ|<0.3, no stream S≥2σ → architecture axis stands) OR S∝Ω like OPT (ρ≥0.70 → unify into
  overlap×absorption). Both pre-committed.
- **P3 (partials, pooled ladder):** ρ(S,Ω | σ_sem) significant AND ρ(S,σ_sem | Ω) not → geometric axis (separates from
  Ramasesh). Reverse → semantic. Both → mixed. All pre-committed.

## Screen = abort-gate only (locked)
Fire MODE=screen on BOTH cells. Read the screen JSON solely as go/abort: `ladder_healthy` (Ω span ≥0.15, no degenerate
stream) AND per-stream recalibration will hold the band in measure mode. If a cell's ladder is unhealthy → VOID-soft
re-screen (do NOT hand-pick). If healthy → fan out MODE=measure for ALL 9 candidates × 2 cells.

## Updated cost
Screen ~1h/cell. Stage-1: **9 streams × 2 cells ≈ 18 instances** (~3h each, fan out in parallel) ≈ ~3–4h wall,
**~$120–150 total.** Analysis local/free.

## Inherited verbatim from v1 (LOCK 11fe73b3)
Cells (OPT/Pythia), task-A=SST-2 / phase-2=subj FIXED, V=top-128 Fisher, fp32, N=100; Ω = cos²θ_min principal-angle overlap
via cross-Gram (M_Ω=12), childhood-subspace per stream vs shared phase-2-from-w_A subspace; σ_sem = linear CKA frozen base;
S = N(phase2)−N(alttask) with REALIZED short-training directions (v5 relabel), randtop16 baseline, paired; per-stream
non-destructive lr recalibration (margin 0.015, K_CAL=20, band ±0.05); verdict map RECOVERED-on-axis / RE-AXIS RECOVERY /
BOUNDARY; bounded re-decomp {σ_sem axis, DSI-absorption-weighted Ω}, stop after 2; ops (a10 excluded, killswitch, heartbeat,
read-off-machine). Three-substrate ambition: Probe-9 (continuous Ω) + Librarian-Ω (free-lane, CPU $0) + Deepfake
seed-robustness (suppression-only negative control, CPU ~$0) → an Ω law across substrates, robust to the n=2-architecture
criticism because the axis is Ω, not architecture.
