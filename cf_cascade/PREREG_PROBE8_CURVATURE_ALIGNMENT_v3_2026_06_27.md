# PRE-REG — Probe-8 v3: targeted vs global curvature alignment, CLEAN dual-cell. 2026-06-27.

Status: LOCKED on write (`PREREG_PROBE8_CURVATURE_ALIGNMENT_v3_LOCK.json`). No GPU until explicit go.
**Supersedes** v2 (`PREREG_PROBE8_CURVATURE_ALIGNMENT_v2_2026_06_26.md`, LOCK 93f26963) and v1 (b33ab105, VOID).
Post-floor regime: N≥100, paired within-run differentials, fp32, per-step heartbeat + smart killswitch, never a10.

## Why v3 (fix two flags from v2, before banking the headline)
v2 (OPT, N=100, bounded estimator) gave a STRONG targeted signal — **Δaniso = +0.156 ± 0.011 (+14.1σ)**, global flatten
g = −0.0015 (≈0): childhood neutralized the phase-2 direction (dphase2 0.131→−0.024) without widening the bowl. BUT it
tripped TWO pre-registered gates, so it is suggestive, not dispositive:
1. **childhood_nondestructive=False by 0.010** — calibration kept selecting the boundary dose (chose 2e-5; full-N task-A
   0.763 vs 0.773 bar). The 12-seed calibration systematically over-estimates retention vs the full-100 by ~0.005–0.010.
2. **insensitive=True** — random in-V perturbations of the realized phase-2 magnitude gave ~0 task-A drop in BOTH arms, so
   the random baseline sat at the floor. Targeting was carried entirely by the phase-2-direction term, with random as a
   clean zero but not a resolving reference.
v3 fixes both: (a) a calibration MARGIN + more calibration seeds → a dose that is RELIABLY non-destructive on the full N;
(b) a perturbation-MAGNITUDE SWEEP → lift the random baseline off the floor and test whether targeting is magnitude-robust.
Run BOTH cells (OPT primary + Pythia cross-arch) clean. v2 is NOT banked as the verdict; v3 is the dispositive version.

## Design (perturb-and-eval, paired, bounded accuracy-drops)
Conditions per seed: NoCH (anchor = w_A) and CH (anchor = w_postCH @ chosen_lr). V = top-128 task-A Fisher @ w_A.
Per condition c: run phase-2(250) from anchor → realized displacement Δ; per-carrier project to V → **Δ_V,c** (unit of the
real interference). Then, for each magnitude multiplier **m ∈ {1, 2, 3}**:
- **dphase2(c,m)** = acc_A(anchor) − acc_A(anchor + m·Δ_V)                    [drop along the realized phase-2 dir]
- **drand(c,m)**   = mean_{K_RAND=5} acc_A(anchor) − acc_A(anchor + R_m)       [R_m random in V, per-carrier ‖R_n‖ = m·‖Δ_V,n‖]
- **aniso(c,m)**   = drand(c,m) − dphase2(c,m)                                 [>0 ⇒ phase-2 dir LESS damaging than random]
All bounded in [−1,1]. (m=1 is the physically-realized magnitude = the honest "what actually happens"; m=2,3 probe whether
the phase-2 direction stays special as the perturbation grows and the random baseline becomes sensitive.)

## Stage-1 calibration (fixed: margin + more seeds)
lr_child grid [1e-5, 1.5e-5, 2e-5, 3e-5]; **K_CAL = 20** seeds (was 12 — tighter estimate). Threshold thr = w_A_acc − 0.05.
**chosen_lr = max{ lr : r_cal[lr] ≥ thr + 0.015 }** (0.015 margin absorbs the cal-vs-full-N gap). If none clears the
margin, fall back to max{ lr : r_cal[lr] ≥ thr } and set flag `tight_dose=True`; if still none, min(grid) + `no_nondestructive_point=True`.

## Metrics & hypotheses (paired, per cell, per m)
- **Targeting:** Δaniso(m) = mean_seed( aniso(CH,m) − aniso(NoCH,m) ). >0 ⇒ childhood makes the phase-2 dir relatively safer
  (targeted). PRIMARY = Δaniso(1).
- **Global flatness:** g(m) = mean_seed( drand(NoCH,m) − drand(CH,m) ). >0 ⇒ generic basin widening (literature).
- **Baseline anisotropy:** aniso(NoCH,m) (is phase-2 already low-resistance pre-childhood?).

## Decision rule (pre-committed), per cell, SE from paired per-seed dist (N=100 ⇒ SE≈σ/10)
- **TARGETED (dispositive, novel):** Δaniso(1) > 0 (≥3σ) AND `childhood_nondestructive`=True AND sensitivity OK (below)
  AND sign(Δaniso(m))>0 for m=1,2,3 (magnitude-robust).
- **GLOBAL-ONLY (matches literature):** g(m) > 0 (≥2σ) at some m AND Δaniso(m) within ±2σ of 0 for all m.
- **BOTH:** g(m)>0 (≥2σ) AND Δaniso(m)>0 (≥2σ).
- **ENTANGLED / dose-dependent:** Δaniso(1)≈0 at the clean dose but was strong at v2's stronger dose ⇒ targeting requires a
  (mildly destructive) larger childhood — report honestly, do not force.
- **ARCHITECTURE-DEPENDENT TARGETING (headline):** TARGETED in OPT AND Δaniso(m)≈0 in Pythia (all m).

## Validity gates (VOID / flag)
- w_A reproduce-anchor: OPT 0.823 ± 0.02, Pythia 0.753 ± 0.02. VOID if off.
- **childhood_nondestructive** (full-N post-childhood task-A ≥ w_A−0.05) — now expected True via the margin; if it still
  fails, VOID-soft.
- **Sensitivity (replaces v2's blanket insensitive flag):** require mean drand(NoCH, m=3) > 0.01 — at the LARGEST magnitude
  a random in-V move MUST dent task-A, proving the random baseline is a live reference, not a dead floor. If drand(·,3) ≤
  0.01, set `insensitive=True` and VOID-soft (perturbations genuinely cannot read curvature even at 3×).
- Datasets load — HARD-STOP (VOID) on failure.

## Regime (locked)
Stage-1 K_CAL=20, margin 0.015. Stage-2: steps_child 400 @ chosen_lr; phase-2 250 @ lr 5e-5; V = top-128 task-A Fisher @
w_A; standard carriers; fp32; **K_RAND=5**; **MULT={1,2,3}**; **N=100** paired; cells **opt350m** (primary) + **pythia410m**
(cross-arch — the load-bearing contrast; run both).

## Ops (binding)
GPU_PREF excludes a10; fire-when-capacity watcher available if dry. Per-step `[hb]`; `[cal]`; `[seed]` DONE with running
Δaniso(1), g(1), drand(NoCH,3) (sensitivity), aniso_NoCH/CH at m=1. Smart killswitch armed on the printed id. Read OFF the
machine — never fabricate. Single cell per instance; launcher self-terminates on land.

## Cost
Per cell: calibration (20×4 childhoods, ~30 min) + per-seed [2 phase-2 runs + 3 magnitudes × (1 phase2-eval + 5 rand-evals)]
≈ ~3.5h on a100/gh200, ~$18/cell. Two cells ≈ ~$36. (Evals are cheap vs the phase-2 runs; the sweep adds little wall-time.)

## Relationship to the arc
Probe-5 (debris) → 6 (not pinning) → 7/7b (reorganization, ~100%) → **8 (is reorganization GLOBAL flatness [lit] or
TARGETED + architecture-dependent [ours]).** v2 said TARGETED (+14σ) but double-flagged; v3 is the clean dispositive read
that decides whether we publish a new brick (anticipatory directional protection, arch-dependent) or a sharpening of known
flat-minima work.
