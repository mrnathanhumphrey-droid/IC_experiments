# PRE-REG — Probe-8 v4: specificity of the directional protection (a null that bites). 2026-06-28.

Status: LOCKED on write (`PREREG_PROBE8_v4_SPECIFICITY_LOCK.json`). No GPU until explicit go.
**Builds on** v3 (`PREREG_PROBE8_CURVATURE_ALIGNMENT_v3_2026_06_27.md`, LOCK 65ae39d4). v3's FINDING 1 (NOT global flatness,
g≤0) STANDS and is not re-litigated; v4 sharpens FINDING 2.

## Why v4
v3 established the protection is DIRECTIONAL, not global (childhood does not soften random directions; g≤0, 10–16σ, both
archs). But its null was weak: random-in-the-full-128-dim-V doesn't dent task-A even at 3× (drand≈0), so the "targeting"
number partly re-expressed Probe-7 ("childhood reduces phase-2 forgetting") rather than proving the protection is *aimed* at
the interference. v4 replaces the dead null with directions that actually bite, and asks the right question:

**Is the childhood's directional protection SPECIFIC to the actual phase-2 interference direction, or does it robustify
task-A against real-task damaging directions GENERICALLY?** (Note: the childhood precedes phase-2 and never sees it, so true
per-trajectory "anticipation" would be surprising; the honest hypotheses are specific-to-interference vs generic-directional.)

## Design (perturb-and-eval, paired, bounded accuracy-drops, single realized magnitude)
Conditions per seed: NoCH (anchor=w_A) and CH (anchor=w_postCH @ chosen_lr, dose via the v3 margin calibration). V = top-128
task-A Fisher @ w_A. From each anchor run phase-2(250) → realized in-V displacement **Δ_V**; set per-carrier magnitude
mag_n = ‖Δ_V,n‖. Build three in-V perturbation directions, each magnitude-matched to mag_n per carrier:
1. **phase2** — Δ_V itself (the ACTUAL interference direction).
2. **alttask** — gradient of a DIFFERENT held-out downstream task (a task NOT used in phase-2 and NOT in the childhood
   stream — locked: TREC), taken at the anchor, projected to V, unit per carrier. A genuine damaging direction the childhood
   was not pointed at. (Guaranteed to bite — it's a real task gradient.)
3. **randtop16** — random direction within the TOP-16 task-A Fisher eigendirections (high-curvature baseline; should bite
   more than random-in-full-V did).
For each direction d and condition c: **dprop(c,d) = acc_A(anchor_c) − acc_A(anchor_c + mag·d)** (random arms averaged over
K_RAND=5 draws). Bounded in [−1,1].

## Metrics & hypotheses (paired, per cell)
Neutralization (childhood's reduction of a direction's damage): **N(d) = mean_seed( dprop(NoCH,d) − dprop(CH,d) )**.
- **Specificity:** S = N(phase2) − N(alttask).
- Curvature baseline: N(randtop16).

- **SPECIFIC-to-interference:** S > 0 (≥2σ) AND N(phase2) > N(randtop16) (≥2σ). Childhood neutralizes the direction it will
  actually be hit by MORE than an equally-damaging different-task direction → the protection is aimed, not generic.
- **GENERIC-DIRECTIONAL:** S within ±2σ of 0 AND N(phase2) ≈ N(alttask) > N(randtop16) (≥2σ). Childhood robustifies task-A
  against real-task damaging directions broadly (not phase-2-specific), beyond a random high-curvature direction. Still NOT
  global (carried from v3). The most likely outcome and an honest, publishable mechanism.
- **WEAK/NULL:** N(phase2) ≈ N(randtop16) → no directional structure beyond generic; would walk back v3's FINDING 2.

## Validity gates (VOID / flag)
- w_A anchor: OPT 0.823 ± 0.02, Pythia 0.753 ± 0.02.
- childhood_nondestructive (full-N post-childhood task-A ≥ w_A−0.05) via margin calibration; VOID-soft if not.
- **Bite gate (replaces v3 sensitivity):** mean dprop(NoCH, phase2) > 0.03 AND mean dprop(NoCH, alttask) > 0.03 — both real
  directions must actually damage task-A (they will). If alttask does not bite, the null is still bad → flag `null_dead=True`,
  VOID-soft.
- alttask (TREC) and datasets load — HARD-STOP (VOID) on failure.

## Regime (locked)
Stage-1 calibration = v3 (lr grid [1e-5,1.5e-5,2e-5,3e-5], K_CAL=20, chosen_lr = max lr with r_cal ≥ (w_A−0.05)+0.015;
fallbacks tight_dose / no_nondestructive_point). Stage-2: steps_child 400 @ chosen_lr; phase-2 250 @ lr 5e-5; V = top-128
task-A Fisher @ w_A; top-16 sub-block for randtop16; standard carriers; fp32; K_RAND=5; **single realized magnitude (no MULT
sweep)**; **N=100** paired; cells **opt350m** + **pythia410m**. alttask = TREC (SetFit/TREC-QC parquet, as in Probe-5).

## Ops (binding)
GPU_PREF excludes a10; watcher if dry. Per-step `[hb]`; `[cal]`; `[seed]` DONE with running N(phase2), N(alttask),
N(randtop16), S, and the NoCH bite levels. Smart killswitch armed. Read OFF the machine — never fabricate. Single cell per
instance; launcher self-terminates.

## Cost
Lighter than v3 (no 3× sweep): calibration (~30 min) + per-seed [2 phase-2 + 1 alttask grad + ~7 evals] ≈ ~3h/cell on
a100/gh200, ~$15/cell; two cells ≈ ~$30.

## What v4 closes
v3: protection is directional, not global (clean). v4: is that directionality AIMED at the interference (specific) or a
broad robustness to real-task directions (generic) — with a null that bites, so the answer is dispositive and not a Probe-7
re-expression. Either outcome (specific or generic-directional) is a clean mechanism past the flat-minima literature.
