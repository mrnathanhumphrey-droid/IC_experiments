# PRE-REG — Probe-8: targeted vs global curvature alignment (the "further from the literature" probe). 2026-06-26.

Status: LOCKED on write (`PREREG_PROBE8_CURVATURE_ALIGNMENT_LOCK.json`). No GPU until explicit go.
Post-floor regime: N≥100, paired within-run differentials, fp32, per-step heartbeat + smart killswitch, never a10.

## Why this probe (white space vs the literature)
Novelty check (3 adversarial lit agents, 2026-06-26) verdict = PARTIALLY-KNOWN. The mechanism the closest priors assert —
Mirzadeh NeurIPS 2020 (forgetting ≈ ½ΔwᵀHΔw = movement × curvature), Mehta JMLR 2023 (pretraining → flatter basins),
RL's Razor 2025 (displacement is a weak forgetting predictor) — is **GLOBAL, undirected flatness**: a protected model sits
in a wider bowl, robust to ANY movement (a single sharpness scalar). NONE of them measure DIRECTION, and none test
architecture-dependence of the mechanism. Our DSI work hints the childhood *routes* interference (the "traffic guard").
So the hypothesis that goes furthest from them: protection is **TARGETED** — the childhood flattens task-A curvature
*specifically along the directions the future task will push*, pre-arranging task-A to be numb to exactly the coming
interference — not generic flatness. Plus a cross-arch contrast none of them do.

## Core question
Is the childhood's protection GLOBAL flattening (literature) or TARGETED alignment to the phase-2 interference direction
(novel), and is any targeting ARCHITECTURE-DEPENDENT (load-bearing OPT vs inert Pythia)?

## Design (perturb-and-eval, paired, no optimizer dynamics in the measurement)
Childhood dose = the NON-destructive lr from the locked calibration (same Stage-1 grid/rule as Probe-7b: lr_child ∈
[1e-5,1.5e-5,2e-5,3e-5]×12 seeds, chosen_lr = max lr with post-childhood task-A ≥ w_A−0.05). V = top-128 task-A Fisher at
w_A (the Probe-5 damage subspace). L_A = task-A (SST-2) cross-entropy loss; also record accuracy.

Per seed, two conditions sharing the seed:
- **NoCH:** anchor = w_A.
- **CH:**   anchor = w_postCH (childhood @ chosen_lr).
For each condition c:
1. Run phase-2 (depth, 250) from anchor_c → realized displacement Δ_c; project per-carrier into V → **Δ_V,c** (the
   direction & size phase-2 actually moves task-A's coords).
2. **rise_phase2,c** = L_A(anchor_c + Δ_V,c) − L_A(anchor_c)   [move by the realized phase-2 vector, in-V only]
3. **rise_rand,c**  = mean over K_RAND=3 random in-V directions R (per-carrier ‖R_n‖ = ‖Δ_V,c,n‖, magnitude-matched) of
   L_A(anchor_c + R) − L_A(anchor_c)
4. **R_c = rise_phase2,c / max(rise_rand,c, ε)**   (directional sensitivity ratio; <1 ⇒ phase-2 dir is safer than random)

## Metrics & hypotheses
- **Targeting (novel):** Δla = mean_seed( log R_NoCH − log R_CH ). >0 ⇒ childhood softens the phase-2 direction MORE than
  random (targeted alignment beyond generic flatness).
- **Global flatness (literature):** g = mean_seed( rise_rand,NoCH − rise_rand,CH ). >0 ⇒ childhood softens even a RANDOM
  direction (generic basin widening).
- **Baseline anisotropy:** R_NoCH (is phase-2 already a low-resistance direction without any childhood? R_NoCH<1 ⇒ phase-2
  naturally flows where task-A is flat).

## Decision rule (pre-committed), per cell, SE from paired per-seed dist (N=100 ⇒ SE≈σ/10)
- **TARGETED (the further-from-lit result):** Δla > 0 (≥2σ). Childhood confers DIRECTION-SPECIFIC robustness.
- **GLOBAL-ONLY (matches literature):** g > 0 (≥2σ) AND Δla within ±2σ of 0. Generic flattening, no targeting → we are a
  clean intervention-replication of Mirzadeh/Mehta, not beyond them. Honest if so.
- **BOTH:** g > 0 (≥2σ) AND Δla > 0 (≥2σ). Childhood widens the basin globally AND extra-softens the interference dir.
- **NULL:** neither ≥2σ → childhood's protection is not visible as local curvature change at the anchor (would itself be a
  surprise, given Probe-7b).
- **ARCHITECTURE-DEPENDENT TARGETING (furthest from lit):** Δla>0 in OPT AND Δla≈0 in Pythia. The headline if it lands.

## Validity gates (VOID if violated)
- w_A reproduce-anchor: OPT 0.823 ± 0.02 (Pythia 0.753 ± 0.02). VOID if off.
- childhood_nondestructive (post-childhood task-A ≥ w_A−0.05 at chosen_lr); if calibration finds no non-destructive dose,
  set flag and treat VOID-soft (same as Probe-7b).
- Perturbations must actually perturb: rise_rand > 0 by a sane margin (else ε/magnitude mis-set) → flag.
- Datasets load (SST-2; subj phase-2; childhood streams) — HARD-STOP (VOID) on load failure.

## Regime (locked — identical substrate to Probe-5/6/7)
steps_child 400 @ chosen_lr; phase-2 depth 250 @ lr 5e-5; held-out subj; V = top-128 task-A Fisher at w_A; standard
carriers; fp32; K_RAND=3; ε=1e-6; **N=100 seeds** paired; cells **opt350m** (primary), **pythia410m** (cross-arch, the
load-bearing point of the probe — run both).

## Ops (binding)
GPU_PREF excludes a10; if only a10 free, WAIT (or use the fire-when-capacity watcher). Per-step `[hb]`; Stage-1 prints
`[cal]`; `[seed]` DONE with running Δla, g, R_NoCH/R_CH. Smart killswitch armed on the printed id. Read OFF the machine —
never fabricate. Single cell per instance; launcher self-terminates on land.

## Cost
≈ Probe-7b (calibration + per-seed phase-2 + a handful of perturbation evals): ~3h/cell on a100/gh200, ~$15/cell; two
cells (OPT + Pythia) for the cross-arch contrast.

## Relationship to the arc
Probe-5 (deposit=debris) → Probe-6 (protection ≠ pinning) → Probe-7/7b (protection = reorganization, ~95%, zero
suppression at non-destructive dose) → **Probe-8 (is that reorganization GLOBAL flatness [lit] or TARGETED+arch-dependent
alignment [ours]).** This is the probe that decides whether we have a sharpening of known work or a genuinely new brick.
