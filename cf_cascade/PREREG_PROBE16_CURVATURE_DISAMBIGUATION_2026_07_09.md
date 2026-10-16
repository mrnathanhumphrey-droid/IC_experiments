# PRE-REG — Probe-16 (C1 disambiguation): does BLOOM's erosion reduce to the KNOWN flatness story (C3), or is it genuinely directional/conjunction (C1, novel)? 2026-07-09.

SHA256-lock before GPU. The 5-agent lit-check found C1 (ALiBi×multilingual conjunction flips the warmup→retention sign) is the
one NOVEL claim, but FRAGILE: reviewers will suspect "ALiBi×multilingual" is a proxy for the KNOWN flatness mechanism (C3 =
Mirzadeh 2020: forget ≈ ½·λ_max·‖Δθ‖²; flatter task minima forget less). This probe tests that reduction head-on by measuring the
task-A curvature SPECTRUM, so we can state whether BLOOM's erosion is or is not the scalar-flatness story.

## The reduction to rule out (two forms)
- **(a) Intrinsic-sharpness:** BLOOM's task-A minimum (at w_A) is intrinsically SHARPER than the protectors' → known "sharp
  minima forget more" → C1 not novel.
- **(b) Childhood-sharpens:** BLOOM's childhood SHARPENS the task-A minimum (raises λ_max) while protectors' childhood FLATTENS it
  → the sign of protection is a scalar-curvature change → C1 reduces to C3.
If EITHER holds and tracks the protect/erode outcome, C1 collapses into C3. If NEITHER does, C1 survives as genuinely directional
(consistent with Probe-13's N_random ≈ 0: the protect/erode sign lives in the interference direction, not in random/global curvature).

## DV — task-A empirical-Fisher spectrum (same F_A as Probe-13's V), scale-robust aggregates
Per carrier n, eigen-spectrum of F_A[n] (`eigvalsh`, clamp≥0). Aggregate over carriers:
- **anisotropy A** = mean_n(λ_max_n / trace_n) — curvature concentrated in the top direction. SHARP → high A. (unit-checked: sharp→1.0, flat→~1/cin.)
- **eff_rank ER** = mean_n(trace_n² / Σ_k ev²) — how many directions carry curvature. FLAT/isotropic → high ER; SHARP → low ER. (unit-checked: sharp→1, flat→cin.)
- **lam_max_raw** = mean_n(λ_max_n) — scale-DEPENDENT; reported but NOT cross-arch-compared (only within-cell Δ).
A and ER are scale-robust → cross-architecture comparable. Within-cell childhood effect = fractional change dA_frac, dER_frac.

## Design
Per cell: (1) BASELINE spectrum at w_A (deterministic). (2) POST-CHILDHOOD spectrum at DOSES = **1e-6, 7e-6** (Probe-13 grid
endpoints: gentle + strong) × **N=8** seeds → mean fractional Δ(A), Δ(ER) + retA. Anchor gate reused (ANCHOR env for XGLM 0.895 /
Falcon 0.88; VOID if off > 0.02). Incremental JSON dump per seed. Harness `cloud/probe16_curvature_lambda.py` (probe13 spine;
`spectrum()` unit-checked). NO interference-direction realization (that's Probe-13) — this is the SCALAR/SPECTRAL axis only.

Cells (all w_A ckpts on disk): **BLOOM-560M (eroder), XGLM-564M + Falcon-rw-1b (protectors)** = the decisive core; **OPT-350M +
Pythia-410M (protectors)** = extend the baseline-sharpness comparison to 5 cells.

## Pre-registered read (locked before result)
- **C3-REDUCTION SUPPORTED if:** (a) BLOOM baseline anisotropy is a clear high outlier (or eff_rank a low outlier) vs the 4
  protectors; OR (b) the childhood-induced dA_frac tracks the outcome — protectors FLATTEN (dA_frac < 0 / dER_frac > 0), BLOOM
  SHARPENS (dA_frac > 0 / dER_frac < 0), i.e. sign(dA_frac) correlates with sign(−N_phase2) across cells. Either ⇒ C1 collapses
  into the known flatness story; report honestly and DOWNGRADE C1.
- **C1 SURVIVES if:** BLOOM baseline sharpness ≈ the protectors (not an outlier) AND dA_frac does NOT track the protect/erode
  sign (BLOOM does not sharpen more than protectors flatten). ⇒ scalar curvature does not explain the sign ⇒ the effect is
  directional/conjunction ⇒ C1 genuinely novel. (Predicted, given Probe-13 N_random ≈ 0 for all cells.)
- Report the cross-cell table of {baseline A, ER} and {dA_frac, dER_frac at each dose}; state which hypothesis the data support.

## VOID / guards
anchor off > 0.02 → hard VOID (baseline still dumped). dataset load fail → VOID. childhood-destructive doses are FINE here (retA
recorded; the spectrum is still a valid measurement of the perturbed minimum) — we are not gating on non-destructiveness.

## Ops
Reuses Probe-13 w_A ckpts + Fisher machinery; a100-first (us-east-1 $1.99), a10 excluded; launcher scp-hang fix live; probe16 in
PROBES + DOSES/NSEEDS/ANCHOR pass-through. Cheap: 2 doses × 8 seeds × (childhood 400 + one Fisher) ≈ far shorter than Probe-13.
1.3B cells (Falcon) a100 mem-safe. Per cell ~$2-4.

## Status
NOT fired (harness built + py_compile-clean + spectrum math unit-checked: sharp→A 1.0/ER 1.0, flat→A ~1/cin/ER cin). Next: LOCK →
fire core 3 (BLOOM/XGLM/Falcon) ± OPT/Pythia → read baseline-sharpness + childhood-Δ against the protect/erode sign.
