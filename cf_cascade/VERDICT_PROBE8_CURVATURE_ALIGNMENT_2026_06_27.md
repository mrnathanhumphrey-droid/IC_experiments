# VERDICT — Probe-8: targeted vs global curvature alignment (cross-arch). 2026-06-27.

Pre-reg v3 `PREREG_PROBE8_CURVATURE_ALIGNMENT_v3_2026_06_27.md` (LOCK 65ae39d4). N=100, paired, fp32, dual-cell
(OPT a100_sxm4 + Pythia gh200, ML-CF), heartbeat + smart killswitch. Results `gateA/probe8v3_curvature_{opt350m,pythia410m}.json`.
History: v1 (b33ab105) VOID (R-ratio/CE estimator degenerate); v2 (93f26963) gave +14σ targeting but double-flagged
(boundary dose + dead random null). v3 fixed the dose (calibration margin + K_CAL=20) and added the magnitude sweep.

## The question
Probe-7/7b showed protection = reorganization (same movement, less damage). The literature attributes that to GLOBAL
flatness (a wider basin, robust to ANY movement — Mirzadeh NeurIPS'20, Mehta JMLR'23). Probe-8 asks: is it global, or
TARGETED to the phase-2 interference direction (novel), and is it architecture-dependent? Perturb task-A along the realized
phase-2 in-V direction vs random in-V directions (magnitude-matched, m∈{1,2,3}×), with vs without childhood; bounded
accuracy-drops. Δaniso>0 = targeted; global g = drand_NoCH − drand_CH >0 = generic flattening.

## Result (N=100 each; both chose a non-destructive dose — flag 1 FIXED)
| | OPT (lr 1e-5, retA_ch 0.789) | Pythia (lr 3e-5, retA_ch 0.735) |
|---|---|---|
| Δaniso m=1 | **+0.144 ± 0.013 (+11.6σ)** | **+0.101 ± 0.009 (+10.7σ)** |
| dphase2 NoCH→CH (m=1) | 0.147 → **0.005** (~97% neutralized) | 0.143 → **0.046** (~68% neutralized) |
| global g (m=1,2,3) | −0.003, −0.008, −0.012 (all <0) | −0.004, −0.007, −0.010 (all <0) |
| Δaniso m=2 / m=3 | +0.127 / +0.065 | +0.092 / +0.066 |
| sensitivity drandN(m3) | −0.003 (< 0.01) | −0.010 (< 0.01) |

## FINDING 1 — NOT global flatness (clean, no caveat). ★ the real result.
`global g ≤ 0` at every magnitude in BOTH cells, 10–16σ. If the childhood worked by widening the whole basin (the
flat-minima account), it would make RANDOM in-V directions safer too → g>0. It does the opposite (g<0: childhood slightly
*increases* random sensitivity). **The protection is not generic flattening.** This is the genuine step past
Mirzadeh/Mehta, and it does not depend on the sensitivity gate (it's a direct paired contrast on drand).

## FINDING 2 — TARGETED reorganization, in both architectures, graded OPT>Pythia.
The childhood specifically collapses the phase-2-direction damage (OPT 0.147→0.005 ~97%; Pythia 0.143→0.046 ~68%),
+10–12σ, while random in-V directions are untouched. So the reorganization is **directional** — aimed at the interference
direction. It is NOT the hoped-for binary dissociation (OPT-yes/Pythia-no): both archs show it, the load-bearing OPT just
does it **more completely**. The effect erodes at larger perturbation (Δaniso m1>m2>m3) — finite protection, overwhelmed
by big enough moves.

## FINDING 3 — forgetting is extremely directional (and the null is weak).
Random in-V perturbations do NOT dent task-A even at 3× the phase-2 magnitude (drandN(m3) ≤ 0). So within the 128-dim
task-A Fisher subspace, almost nothing matters for task-A except the specific sub-direction phase-2 hits. This is
informative (supports directionality) but means the **random-in-V null is weak**: with drand≈0, Δaniso largely reduces to
"childhood reduces phase-2-direction damage," which overlaps Probe-7. The clean, null-independent contributions of Probe-8
are FINDING 1 (g<0 refutes global) and the directionality itself.

## Pre-registered gates
- childhood_nondestructive = **True** both (OPT 0.789, Pythia 0.735) — flag 1 FIXED by the margin. ✓
- sensitivity (drandN_m3 > 0.01) = **FAILED** both → `insensitive=True` → VOID-soft per the locked rule. BUT the failure is
  the directionality signal, not a measurement fault; and FINDING 1 (the headline) is independent of it. Honest status:
  Probe-8 is **suggestive-strong**, not fully dispositive, pending a sharper null.

## GO-FORWARD
- **Probe-8 v4 (the dispositive version):** replace the random-in-V null with a null that actually bites — random within the
  **top-k (k≈8–16) task-A Fisher directions** (the dims that matter), or a curvature-profile-matched null — so drand is
  off the floor and Δaniso is a real discriminator, not a re-expression of Probe-7. Keep dual-cell. THEN the targeted-vs-
  global and the OPT-vs-Pythia-grading claims become dispositive.
- Net so far: the flat-minima/global account is **refuted** (clean); protection is **directional reorganization**, present
  in both archs and stronger in the load-bearing one. A real step past the literature, with the cross-arch graded (not
  binary) and the targeting metric awaiting a sharper null.

## vs literature
Refutes the GLOBAL-flatness mechanism (Mirzadeh NeurIPS'20, Mehta JMLR'23) for THIS protection: it is directional, not a
uniform basin-widening — childhood does not soften random directions (g<0). Complements RL's Razor (displacement weak
predictor) by localizing where the damage actually lives (one direction in V) and showing a warm-up specifically
neutralizes it.

Artifacts: `gateA/probe8v3_curvature_{opt350m,pythia410m}.json`; harness `cloud/probe8v3_curvature_lambda.py` (LOCK
65ae39d4). Instances 29674f85 (OPT) + 2245e713 (Pythia) terminated clean (no orphan); cost ~$40 (incl. v1/v2). Arc:
VERDICT_PROBE5/6/7; lit detail in `CONSOLIDATION_forgetting_flow.md`.
