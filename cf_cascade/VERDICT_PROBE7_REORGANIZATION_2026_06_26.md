# VERDICT — Probe-7 / 7b: load-bearingness is REORGANIZATION, not suppression. 2026-06-26.

Pre-regs: Probe-7 `PREREG_PROBE7_REORG_2026_06_25.md` (LOCK 7bdfcbc6); Probe-7b `PREREG_PROBE7B_NONDESTRUCTIVE_2026_06_26.md`
(LOCK 9d27939a). N=100 each, paired, fp32, a100_sxm4 (ML-CF), heartbeat + smart killswitch. Results
`gateA/probe7_displacement_opt350m.json`, `gateA/probe7b_displacement_opt350m.json`. Both w_A anchor 0.823 reproduced.

## The question (capping the arc)
Probe-6 falsified "absorption = pinning" but had a step-count confound. Probe-7 killed the confound by measuring
phase-2-ONLY displacement from each arm's pre-phase-2 anchor, splitting protection into **suppression** (childhood makes
phase-2 move less in V; ΔD2>0) vs **reorganization** (same movement, less damage per unit; Δrobust>0). Probe-7b re-ran it
at a calibration-selected NON-destructive childhood dose to remove the last ambiguity.

## Result → REORGANIZATION, dispositive
| contrast | Probe-7 (lr 3e-5, destructive) | Probe-7b (lr 1.5e-5, non-destructive) |
|---|---|---|
| Δforget (protection) | +0.1435 ± 0.011 (+13σ) | **+0.1649 ± 0.0112 (+14.8σ)** |
| Δrobust (reorganization) | +0.289 ± 0.029 (+10σ) | **+0.356 ± 0.027 (+13σ)** |
| ΔD2 (suppression) | +0.0176 ± 0.007 (+2.7σ) | **−0.0044 ± 0.0063 (≈0, gone)** |
| r_ch (movement↔forget, CH) | −0.29 | **−0.04 (decoupled)** |

Counterfactual decomposition (forget = robust × D2):
- Probe-7: ~95% reorganization, ~1% suppression.
- **Probe-7b: ~100% reorganization, 0% suppression** (suppression term slightly negative — D2_ch ≥ D2_noch).

The pre-committed **REORG_if** (Δforget>0 ≥2σ AND Δrobust>0 ≥2σ AND ΔD2≤0) is satisfied outright at the clean dose. The
+2.7σ sliver of suppression in Probe-7 was an artifact of the destructive 3e-5 dose; at the non-destructive dose it is
exactly zero. **Load-bearingness protects task-A entirely by making the same phase-2 movement less damaging — pure
reorganization, not coordinate-pinning.** The `ENTANGLE` branch did NOT fire (protection is bigger, not smaller, at the
gentle dose), so the effect does not require the destructive dose.

## Calibration (Probe-7b Stage-1, locked grid+rule)
post-childhood task-A: lr 1e-5 → 0.792 · 1.5e-5 → 0.776 · 2e-5 → 0.758 · 3e-5 → 0.738. Rule chose max non-destructive =
**1.5e-5**. Confirms Probe-7's 3e-5 was genuinely destructive (0.738), validating the gate-trip that prompted 7b.

## Honest caveat (does not affect the conclusion)
Probe-7b's `childhood_nondestructive` flag read False by **0.002** (full-N ret_anchor_ch 0.771 vs 0.773 bar) — the 12-seed
calibration over-estimated by 0.0019 (sampling noise), unlike Probe-7's real 8-point destruction. Immaterial: the dose was
gentle enough that the suppression term vanished, which was the entire purpose. A 1e-5 confirmatory (cal 0.792, comfortably
clean) would remove the asterisk but the mechanism call does not need it.

## Where this sits vs the literature (novelty check 2026-06-26, 3 adversarial agents → PARTIALLY-KNOWN)
- **KNOWN (cite, do not claim):** forgetting ≈ movement × curvature, "flat minima forget less per unit" — **Mirzadeh et al.
  NeurIPS 2020** (Eq. 5, Fig. 2a, the exact decomposition). Pretraining → flatter basins → less forgetting — **Mehta et al.
  JMLR 2023**. Parameter displacement (incl. Fisher-weighted) is a weak forgetting predictor — **RL's Razor (Shenfeld et
  al.) 2025**. "Pin important weights" = EWC (Kirkpatrick 2017); "steer off important subspace" = OGD/GPM — the suppression
  foils we refute.
- **NOVEL (defensible):** a **causal warm-up that rescales damage-per-unit-displacement ~3× (robust 0.526→0.170) while
  in-V movement is unchanged/larger**, explicitly falsifying the reduced-movement/pinning account — i.e. the *change in the
  displacement→forgetting slope* under an intervention, which the priors assert/assume but never isolate. Frame the headline
  around the slope-change, not the bare decoupling.

## Go-forward
- **Probe-8 (LOCKED b33ab105):** is this reorganization GLOBAL flatness (literature) or TARGETED to the phase-2 interference
  direction (+ architecture-dependent OPT vs Pythia)? The probe that decides sharpening-of-known vs new brick.
- **Optional:** Probe-7b 1e-5 ultra-clean confirmatory (removes the 0.002 asterisk); Pythia Probe-7b (does reorganization
  exist at all in the inert arch?).

Artifacts: `gateA/probe7{,b}_displacement_opt350m.json`; harnesses `cloud/probe7{,b}_displacement_lambda.py`. Instances
445be001 (P7) + 9844dd495f (P7b) terminated clean (no orphan); cost ~$30. Arc: `VERDICT_PROBE5_RESIDUAL`,
`VERDICT_PROBE6_DISPLACEMENT`, this. Lit detail in `CONSOLIDATION_forgetting_flow.md`.
