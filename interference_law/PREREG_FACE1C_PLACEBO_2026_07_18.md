# PREREG — Face-1c P1 PLACEBO / NULL CONTROL (does curl reproduce without task-specific forgetting?)

**Date:** 2026-07-18 · Addresses external pushback (#1 projection-manufactures-curl, #2 low-SNR Hodge-hallucination). The projection-VARIATION control (`VERDICT_FACE1C` addendum) was necessary but not SUFFICIENT — this is the missing negative control.

## Design
Three conditions, ONE base / projection-grid / caps / seeds (apples-to-apples):
- **REAL** — LoRA-SFT on perturber class j's texts (task-specific interference).
- **RANDTEXT** — LoRA-SFT on a task-AGNOSTIC random text mix per column (controls "any real training causes curl").
- **NOISE** — isotropic Gaussian weight-noise, ‖Δθ‖ MATCHED per-column to REAL's LoRA delta (controls "any movement of this magnitude causes curl").
Build directed F per condition, Hodge-decompose, cross-seed curl-ρ + dominance-ρ, over projection grid {12,24}×{0,1}. Compare in-band median curl-ρ across conditions. 16 caps × 3 seeds. ~$2.3, ~70 min.

## Pre-committed decision
- **GENUINE:** real curl-ρ ≥ 0.30 AND (real − max(placebo curl-ρ)) ≥ 0.15. → cycles are task-specific interference, not pipeline/projection artifact.
- **ARTIFACT:** ANY placebo curl-ρ ≥ 0.30. → the cyclic structure reproduces without real forgetting → RETRACT the Face-1c cyclic claim.
- **AMBIGUOUS:** separation < 0.15 → needs SNR (P3) before believing.

## Priors (stated to lose)
The real result (curl-ρ 0.37) already survived projection variation, so I lean GENUINE (~65%). But the SNR concern is real (medFgt on the floor), so a magnitude-matched NOISE placebo showing curl-ρ ≥ 0.30 is a live, respected outcome — and if it fires, the honest move is retraction. Falsifier = placebo curl-ρ ≥ 0.30.

## Output
`out/face1c_placebo.json` → verdict appended to `VERDICT_FACE1C_LLM`. SMOKE self-check first.
