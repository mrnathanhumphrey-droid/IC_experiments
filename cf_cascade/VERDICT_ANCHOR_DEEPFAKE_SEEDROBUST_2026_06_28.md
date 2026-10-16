# VERDICT — Anchor B: Deepfake seed-robustness (suppression-only negative control). 2026-06-28.

Pre-reg `PREREG_ANCHOR_DEEPFAKE_SEEDROBUST_2026_06_28.md` (LOCK a227f6a5). Local CPU, $0. Driver
`cloud/deepfake_anchor_seedsweep.py` — COPY-NOT-EDIT: imports the physics-detector harness functions READ-ONLY (originals
untouched), adds SEED + the two Ω-placement cosines (transparent defs), writes only to `gateA/`. Corpus found on disk
(`D:/corpus/face_swap_corpus/{deepfakeface,facefusion_matched}`), env MULTIFAMILY=1 INCLUDE_DIFFUSION=1. Substrate =
FROZEN DetectorCNN backbone → 128-d → linear head only ⇒ **reorganization is structurally impossible** (head-only).

## Result (5 seeds, mean ± SE)
| quantity | value | read |
|---|---|---|
| cos_meanshift_swap_diff | **−0.992 ± 0.001** | antagonistic endpoint FIRMS (single-seed −0.95 was not a fluke; rock-stable) |
| base swap AUC | 0.867 | the memory to protect |
| naive swap AUC | 0.210 ± 0.015 | catastrophic forgetting on plain continue-train |
| **GPM swap AUC (real projection)** | **0.855 ± 0.007** | **suppression WORKS** — holds swap at baseline |
| **randctrl swap AUC (random projection)** | **0.211 ± 0.015** | **ZERO protection** — same as naive |
| gpm diffusion AUC | ~0.27 | boundary in representation (diffusion not learnable at the head) |

Pre-registered reads: **antagonistic_endpoint_firms=True, suppression_works=True, randctrl_zero_protection=True** (all 5
seeds; `gateA/anchor_deepfake_seedrobust.json`). (cos_w_swap_w_diff more seed-variable, −0.02…−0.30; the mean-shift cosine
is the dispositive antagonism measure and is nailed at −0.99.)

## What it banks (the foil)
In a substrate where **reorganization cannot happen** (frozen backbone), protection behaves EXACTLY as pinning/suppression
theory predicts: **AIMED suppression** (GPM projecting off the real swap-memory subspace) protects the old task, while
**generic suppression** (random subspace) gives zero protection — identical to no protection at all. This is the
**suppression-only negative control** and the structural foil to the cf_cascade arc's reorganization result: where OPT
(Probe-7/7b) protected by *reorganization* (same movement, less damage) and *aimed* it (Probe-8 v4), here — with
reorganization removed by construction — only aimed *suppression* is available, and it is the only thing that works.

## Axis placement + caveat
Places this substrate at the **antagonistic (anti-aligned) Ω extreme** (cos_ms −0.99) with "protection = suppression only."
**Caveat (carried from the original confound check):** the antagonism is DOMAIN-driven, not forgery-driven
(`antagonism_confound_result.json`: cos_domain −0.91, cos_forgery +0.67) — the swap↔diffusion anti-alignment is largely the
IMDB-WIKI domain shift of the diffusion corpus. So this anchor is "antagonistic extreme + suppression-only control," with
the antagonism's source (domain) named, not overclaimed as pure paradigm antagonism.

## Role in the Ω law
Third substrate: **antagonistic / no-reorganization extreme** (suppression-only). With Anchor A (Librarian, free-lane,
moderate Ω, protection holds) and Probe-9 (OPT/Pythia continuous Ω sweep, reorganization), this spans the Ω axis from
antagonistic→free-lane across THREE independent substrates — robust to the n=2-architecture criticism because the axis is
Ω, not architecture. Artifacts: `gateA/anchor_deepfake_seed{0..4}.json`, `gateA/anchor_deepfake_seedrobust.json`.
