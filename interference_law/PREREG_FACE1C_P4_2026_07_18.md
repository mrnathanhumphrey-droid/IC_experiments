# PREREG — Face-1c P4: CROSS-SUBSTRATE REPLICATION (does the LLM cyclic finding survive a new architecture AND a new dataset?)

**Date:** 2026-07-18 · Addresses the last open critique (#3, the cf_cascade "n=1 eroder = BLOOM" lesson): every Face-1c result so far rests on **one** model (Qwen2.5-1.5B) and **one** dataset (banking77, a domain-cramped all-banking intent set). Projection (#1) and low-SNR Hodge (#2) are already killed by the projection-variation + placebo controls. This is the generalization test.

## The claim under test
On a plastic LLM (LoRA-SFT), the directed forgetting matrix carries a **reproducible cyclic (rock-paper-scissors) component** (cross-seed curl-ρ ≥ 0.30) that persists across projections. Is that a property of *interference at LLM scale*, or an idiosyncrasy of Qwen/banking77?

## Design — 3 cells, one instance, one teardown
Each cell = the pre-registered **projection-sweep persistence test** (`face1c_control` machinery): cache all fine-tunes once, sweep a projection grid {8,12,24,48}×{0,1,2}, cyclic only if curl-ρ ≥ 0.30 PERSISTS in ≥66% of in-band grid points.
- **`pythia1.4b:clinc150`** — BOTH new (GPT-NeoX arch ≠ Qwen; clinc150 = 150 intents across 10 domains, genuinely cross-domain ≠ all-banking). **The money cell.**
- **`pythia1.4b:banking77`** — isolates the ARCHITECTURE axis (new model, old data).
- **`qwen:clinc150`** — isolates the DATASET axis (old model, new data).

Arch-agnostic LoRA via `target_modules='all-linear'` (Pythia's NeoX module names differ from Qwen's). `LAYER = n_layers//2` auto per model. 16 caps × 3 seeds × 200 steps, same LoRA r=16/α=64/lr=5e-4 as Face-1c.

## Pre-committed decision
- **REPLICATES:** the **both-new** cell (`pythia1.4b:clinc150`) has in-band median curl-ρ ≥ 0.30 AND frac ≥ 0.66. → cyclic forgetting is not Qwen/banking77-specific; it generalizes across arch and data jointly.
- **REPLICATES-GENERAL:** all 3 cells persist. → each axis (arch, data) independently generalizes.
- **DOWNGRADE:** the both-new cell FAILS (median curl-ρ < 0.30 or frac < 0.66). → honest downgrade: the cyclic finding is substrate-specific (Qwen/banking77), and the isolate-axis cells tell us WHICH axis breaks it.

## Priors (stated to lose)
The placebo showed the curl is task-interference-driven, not pipeline; task interference is a general phenomenon, so I lean **REPLICATES (~70%)**. The live losing outcome: cycles are a quirk of Qwen's geometry OR of banking77's tight single-domain intent structure (10 domains in clinc150 could wash out cross-task interference into cleaner transitive dominance). If the both-new cell fails, that's the real, publishable-either-way answer. Falsifier = both-new curl-ρ < 0.30.

## Cheap-first discipline
SMOKE=1 first (1 tiny cell: pythia1.4b:clinc150, ncap=3, 2 seeds, 40 steps, 1 projection) to validate the clinc150 loader + Pythia (new arch) load + `all-linear` LoRA actually moving the backbone + Hodge end-to-end, BEFORE the real 3-cell spend. Real run only fires on explicit go.

## Output
`out/face1c_p4.json` (written incrementally after each cell — a mid-run crash still lands completed cells) → verdict appended to `VERDICT_FACE1C_LLM`. Follow-up if the both-new cell replicates: a single RANDTEXT placebo on that cell (P4b, ~$1.5) to confirm the null holds on the new substrate too.
