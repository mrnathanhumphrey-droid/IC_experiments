# PRE-REG — Low-rank adjudication: single-task forgetting (low-rank, per lit) vs. the interference MAP (high-rank, per repo)

- **Date:** 2026-07-27
- **Status:** LOCKED — pre-registered. Not fired.
- **LOCK:** `b5c3929a` (sha256[:8] of draft body at lock, 2026-07-27T00:06; lock order 4 of 4)
- **Arc:** interference_law (extends `out/lowrank.json`)
- **Priority:** #4 of the 2026-07-27 subspace/eigenspace audit (★★★ — near-$0, runnable tonight on cached artifacts)

## One-line hypothesis
Reconcile the direct conflict between fresh literature and the repo's own data:
- **2606.18024 "Catastrophic Forgetting is Low-Rank"** — old-task loss is controlled by a small set of low-rank directions in the kernel eigenspace.
- **Repo `out/lowrank.json`** — the pairwise interference **interaction** is **NOT low-rank completable** (eff-rank 9.13 cifar / 8.45 dtd; flat top-6 spectrum 0.21/0.14/0.13/0.10/0.09/0.06; any rank-k>0 term makes prediction *worse*; verdict "high-rank/idiosyncratic").

**Proposed synthesis:** each *single task's* forgetting operator is **low-rank** (few directions), but the **MAP** `(source_j, target_i) → forgetting` is **high-rank** because each pair's low-rank subspace is **misaligned / dispersed** across the task bank. Testable: per-task forgetting-operator rank `r_i` is small (~1–3); assembled interaction eff-rank is high (~9, measured); and the **dispersion** of per-pair top forgetting directions explains the gap.

## Falsifier (pre-registered kill conditions)
- **KILL (no dichotomy):** if the **per-task** forgetting operator rank is *also* high (eff-rank ≥6, comparable to the interaction eff-rank) → there is no low-vs-high structure to reconcile; the repo simply disagrees with 2606.18024 flatly (report as honest disagreement, no synthesis).
- **KILL (mechanism wrong):** if per-pair top forgetting directions are **aligned** (low dispersion) yet the interaction is still high-rank → the "misaligned low-rank subspaces" mechanism is falsified; the high rank comes from something else.

## Priors to lose (pre-register against)
- 2606.18024 (forgetting low-rank) and 2510.09181 (low-rank Jacobian confines updates into a shared low-dim subspace) — both would naively predict the **interaction** is *also* low-rank. The repo's measured high interaction rank is the tension; the synthesis must show *where* the rank comes from (dispersion), not just assert disagreement.

## Must-beat baselines
1. The existing rank-completion curve in `lowrank.json` (best_k=0; any interaction rank hurts) — the per-task-rank measurement must be consistent with it.
2. A "shared subspace" null: if all per-pair top directions were identical, interaction rank would be ~1 — quantify the observed dispersion against this null.

## Method (mostly cached / near-$0)
1. **Per-task forgetting operator rank:** for each target i, assemble the change-in-readout (frozen) or change-in-representation (plastic) induced by training each source j; compute its rank / effective rank `r_i`. Is `r_i` low (~1–3)?
2. **Assembled interaction eff-rank:** already 9.13/8.45 in `lowrank.json` — reuse.
3. **Dispersion:** collect the top-1 forgetting direction for every (i,j) pair; measure their mutual principal-angle dispersion (mean pairwise angle / effective number of distinct directions). Correlate dispersion with the interaction eff-rank across bootstraps.

## Confound controls (pre-registered)
- **Regime:** frozen readout = 1-D discriminative overwrite (memory: forgetting = 1-D readout overwrite), so per-task rank may be *trivially* 1 in frozen → run the fair test in the **plastic** regime (LoRA reshapes backbone). Reuse the **P2 plastic pythia-1.4b cache** (clinc150+banking77, already on disk) so this stays near-$0.
- Report frozen and plastic separately; the synthesis claim is strongest if per-task rank is low in *both* while interaction rank stays high.
- Haar null band for the dispersion significance (`float_module` QX1 technique).

## Stats
Bootstrap over tasks; report eff-rank **distributions** with CI (not point estimates); dispersion-vs-interaction-rank correlation with perm-null.

## Cost / lane
Near-$0 frozen (all cached behind `lowrank.json`); reuse P2 plastic cache for the fair test → maybe $1 if a small recompute is needed. **Cheapest of the four; runnable immediately on cached artifacts.**

## Standing floor (publishable regardless)
- **Synthesis holds:** "forgetting is low-rank per task but the interference map is high-rank because per-pair subspaces are dispersed" — a clean structural result engaging a fresh (2606) paper, and it *explains* the repo's own high-rank finding rather than leaving it as an unexplained null.
- **Synthesis fails:** honest disagreement with 2606.18024 (interference genuinely high-rank at every level), still worth reporting given the paper is brand new.
