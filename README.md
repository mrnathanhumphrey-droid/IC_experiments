# IC_experiments: interference and forgetting in weight space

An active, pre-registered research log on **how capabilities interfere inside a network**: when training one thing damages another, what geometry predicts it, and what protects against it. Language models (OPT, Pythia, BLOOM, Jais, XGLM, Falcon, Gemma, Qwen) and vision models (ResNet on CIFAR and DTD).

Every probe is `PREREG_*.md` → run → `VERDICT_*.md`. **The VERDICT files are the current results**; older prose in this log may be superseded by later verdicts.

## Findings so far

| Finding | Number | Verdict |
|---|---|---|
| Early ("childhood") training protects a capability by **reorganizing** it, not by suppressing interference | Δrobust +0.356 ± 0.027 (13σ, N = 100) | [`cf_cascade/VERDICT_PROBE7_REORGANIZATION_2026_06_26.md`](cf_cascade/VERDICT_PROBE7_REORGANIZATION_2026_06_26.md) |
| Concepts are carried by a **broad subspace**, not one direction | 9–17 causal directions per concept; none carries > 20% | [`cf_cascade/VERDICT_SUBSPACE_DIMENSIONALITY_PR_P0_2026_07_27.md`](cf_cascade/VERDICT_SUBSPACE_DIMENSIONALITY_PR_P0_2026_07_27.md) |
| Forgetting is **low-rank per task but high-rank across tasks**: each task's damage lives in ~2 dimensions, the pairwise map needs ~9.5 | eff. rank 2.00 vs 9.57 (CIFAR; replicated on DTD) | [`interference_law/VERDICT_LOWRANK_ADJUDICATION_CIFAR_2026_07_27.md`](interference_law/VERDICT_LOWRANK_ADJUDICATION_CIFAR_2026_07_27.md) |
| With a frozen backbone, gradient cosine predicts forgetting; the law disappears once the backbone can move | ρ = −0.74 frozen; gone by LoRA rank 2 | [`interference_law/VERDICT_PHASE_RANK_2026_07_14.md`](interference_law/VERDICT_PHASE_RANK_2026_07_14.md) |
| Forward transfer is reliable but is **not** governed by the forgetting geometry | r_SB 0.95–0.98; uncorrelated with cosine | [`interference_law/VERDICT_P0_DISSOCIATION_2026_07_26.md`](interference_law/VERDICT_P0_DISSOCIATION_2026_07_26.md) |

Next, locked but not yet run: cyclic forgetting as holonomy on the bundle of task subspaces ([`interference_law/PREREG_HODGE_HOLONOMY_2026_07_27.md`](interference_law/PREREG_HODGE_HOLONOMY_2026_07_27.md)).

## Layout

| folder | question |
|---|---|
| `cf_cascade/` | does early training protect a capability from later interference, and how (LLMs) |
| `interference_law/` | what geometry predicts forgetting and transfer (vision + LLM) |
| `detector_cf/`, `audio_cf/`, `librarian_cf/` | the same laws on applied systems |
| `alibi_multi_cf/`, `tier_abstraction_v0_1/` | targeted side tests |

Datasets, checkpoints and caches are not tracked (see `.gitignore`). Runs used rented single-GPU instances; launchers are in each folder's `cloud/`.

MIT license.
