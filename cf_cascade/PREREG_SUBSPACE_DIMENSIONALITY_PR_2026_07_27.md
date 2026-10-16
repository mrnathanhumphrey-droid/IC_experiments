# PRE-REG — Spectral dimensionality of a protected/steered subspace (participation ratio predicts steerable-axis-vs-broad-subspace)

- **Date:** 2026-07-27
- **Status:** LOCKED — pre-registered. Not fired.
- **LOCK:** `10d04137` (sha256[:8] of draft body at lock, 2026-07-27T00:06; lock order 1 of 4)
- **Arc:** cf_cascade (extends Probe-18) + new concept-steering substrate
- **Priority:** #3 of the 2026-07-27 subspace/eigenspace audit (upgraded ★★★→★★★★ by the concept-subspace lit)

## One-line hypothesis
A single pre-registered spectral scalar — the **participation ratio** `PR = (Σσ²)² / Σσ⁴` of the contrastive **difference-in-means matrix** `D ∈ ℝ^{N×d}` for a concept/capability — predicts the concept's **causal effective steering rank** `k*` (the number of top singular directions needed to recover ≥90% of the full ablation/steering effect), and secondarily predicts **robustness** (OOD steering brittleness, ablation/jailbreak resistance) and **unlearning reversibility**.

`PR ≈ 1` ⇒ single steerable axis; `PR ≫ 1` ⇒ broad subspace.

## Why now / the wedge
The refusal-direction literature is actively split and nobody has a validated spectral test that adjudicates it or predicts robustness:
- **Single-axis pole:** Arditi 2406.11717 (refusal = one direction), 2508.20766 (rank-one safety injection).
- **Broad-subspace pole:** Wollschläger 2502.17420 (concept cones), Joad 2602.02132 (11 distinct directions, shared core), Teo 2507.20333 (safety in high-dim subspaces; low-dim projection preserves alignment but blocks jailbreaks).
- **Repo tie-in:** Probe-18 (`gateA/probe18_steer_opt350m.json`) showed protection is a **broad subspace by elimination** (orth-to-single-axis 0.122 ≈ random-orth 0.158, N=15) but **never positively parameterized it** (how many dims? what basis?). PR is that missing measurement.

## Falsifier (pre-registered kill conditions)
- **PRIMARY KILL:** `|Spearman(PR, k*)| < 0.30` across ≥30 held-out concepts (perm-null p>0.01, or fails cross-model replication) → PR does not capture causal dimensionality → the "spectrum distinguishes steerable-axis from broad-subspace" claim dies.
- **DOWNGRADE (secondary):** if PR predicts `k*` (|ρ|≥0.30) but does **NOT** predict robustness/reversibility (|ρ|<0.30 both) → conclusion downgrades to *"spectrum measures dimensionality, not safety"* — a clean, publishable negative against the safety-subspace framing (honest-negative floor).

## Priors to lose (pre-register against)
- Arditi single-direction: predicts `PR≈1` and `k*=1` for refusal specifically. If measured refusal `PR≫1`, that alone contests Arditi (report separately).
- Tan 2407.12404: steering is dataset-dependent and brittle → **control for N-pair dataset** (see confounds); PR must beat "brittleness is just a dataset property."

## Must-beat baselines
1. Raw effect size of the **top-1** difference direction (the naive single-axis assumption).
2. **Stable rank** `‖D‖²_F / σ₁²` (alternate spectral scalar) — PR must not be strictly dominated.
3. Naive **SAE-latent count** for the concept (GemmaScope) as a discrete dimensionality proxy.
4. For reversibility: **Mean-PCA-distance** (2505.16831) — PR-as-*rank* must add signal over their drift metric.

## Method
- **Substrates (activations only, no training except tiny unlearn probe):** Gemma-2-2B, Qwen2.5-1.5B, Llama-3.2-1B/3B. GemmaScope SAEs free for cross-check.
- **Concepts (~30–50):** refusal, sentiment, sycophancy, language-ID, day-of-week (Engels 2405.14860 multi-D control — should read `PR≫1`), several factual toggles.
- **Pipeline per concept:** collect N contrastive prompt pairs → difference vectors `D` (N×d) per layer → SVD → PR + stable rank.
- **Causal curve:** steer/ablate using only top-k singular directions of `D`; measure fraction of behavioral effect recovered vs k → define `k*` = rank to reach 90%.
- **Robustness:** OOD-prompt steering transfer (link 2407.12404) + ablation/jailbreak resistance (link 2507.20333).
- **Reversibility:** small unlearn→relearn probe on 1 model (link 2505.16831); does `PR`/`k*` predict recoverability.

## Confound controls (pre-registered)
- **N-pair bias:** PR is biased by sample count N → subsample-match N across concepts; report PR at fixed N.
- **Layer:** report per-layer; pre-commit to one analysis layer (mid-depth) chosen on a SMOKE concept set, frozen before the main run.
- **Day-of-week positive control:** must return `PR≫1` (known multi-D) or the metric is broken.
- **Haar null band:** significance of any subspace-overlap uses the QX1 finite-N random-rotation null band (`float_module` QX1 technique) — random high-dim subspaces have non-trivial overlap; the null band sets the real bar.

## Cheap SMOKE first ($0, cached)
Before the concept study: compute PR on the **existing Probe-18 in-V Fisher subspace** (OPT-350m basis already cached in cf_cascade) → a $0 positive answer to "how broad is the protected subspace" and a sanity check that PR ≫ 1 there (consistent with the broad-subspace-by-elimination result).

## Stats
≥30 concepts; bootstrap CI over concepts; permutation null; **cross-model replication required for a HIT** (same-sign, |ρ|≥0.30 on ≥2 model families).

## Cost / lane
Activations only → ~$5–15 on one A10 (batchable forward passes); SMOKE is $0 local on cached Probe-18 basis. No fine-tuning except the tiny reversibility probe.

## Standing floor (publishable regardless)
Either (a) PR predicts causal dimensionality **and** robustness = first validated spectral test for steerable-axis-vs-broad-subspace, resolving the refusal debate; or (b) PR predicts dimensionality but **not** safety = clean negative that spectrum ≠ safety. Both engage the live 2024–2026 refusal-geometry literature.
