# PRE-REG — Full principal-angle spectrum predicts interference ASYMMETRY (what min-angle cannot)

- **Date:** 2026-07-27
- **Status:** LOCKED — pre-registered. Not fired.
- **LOCK:** `4d5b9bc9` (sha256[:8] of draft body at lock, 2026-07-27T00:06; lock order 3 of 4)
- **Arc:** interference_law
- **Priority:** #2 of the 2026-07-27 subspace/eigenspace audit (★★★★ — fuses into #1 Hodge for near-free)

## One-line hypothesis
A **directed** subspace-overlap statistic derived from the **full ordered principal-angle spectrum** (not the min/first/mean scalar everyone uses) predicts the **asymmetric** part of pairwise interference, `ΔF(A,B) = F(A|B) − F(B|A)` — which any *symmetric* scalar angle cannot predict by construction.

Specific candidate: **directed subspace projection energy** — the fraction of task-A's discriminative subspace captured by task-B's top-k span, `p(A→B) = ‖P_{U_B} U_A‖²_F / k`, is **not symmetric** (`p(A→B) ≠ p(B→A)`) and is spectrum-derived. Predict `ΔF ∝ p(A→B) − p(B→A)`.

## Why now / the wedge
Every published angle→interference law collapses the spectrum to **one symmetric scalar**:
- 2603.02224: **min** principal angle (`F = α·sin²θ_min + β`). Symmetric → structurally blind to ordering.
- Nature Human Behaviour 2025 (s41562-025-02318-y): **first** principal angle.
- 2606.13637: angle-**drift** (r=−0.862). Scalar.

A symmetric scalar literally cannot predict `F(A|B) ≠ F(B|A)`. Interference asymmetry (task order matters — 2205.13323, 2502.03350) is exactly the **transitive/gradient term** of the Hodge split in #1. So this thread is the geometric instrument that explains the gradient half, while #1 explains the harmonic half.

## Falsifier (pre-registered kill conditions)
- **PRIMARY KILL:** no full-spectrum / directed statistic reaches `|ρ|≥0.30` with `ΔF` (perm-null p<0.01, replicated cross-dataset) → the spectrum adds nothing over min-angle → **kill**; the 2603.02224 min-angle law stands as complete and asymmetry is not geometric.
- Report the symmetric-magnitude fit as a sanity anchor (min-angle should still predict `|F|`, not `ΔF`).

## Priors to lose (pre-register against)
- 2603.02224 min-angle (symmetric — **cannot** predict `ΔF`; that's the wedge, but if directed-overlap fails too, min-angle wins the whole game).
- NHB 2025 first-angle; 2606.13637 drift. All must be **beaten on the ΔF target specifically**.

## Must-beat baselines
1. **min** principal angle (2603.02224 scalar).
2. **first** angle; **mean** angle over the spectrum.
3. Symmetric full-spectrum stats (Binet–Cauchy / Grassmann distance) — these are symmetric so should ALSO fail on ΔF; included to prove the win is specifically about *directedness*, not *fullness*.
4. Effective-overlap-dimension = `#{θ_k < 15°}` (directed variant).

## Method
- **Shared with #1** — same cached subspaces (Qwen×banking77 Face-1c grid) plus the frozen resnet18 CIFAR/DTD matrices (directed-FT cheap, cached from P0/P1).
- Requires a **directed** interference target: use Face-1c directed-F, or frozen directed forward-transfer/forgetting matrices.
- Per pair compute: full principal-angle spectrum between top-k subspaces; directed projection energies `p(A→B), p(B→A)`; their difference.
- Correlate directed-overlap-asymmetry with `ΔF` across all ordered pairs.

## Confound controls (pre-registered)
- Same projection-dim sweep {8,24,48} and Haar null band as #1.
- Subspace rank k pre-committed (report k∈{5,10}); k chosen on SMOKE, frozen.
- **Symmetry placebo:** the symmetric baselines (Binet–Cauchy) must score ≈0 on ΔF — if a symmetric stat "predicts" ΔF, the target is contaminated (leakage check).

## Stats
Seeds ×3; bootstrap CI over pairs; perm-null; **cross-dataset replication** (CIFAR+DTD, or Qwen+one more) required for a HIT.

## Cost / lane
Near-$0 when fused with #1 (same subspaces) + frozen matrices already cached; ~$1–2 standalone.

## Standing floor (publishable regardless)
- **Null:** confirms 2603.02224 min-angle is the whole story and interference asymmetry is non-geometric.
- **Hit:** first geometric descriptor that predicts interference **asymmetry / task ordering** — a dimension every scalar-angle paper is structurally blind to.
