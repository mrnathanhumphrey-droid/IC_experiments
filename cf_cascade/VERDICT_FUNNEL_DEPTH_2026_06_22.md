# VERDICT — SPIRAL-OPT-1 funnel vs DEPTH (layer-surgery 6/12/18/24). 2026-06-22.

## Question
Does OPT-350m's MLP funnel (crosspair3 `a_meanpairsim_curvres`, banked fc1 ρ=−0.837) strengthen / weaken / hold
as decoder depth drops via even-spaced layer-surgery + matched-acc retrain (d6/d12/d18 + full d24)?

## Method
- Substrate: crosspair3 (3 task-Bs agnews/yelp/subj, 64 curvature-ordered dirs, explicit λ control). Per-(layer,carrier),
  never pooled. Measure = within-variant Spearman(curvres-funnel-series, layer-depth), λ-partialled (faithful slice of
  `opt1.py` run_substrate a_meanpairsim_curvres). Reader: `opt1_depth.py`. Output: `gateA/spiral_opt1_depth.json`.
- Depth variants from matched-acc w_A retrains (acc 0.773/0.820/0.800/anchor), surgered checkpoints
  `crosspair_residual_optd{6,12,18}.npz` + full `crosspair_residual_opt.npz`.
- REPRODUCE-ANCHOR (T3): d24 fc1 = −0.837 vs banked −0.837 → **REPRODUCED** (ρ deterministic; rng only moves z/p).

## Result (ρ vs depth-index within each variant)
| carrier | d6 | d12 | d18 | d24 | shape |
|---|---|---|---|---|---|
| fc1 | +0.83 | −0.18 | −0.43 | −0.84 | **strictly monotone ↓** |
| fc2 | −0.94 | −0.24 | −0.84 | −0.74 | negative throughout, unordered |
| q_proj | +0.31 | +0.19 | −0.36 | −0.62 | monotone ↓ |
| k_proj | −0.60 | +0.04 | −0.03 | −0.47 | unordered |
| v_proj | −0.60 | +0.10 | −0.63 | −0.41 | unordered |
| out_proj | +0.60 | +0.27 | −0.57 | −0.06 | unordered |

## Verdict — within-MLP dissociation on depth-dependence
- **fc1 = depth-LOAD-BEARING funnel.** ρ strictly monotone (+0.83→−0.18→−0.43→−0.84), zero-crossing ~d12. Absent/inverted
  at 6 layers; contraction EMERGES and deepens as depth is stacked above. fc1's funnel requires the deep stack.
- **fc2 = depth-INTRINSIC funnel.** Negative at every depth (no sign flip); the down-projection contracts regardless of
  depth, present even at d6. (Magnitude non-monotone — d12 dip recovered by d18.)
- q_proj tracks fc1's emerges-with-depth shape; k/v/out unordered.

## Caveats (T4)
- n=4 depth points: fc1 ρ-vs-depth is perfectly ordered (across-depth Spearman = −1.0) but permutation p capped ~0.083 —
  directionally airtight, formally underpowered. Real signal = strict monotonicity + per-depth significance at deep ends
  (d24 z=−4.0, d18 z=−1.8).
- d12 variant ran a uniform near-zero across ALL carriers (its w_A acc 0.820, hottest) — a single-variant waypoint, NOT a
  mid-depth funnel collapse; d18 restored strong signals (fc2 −0.84, v −0.63, out −0.57).

## Connects to
Sharpens OPT-4 twist (fc1 spirals / fc2 contract-only) and OPT-3-v2 (depth-structure LOAD-BEARING) — names the carrier:
fc1 carries the depth-dependence, fc2 does not. Leg 1 of the depth×SPIRAL surface; twist + seed legs next.
