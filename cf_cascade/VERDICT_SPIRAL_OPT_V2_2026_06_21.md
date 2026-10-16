# VERDICT — SPIRAL-OPT v2: OPT's depth-structure is LOAD-BEARING at parity, in OPT's own signature. 2026-06-21.

Pre-reg `PREREG_SPIRAL_OPT_V2` (lock `59ebf808`). CPU analysis on OPT per-layer artifacts I generated to Lambda
(`taskA_variation_opt.npz`, `perlayer_seed_opt.npz` via `taskavar_seed_opt_lambda.py`; guards: w_A SST acc 0.793 ~0.77;
fG/tG median 0.080/0.080 vs banked rung2b_opt 0.087/0.089). Scripts `opt4.py`, `opt3.py`. Six tripwires bound the run.
User adjudication 2026-06-21: depth measure canonical (see TRIPWIRE-4 flag).

## HEADLINE
The cross-arch dissociation EXTENDS to the structural (depth) layer. Pythia and OPT BOTH have depth-geometry —
**opposite carriers, opposite signs** — and on **OPT the depth-structure carries the protection load**, where on Pythia
it was inert. Morning's "depth-geometry closed" is **OVERTURNED at parity, in OPT's signature** (not Pythia's).

## OPT-1-v2 — EXISTENCE (re-affirmed, bit-stable)
OPT funnel = MLP (fc1/fc2) resolution+universality CONTRACTION with depth: fc1 a_curvres ρ−0.732/z−3.54 (breadth),
−0.837/z−4.02 (crosspair); fc2 −0.735/z−3.62; fc2 c −0.664/z−3.15. Reproduced bit-stable; guard (c) vs Pythia 8.9e-16.

## OPT-4-v2 — TWIST (mixed by carrier)
Per carrier, fragile__sst2 subspace consecutive principal-angle coherence + gap-decay, depth-shuffle null:
- **fc1: TWIST** (coh 0.702 vs shuffle 0.473, z+8.96; ρ_gap=−1.000 monotone decay, z−3.25) = true spiral (contract+twist).
- **fc2: contract-only** (coh 0.059 ≈ shuffle 0.058; no depth-coherent rotation — 16-dim in R^4096 near-orthogonal).
- q/k/v_proj: twist (coh ~0.61, z~8.8, ρ_gap −1.0). out_proj: none (coh 0.098).

## OPT-3-v2 — LOAD-BEARING (PARITY-CRITICAL). Two corroborating rows + mandatory swap (TRIPWIRE 3):
**Row 1 — seed concentrates deep / where contraction accumulates (deposit signal):** seed~depth fc1 ρ+0.648 z+3.14,
fc2 ρ+0.923 z+4.42 (v_proj +0.955 z+4.61); seed~resolution-value −0.541/−0.702 (seed high where resolution contracted).
Depth-shuffle null cleared.
**Row 2 — mandatory fragile→topcurv SWAP SURVIVES (deposit-specific, not curvature routing):** generic curvature
lam~depth is OPPOSITE (fc1 z−4.75 r−0.984, fc2 z−4.65 r−0.963) — curvature concentrates SHALLOW while the
forgetting-seed concentrates DEEP. The deep-concentration runs AGAINST the curvature gradient → not a curvature
artifact → deposit-specific. Swap survives on both MLP carriers + v_proj.

### TRIPWIRE 4 — operationalization error, recorded
The v2 pre-reg LOCKED the OPT-3 primary as ρ(seed, contraction-*|slope|*), which is NULL (fc1 z+0.24, fc2 z+0.09).
`|local slope|` is a poor proxy for "where contraction has *accumulated*" (= depth / low-resolution). The firing
measure — seed~depth — is the v1 pre-registered *direction* ("seed concentrates DEEP on OPT") and the v2 prompt's
explicit language ("high-depth contracted"). User adjudicated the depth measure canonical; the |slope| null is logged
as a mis-operationalization, not a negative result. (Locked pre-reg left byte-intact; this is the correction of record.)

## CROSS-ARCH FORK RESOLUTION
MLP contraction real (OPT-1) AND OPT-3-v2 fires (seed concentrates deep, deposit-specific, swap survived) →
**depth-structure LOAD-BEARING on OPT in its own signature.** Strongest cross-arch structural finding: same dissociation
at protection layer (deposit +0.055 sig) AND depth layer, signed mechanically OPPOSITE across architectures (Pythia
eff-rank collapse on qkv/h4h, inert; OPT resolution-contraction on fc1/fc2, load-bearing).

## PER-CARRIER (never pooled)
| carrier | contraction (OPT-1) | twist (OPT-4) | seed-load-bearing (OPT-3) |
|---|---|---|---|
| fc1 | strong (−0.84) | YES (spiral) | YES (deep, swap-survived) |
| fc2 | strong (−0.74) | no (contract-only) | YES (deep, swap-survived) |
| q/k/v_proj | weak/mixed | yes | v_proj yes; q/k no |
| out_proj | anomalous | no | no |

## DOWNSTREAM
SportsThought hierarchical-deep-map upgrade GATE = OPT-3-v2 → **PROCEEDS** (load-bearing confirmed at parity).

## OPT-CARRIER-v2 — carrier-split NULL (deposit not carrier-resolved by depth-geometry)
Generated per-carrier mapfrac to Lambda (`opt_carrier_pc_lambda.py`, 30 seeds; guards: A_forget 0.243 >0.15 not-VOID;
pooled mapfrac B/D 0.219/0.014 reproduces banked dispositive; B−A +0.081±0.021 confirms deposit load-bearing).
**Per-carrier deposit-mapfrac (arm B) is FLAT across comparable (in_dim 1024) carriers:** q 0.303, k 0.295, v 0.293,
out 0.294, fc1 0.294. spearman(mapfrac, contraction-strength)=−0.2 (no match), spearman(mapfrac, seed-depth)=−0.89.
**TRIPWIRE-4 caveat:** fc2 mapfrac 0.079 is NOT comparable — the fixed 128-dim map is 12.5% of fc1's R^1024 but 3% of
fc2's R^4096 (dimensionality-deflated). Excluding fc2, mapfrac is uniform.
**VERDICT:** per-carrier depth-geometry does NOT predict which carrier holds the deposit — same as Pythia SPIRAL-5
(carrier-split not predicted by depth-geometry). OPT's depth-structure is load-bearing GLOBALLY (OPT-3) but NOT
carrier-resolved. Does not change the SportsThought gate (already cleared by OPT-3-v2 global load-bearing).
