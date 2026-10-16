# PRE-REG — SPIRAL-OPT v2: characterize OPT's OWN depth-geometry + test load-bearing parity at OPT's signature.
2026-06-21. Parallel agent. Supersedes the v1 direction (which mis-mapped Pythia's signature onto OPT — see
VERDICT_SPIRAL_OPT amendment). CPU-only on existing OPT per-layer artifacts. Six tripwires BINDING.

## TRIPWIRE 4 — OPT'S DEPTH SIGNATURE IS THE MEASUREMENT OBJECT (not a Pythia-relative prediction)
**OPT depth-geometry (DEFINITION, in OPT's own measure+sign):** per-carrier resolution + universality CONTRACTION
with depth, carried by MLP (fc1/fc2), absent on attention. Bit-stable anchor (reproduced in this harness 2026-06-21):
- fc1 `a_meanpairsim_curvres` ρ=−0.732 z=−3.54 (breadth) / ρ=−0.837 z=−4.02 (crosspair)
- fc2 `a_meanpairsim_curvres` ρ=−0.735 z=−3.62 (breadth)
- fc2 `c_universality` ρ=−0.664 z=−3.15 (p=0.0002); fc1 c ρ=−0.510 z=−2.43
- attn (q/k/v/out) + eff-rank: weak/absent (all |z|<2). Pythia's eff-rank-collapse signature does NOT live here.
This is OPT's funnel. The measurement template below is derived from THIS, not imposed from Pythia. (Methodology
ledger, 4th catch of the species today: matched-measurement-template — each substrate's own signature, not the other's.)

## OPT-1-v2 — EXISTENCE (RE-AFFIRMED, banked). No re-run beyond the reproduce-anchor (done, bit-stable above).

## OPT-4-v2 — TWIST on MLP carriers (does OPT's funnel twist, or contract-only?)
Per (fc1, fc2) SEPARATELY: consecutive-layer subspace principal-angle coherence across depth, depth-shuffle null.
- **Measurement:** for each carrier, per-layer top-K subspace V_L (R^D); principal angles between V_L and V_{L+1};
  coherence = mean cos(principal angles) vs depth, and ρ(coherence, depth) with depth-shuffle null.
- **Pre-reg (within-OPT):** coherent consecutive rotation that decays smoothly with depth-gap = a true spiral
  (contract + twist) in MLP; flat/noisy principal-angle structure = contraction-only (smooth funnel, no twist).
- **POSITIVE (TRIPWIRE 3, both rows):** consecutive-overlap coherence present AND ρ_gap (overlap decays with
  depth-gap) clears its depth-shuffle null. Per carrier, never pooled.

## OPT-3-v2 — LOAD-BEARING for seed-location (PARITY-CRITICAL)
Does the task-A seed concentrate in the layers where OPT's MLP contraction is sharpest?
- **Measurement:** per carrier (fc1/fc2): per-layer seed energy profile (from perlayer_seed_opt) vs the per-layer
  MLP-contraction-sharpness profile (the |layer-local slope| of `a_curvres`/`c` series from spiral_opt1.json).
  ρ(seed_energy, contraction_sharpness) with depth-shuffle null.
- **Pre-reg (parity hypothesis at OPT's signature):** OPT's deposit is load-bearing at the dispositive (B−D=+0.055
  sig, surgery-confirmed). IF OPT's depth-structure carries that protection, seed concentrates where MLP contraction
  is sharpest → ρ>0. Pythia's SPIRAL-3 was inert because Pythia's map is inert; this is the same measurement on the
  architecture where the map IS load-bearing.
- **POSITIVE (TRIPWIRE 3, THREE rows):** per-layer seed-vs-contraction ρ AND depth-shuffle null AND the MANDATORY
  Cut-1 fragile→topcurv swap (if seed concentrates with contraction, swap fragile→topcurv at OPT; lift must NOT
  survive — if topcurv-contraction trends the same way, it is curvature-family routing, not deposit-specific → VOID).
- **FORK:** ρ>0 + swap survives → depth-structure LOAD-BEARING on OPT (morning's close overturned at parity, in OPT's
  signature). ρ≈0 (no trend) → inert on OPT too (close HOLDS at parity; hierarchical-map upgrade dies; dissociation
  deepens to the depth layer).

## OPT-CARRIER-v2 — carrier-split parity (does fc1-vs-fc2 contraction predict deposit-carrier?)
Per-carrier (fc1 vs fc2, never pooled): does the carrier with sharper MLP contraction hold the load-bearing deposit?
Cross-check vs the dispositive's PER-CARRIER mapfrac (fc1 vs fc2). **Per-carrier mapfrac is NOT on disk** (dispositive
mfB/mfD are per-step 30-vectors) → REQUEST from main lane; do NOT proxy from per-step curves.
- **POSITIVE:** per-carrier contraction order == per-carrier deposit-mapfrac order (fc1 vs fc2). Mixed → report which.

## CROSS-ARCH FORK v2 (pre-committed, at OPT's signature)
- MLP contraction real (✓ established) AND OPT-3-v2 fires (seed concentrates where MLP contracts, swap survives) →
  depth-structure LOAD-BEARING on OPT in its own signature. Cross-arch dissociation extends to the structural layer:
  Pythia & OPT both have depth-geometry, opposite carriers + opposite signs, and on OPT it carries protection load.
- MLP contraction real AND OPT-3-v2 inert → depth exists but inert for load-bearing; morning's close holds at parity;
  hierarchical-map upgrade dies cleanly; dissociation must live elsewhere than depth.
- Mixed by MLP carrier (fc1 fires, fc2 doesn't or vice versa) → partial parity; report which carrier; map to deposit.

## DISCIPLINE (carry all)
Per-(layer,carrier) NEVER pooled — fc1/fc2 separate, never averaged. Depth-shuffle null per carrier per measurement.
Curvature-controlled (a_curvres, c) = clean read. Effect size vs OPT's OWN benchmark (fc1 ρ−0.84 z−4.02), NOT Pythia's.
Pre-reg direction at OPT's signature (contraction sign). Cut-1 swap mandatory on any OPT-3-v2 positive. Reproduce-anchor
before banking (done). STOP-on-missing-canonical: no proxy of per-layer subspace/seed from crosspair profiles.

## REQUIRED ARTIFACTS (TRIPWIRE 2; main lane to generate — w_A checkpoint exists, extraction not retraining)
- `taskA_variation_opt.npz` — per-layer fc1/fc2 (and attn) subspace eigenvectors `fragile__{sst2,ag}__{carrier}` AND
  `topcurv__{sst2,ag}__{carrier}` (24,D,16). [OPT-4 twist + OPT-3 swap control] — MISSING
- `perlayer_seed_opt.npz` + overlap json — per-layer task-A seed energy per carrier (24,…). [OPT-3 seed concentration] — MISSING
- `deflation_artifacts_opt.npz` — per-layer contest/M1 if needed for the contraction-sharpness control. [OPT-3] — MISSING
- per-carrier dispositive **mapfrac** (fc1 vs fc2 deposit fraction). [OPT-CARRIER cross-check] — MISSING (request)
