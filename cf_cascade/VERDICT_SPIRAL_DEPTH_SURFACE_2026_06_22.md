# VERDICT — depth×SPIRAL surface (OPT-350m, layer-surgery 6/12/18/24). 2026-06-22.

All three SPIRAL-OPT measures run across the depth axis on matched-acc surgered checkpoints (acc 0.763/0.820/0.800 +
full-24 anchor), identical deposit per depth (each leg loads the same `opt350m_d{K}_sst2_wA.pt`). Readers:
`opt1_depth.py` (funnel), `opt4_depth.py` (twist), `opt3_depth.py` (seed). Anchors reproduced exactly each leg
(funnel fc1 −0.837 / twist fc1 coh 0.7024 / seed fc1 rho_seed_depth +0.648). Per-(layer,carrier) never pooled.

## The surface — fc1 vs fc2 dissociate, and the split is MEASURE-DEPENDENT
| measure | fc1 (up-proj) | fc2 (down-proj) |
|---|---|---|
| **funnel** (cross-task contraction, opt1 a_meanpairsim_curvres) | **depth-BUILT** +0.83→−0.18→−0.43→−0.84 (absent/inverted @d6, monotone) | depth-intrinsic −0.94/−0.24/−0.84/−0.74 (neg @ all) |
| **seed-deep** (opt3 rho(seed_conc,depth)) | **depth-BUILT** +0.03→+0.27→+0.68→+0.65 (flat @d6, monotone) | depth-intrinsic +0.54→+0.60→+0.92→+0.92 (deep @ d6 already) |
| **twist** (opt4 fragile-subspace coherence) | **depth-ROBUST** coh 0.60/0.67/0.70/0.70 (strong & coherent @d6, rg=−1.0 all) | absent @ all — coh 0.06, never twists (contract-only) |

## Reading
- **fc1's forgetting-relevant STRUCTURE is built by depth, but its geometric ROTATION is intrinsic.** The cross-task
  resolution funnel AND the deep seed-concentration both EMERGE monotonically as you stack layers above (absent at 6
  layers); yet fc1's layer-to-layer subspace coherence (the spiral/twist) is already strong at d6 (coh 0.60) and barely
  moves. Two different mechanisms inside one carrier: WHERE/HOW-MUCH it contracts needs the deep stack; HOW its fragile
  subspace rotates does not.
- **fc2 is the "always-on" contraction carrier.** Negative funnel + deep seed at every depth incl. d6, but geometrically
  flat — never twists. Depth-intrinsic on the energetic measures, null on the geometric one.
- **Attention q/k/v** spiral robustly at all depths (coh ~0.58–0.66, rg −1.0); their funnel emerges with depth (like
  fc1). out_proj never twists (like fc2) and its seed-deep weakens at full depth (inverted).

## SportsThought parity gate — stable
The pre-reg PRIMARY seed~contraction-SHARPNESS test (the signal that would overturn the close at parity) stays NULL at
EVERY depth: max |z_seed_sharp| = 1.8 (d6 fc2, wrong sign), never clears 2 on any carrier/depth. Depth is NOT the
missing variable that turns it on. The OPT-3-v2 verdict — load-bearing via seed~DEPTH (+0.648) + swap, NOT seed~sharpness
(user-adjudicated depth-measure canonical) — is depth-stable. fc1's seed~depth is itself depth-built; fc2's is robust.

## Caveats (T4)
- "Depth-BUILT monotone" reads are 4-point trajectories (same n=4 underpowering as the funnel leg): across-depth ordering
  is clean but permutation p capped ~0.083. Signal = strict monotonicity + per-depth significance at deep ends.
- coh VALUE (not z) is the depth-comparable twist quantity: z grows with depth only because more layers = stronger
  shuffle null. fc1 coh 0.60 @ d6 is already high → twist genuinely depth-robust, not a power artifact.
- d12 ran a uniform-flat funnel waypoint (acc 0.820 hottest); d18 restored — single-variant, not a depth effect.

## Artifacts
`gateA/spiral_opt1_depth.json` (funnel), `spiral_opt4_depth.json` (twist), `spiral_opt3_depth.json` (seed).
Depth substrates `gateA/{crosspair_residual,taskA_variation,perlayer_seed}_optd{6,12,18}.*` + full-24 banked.
Generators `cloud/{crosspair_depth,taskavar_seed_depth}_lambda.py` + launchers `cloud/lambda_launch_depth_{cpr,tav}.py`.
Funnel leg also banked separately: `VERDICT_FUNNEL_DEPTH_2026_06_22.md`.
