# VERDICT — iNat Deep-Tree CONFIRMATORY (Form-1 geometry). 2026-06-13.

**RESULT: Form-1 (hyperbolic geometry as the abstraction form) is REFUTED.**

## Design (locked, per PREREG_INAT_DEEPTREE_2026_06_13)
- Substrate: iNaturalist 2021_valid, 100,000 images, frozen ImageNet ResNet-18 features (512d), cached
  to persistent FS (`inat_feats.npz`). Tiers: class(51) / order(273) / family(1103).
- Architecture: **3-tier stacked VIB** (Arm-E), class→order→family, each tier consumes prev tier's mu.
  Two arms trained separately: EUC (L2 mean prototypes) vs HYP (Poincaré ball, expmap0 latents,
  Einstein-midpoint prototypes, Poincaré distance).
- Transfer: **balanced-k** NCM (k∈{1,2,5,10,20} examples PER CLASS at target level) to NOVEL species
  (1/7 held out). 3 seeds, bootstrap 95% CI on paired hyp−euc.

## Result: hyp−euc, every cell negative, CI excludes 0 everywhere
| tier   | k=1   | k=2   | k=5   | k=10  | k=20  |
|--------|-------|-------|-------|-------|-------|
| class  | −0.193| −0.129| −0.116| −0.058| −0.037|
| order  | −0.069| −0.136| −0.107| −0.104| −0.113|
| family | −0.026| −0.037| −0.057| −0.076| −0.081|

EUC absolute acc healthy (class .67 / order .29 / family .10 @k=20) → model trains fine; hyp just loses.

## Gate outcome
- CONFIRM := hyp−euc ≥ +0.02 AND CI>0 at low-k (k≤2) AND grows with depth. → **not a single cell clears it.**
- REFUTE  := hyp ≤ euc at low-k across tiers. → **satisfied at every k, every tier.**
- **VERDICT: REFUTE.**

## Why this is a real result, not a method failure
- **Balanced-k dissolved the scout's only positive.** Scout (raw pools) showed class/pool=50 = **+1.3%**
  (sub-threshold, CI just >0). Same cell under balanced-k = **−0.193**. The +1.3% was a populous-class
  sampling artifact — exactly the confound balanced-k was pre-registered to remove. Confirmatory > scout.
- Hyp deficit at the coarsest tier SHRINKS as k grows (Einstein midpoint stabilizes with more points)
  but never crosses zero → hyperbolic adds variance without bias benefit; geometry isn't buying anything.

## Full Form-1 closure (both substrates)
- CIFAR-100 (shallow 2-level): NULL, fair-eval (Einstein-midpoint) confirmed.
- iNat (deep 6-level, 100k imgs, balanced-k, stacked 3-tier): **REFUTED**, hyperbolic actively hurts.
- **Depth-rescue hypothesis is dead.** Hyperbolic geometry does not buy tier abstraction at any depth.
  Does NOT touch pricing-bridge / pays-in-low-data (those are omission/compartmentalization, not geometry).

## Engineering notes (banked)
- Persistent-FS feature cache is the unlock: download+extract paid ONCE, every subsequent run loads
  `inat_feats.npz` in seconds. Confirmatory cost ~$0.14, ~6 min compute.
- Watcher final design: mtime-staleness liveness (NOT process-presence — that false-killed 2 instances),
  traceback-detect, stall+cap backstops, explicit-array terminate body + verify-0. Local smoke test
  before paying for a boot caught the ModuleList/Parameter + dtype bugs.
