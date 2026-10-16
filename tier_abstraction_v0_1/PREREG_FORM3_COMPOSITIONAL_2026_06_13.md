# PRE-REGISTRATION — Form 3: Compositional vs Holistic abstraction. LOCKED 2026-06-13 (user: "lock as drafted").

**Motivation.** Form-1 (geometry) REFUTED. Form-3 is the canonical "abstraction pays" case and the
biggest potential lift: does a FACTORED abstraction (parallel z_trans modules, one per attribute axis)
beat a capacity-matched HOLISTIC abstraction at transfer to **novel COMBINATIONS** of attributes?
The parent doc (PREREG_FORM_FRONTIER) flags this as acutely vulnerable to the Locatello pathology
("design the task to favor your method"). This doc's job is to make the task adversarially fair.

## Question (one sentence)
Holding total latent capacity and supervision constant, does splitting the abstraction into one
module per ground-truth factor improve generalization to attribute combinations never seen together
in training, relative to a single holistic latent?

## Substrate (locked candidate)
**3DShapes** (Burgess & Kim 2018): 480k 64x64x3 images, 6 INDEPENDENT ground-truth factors —
floor-hue(10), wall-hue(10), object-hue(10), scale(8), shape(4), orientation(15). Chosen because
factors are exactly independent and I did NOT design them — removes "I picked separable attributes."
Fallback: dSprites (5 factors, binary) if 3DShapes h5 retrieval is a hassle.

## Two arms (capacity-matched — the comparison is FACTORING, not budget)
- **COMPOSITIONAL:** F parallel z_trans modules, each dim d, module f trained to predict factor f.
  Shared frozen backbone OR small shared CNN trunk -> F heads. Total latent = F*d.
- **HOLISTIC:** ONE z_trans of dim F*d (identical total capacity), trained to predict ALL F factors
  jointly from the single latent (F linear heads off the one latent).
- Both: same trunk, same VIB (beta), same optimizer/epochs/seeds. ONLY the factoring differs.

## Hardened held-out-COMBINATION protocol (the anti-rig core)
1. **Novel pairings of INDIVIDUALLY-SEEN values** (true combinatorial generalization, NOT extrapolation
   to unseen factor values). Every factor value appears in training; only specific COMBINATIONS are held
   out. Example: hold out all (shape=cube AND object-hue=red) images; cubes and red each appear
   elsewhere in training, never together.
2. **Multiple RANDOM held-out splits (>=5), not one hand-picked.** Each split holds out a random set of
   ~10% of factor-pair combinations across 2 randomly chosen factor axes. Report the DISTRIBUTION of
   the compositional-minus-holistic gap across splits, not a single number. A hand-picked split is the
   rigging vector; randomizing it removes the degree of freedom.
3. **Both arms see the EXACT same train/test split.** No per-arm task tuning.
4. **Pre-commit the metric before any run:** top-1 accuracy on held-out-combination images, averaged
   over the held-out factor axes, paired bootstrap 95% CI on (compositional - holistic).

## Locked gate
- **CONFIRM Form-3-pays** iff compositional beats holistic on held-out-combination transfer by
  >= +3% top-1 (paired, CI excludes 0) in a MAJORITY (>=3/5) of random splits. (Higher bar than
  Form-1's +2% because compositional has more knobs and the rigging risk is higher.)
- **REFUTE** if compositional <= holistic within 3% in a majority of splits -> factoring is not the
  lever for combinatorial transfer at matched capacity.
- **Secondary read:** does the compositional advantage CONCENTRATE on the held-out combinations vs
  in-distribution (it should, if it's real composition and not just a better classifier)?

## Anti-rigging safeguards (Locatello defense, explicit)
- Factors are dataset-given, not chosen by me. Holdout is random across splits, not curated.
- Capacity matched exactly (F*d both arms). Supervision identical (both predict all F factors).
- In-distribution accuracy reported alongside held-out: if compositional also wins ID by the same
  margin, the effect is "better classifier," NOT "composition" -> would DOWNGRADE the claim.
- Report splits where holistic wins too (no cherry-pick).

## Infra (implementation, not locked science)
- Precompute backbone features ONCE, cache to FS (iNat-style) if using a frozen backbone; or small
  CNN trunk trained from pixels (3DShapes is simple). Lambda a10, cached substrate on FS.
- Reuse watcher (mtime-liveness) + local CPU smoke-test-before-boot discipline from the iNat run.

## What a CONFIRM / REFUTE would mean
- CONFIRM -> "the FORM that pays is compositional/factored" — a positive design principle, the first
  since Arm-C compartmentalization, and directly downstream of the pricing-bridge (factoring = pricing
  each axis independently). Big.
- REFUTE -> holistic latents already compose at matched capacity; factoring isn't the lever ->
  the pricing-bridge/omission account stands alone and the FORM frontier is largely closed (1 null,
  1 refute), pointing back to REGIME (Axis-2) or a structural reset.
