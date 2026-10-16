# PRE-REGISTRATION — Axis-G: Objective-Pricing Granularity. LOCKED 2026-06-14.

**Motivation.** FORM frontier closed (Form-1 + Form-3 both REFUTE) → the lever is OMISSION /
objective-pricing, not form. The synthesis says "more pots on the table beats one pot." We have
only ever tested a 2-objective split (Arm-C, +0.13 selectivity on CIFAR). This tests whether the
pricing lever is GRADED (more objectives keep buying abstraction) or BINARY (one split suffices),
and replicates the Arm-C selectivity gain on a NEW substrate (iNat).

## Substrate
Cached iNat frozen-ResNet18 features (`/lambda/nfs/iNat/inat_feats.npz`, 100k imgs, 512d) with 7
taxonomic levels (kingdom..species). Loads in seconds — near-free.

## Architecture (locked)
Chain of bottlenecked VIB tiers, FINEST consuming raw features, each next tier trained on a COARSER
level and consuming the previous tier's mean. The coarsest objective (class) sits on TOP. Euclidean
only (geometry refuted). Objective-count k = number of stacked tiers; top objective fixed = class.
- k=1: [class]                      (class tier consumes raw)
- k=2: [order, class]               (class tier consumes order-distilled rep)
- k=3: [family, order, class]       (class tier atop family->order chain)

## Metric (locked) — top-tier SELECTIVITY w.r.t. class
On the class-tier representation, fit an ORDER-level NCM; for each misclassified query, check whether
the predicted order shares the TRUE order's class. selectivity = within_class_error_rate − chance,
where chance = P(two random distinct orders share a class). Higher = order-confusions respect the
class boundary = the tier has abstracted class structure cleanly.

## Gates (locked, 3 seeds, bootstrap 95% CI)
- **G0 (sanity):** selectivity(k=1) > 0 with CI>0. (Metric valid; one objective already abstracts.)
- **G1 (THE LEVER):** selectivity(k=3) − selectivity(k=1) ≥ +0.02, paired CI>0. → granularity buys
  abstraction; the pricing lever is real and graded enough to pull.
- **G2 (graded vs binary):** selectivity(k=3) − selectivity(k=2) CI>0 → GRADED (headroom beyond one
  split); else k=2≈k=3 → BINARY (an objective split helps, but only the first one).

## Verdict logic
- G1 holds → **pull the lever**: objective-pricing granularity is a confirmed design principle;
  next is the semantic-DISTANCE axis (does the gain scale with how far apart the objectives span the
  tree — Wilson's prediction) and applying it in production abstraction stacks.
- G1 fails → no granularity lever beyond (at most) one split → **architect**: build the float module
  to SEARCH for levers rather than hand-guessing them. (The honest "we don't know the lever" branch.)

## Out of scope for v0 (deferred)
Semantic-distance arm (FAR vs NEAR spanning stacks) — run only if G1 holds. Don't debug the lever's
EXISTENCE and its DRIVER simultaneously.

## AMENDMENT 2026-06-14 — PARALLEL re-test (the Arm-C-faithful architecture)
The SEQUENTIAL run (above architecture) INVERTED: selectivity 0.645(k1) > 0.613(k2) > 0.600(k3),
k3-k1 = -0.045 [CI excludes 0], G1 FALSE. But sequential tiers (class CONSUMES order CONSUMES
family) BURDEN the abstract tier — the opposite of Arm-C, where the fine objective lives on a
SEPARATE parallel path so the coarse tier is FREED. Sequential result stands as a BOUNDARY for
sequential stacking; this amendment is a pre-committed RE-DECOMPOSITION onto the parallel axis (the
RMD-SRC recover-vs-boundary move), NOT a goalpost change to the sequential gate.
- **CONJUNCTION architecture (the principle, not just "parallel"):** objectives must price ONE
  representation IN CONJUNCTION (jointly, at once), not pass through each other as serial GATES.
  Realized as: shared trunk = the conjunction (all k objectives jointly shape it) + separate
  bottlenecked VIB heads = the compartmentalization (each level its own readout off the SAME trunk;
  no head consumes another). NOT one rep doing everything (Arm-A failure), NOT serial gates (the
  sequential inversion). Class head always present. Measure CLASS-head selectivity vs k.
  k=1 [class] / k=2 [class,order] / k=3 [class,order,family].
- **Read the k-ladder as ONE JOINT SHAPE, not 3 independent gates** (reading G0/G1/G2 as serial
  pass/fail repeats the gating mistake at the analysis level). Core question: does class-tier
  selectivity RISE as objectives are conjoined? G1 (k3-k1 >= +0.02, CI>0) is the summary, but read
  the whole monotone trend.
- **The inversion is positive evidence for conjunction:** serial gating decayed selectivity
  monotonically (-0.045) = each gate strips what the next can't recover = the objectives INTERACT
  and gating destroys the interaction. Conjunction should preserve/exploit it.
- **Verdict:** selectivity RISES with conjoined objectives -> lever real, needed conjunction -> PULL.
  Flat/inverts under conjunction too -> granularity genuinely doesn't pay -> ARCHITECT.
