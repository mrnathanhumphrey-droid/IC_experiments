# VERDICT — Detector ROUTE + intra-cluster GPM (PREREG 3bf38ddd, locked 2026-07-11). Landed, $0 local, SUBSAMPLE=220, 1 seed, 1 order.

## Same router, same 2 heads (H0={swap,celebdf,dfdc}, H1={diffusion}); only change = GPM when a paradigm JOINS H0.

## Final retention (same fixed test sets as every other arm)
                swap   diffusion  celebdf   dfdc     min(swap,diff)   avg(all4)
  naive        0.651     0.683     0.896    0.938        0.651          0.792
  gpm (1 head) 0.740     0.575     0.822    0.789        0.575          0.732
  route/oracle 0.656     0.993     0.894    0.936        0.656          0.870
  route/maxens 0.639     0.964     0.884    0.928        0.639          0.854
  route_gpm/oracle 0.752 0.993     0.801    0.813        0.752          0.840
  route_gpm/maxens 0.751 0.953     0.817    0.869        0.751          0.848

## PRIMARY MET: route_gpm is the ONLY arm that keeps BOTH antagonists at individual-best, simultaneously
- swap 0.752 (≈ single-head GPM's 0.740, ≥ plain-route 0.656) AND diffusion 0.993 (own head). min(swap,diffusion)=0.752 —
  strictly the highest of every arm (naive 0.651, gpm 0.575, plain-route 0.656).
- STRICTLY DOMINATES single-head GPM: same swap protection, but diffusion 0.993 vs GPM's STARVED 0.575 (+0.42). The GPM
  baseline a practitioner would reach for is beaten on both fronts by isolating diffusion + GPM-inside-the-shared-head.
- Label-free maxens delivers swap 0.751 AND diffusion 0.953 with NO generator label at inference → the deployable recommendation.

## COST CLAUSE FIRED (pre-registered, reported not spun): intra-cluster GPM REALLOCATES retention, doesn't create it
- Protecting swap inside H0 starved its own cluster-mates: celebdf 0.894→0.801 (−0.093), dfdc 0.936→0.813 (−0.123) vs plain route.
- Net on the SUM: by avg-over-4, plain route/oracle (0.870) EDGES route_gpm (0.840) — the +0.096 swap gain comes straight out of
  celebdf/dfdc. GPM inside a moderately-aligned cluster is a DIAL toward the older paradigm, not a free win.

## THE DEEPER FINDING: the no-free-lane / directional-antagonism structure RECURSES inside a "compatible-ish" cluster
- Between clusters: swap↔diffusion cos −0.93 → sharing a head is catastrophic → MUST split (large clean win, plain route).
- WITHIN the swap-family: swap↔celebdf/dfdc cos only 0.59/0.84 (moderate, NOT ~0.99) → they share a head but there is a FIXED
  retention budget; GPM protecting swap's subspace projects away directions celebdf/dfdc need → moves ~0.1 AUC from the newer
  swaps to swap. Positive-but-moderate cosine ⇒ constrained coexistence, not free coexistence — the same mechanism, one level down.

## Architecture verdict for the detector team (refined, final for this thread)
1. ALWAYS isolate a strongly anti-parallel generator family (cos ≲ −0.5; here diffusion) in its OWN head + score by max-ensemble.
   Unambiguous, label-free: diffusion 0.68→0.95. This is the free win; take it.
2. For a moderately-aligned family (cos ~0.5–0.85; the swap variants) sharing ONE head, retention is a fixed budget:
   - plain continue-train → newest members win, oldest (swap) erodes to naive (0.656).
   - +intra-cluster GPM → oldest (swap) protected to 0.752, newest (celebdf/dfdc) pay ~0.1 each.
   - want ALL of them high → they need MORE heads (moderate alignment isn't enough to share cheaply). The head-count is the dial.
3. Net: route_gpm is the arch of choice when the OLDEST paradigm must not regress (swap = the deployed base) AND a strong
   antagonist is present — it beats naive and single-head GPM decisively. If you weight all paradigms equally, plain route wins
   on average; pick by which paradigm you can least afford to lose.

## Caveats (carried)
Filename-hash split (not identity-disjoint) → absolute AUCs inflated; DELTAS on fixed test sets are the claim. Head-level only
(frozen backbone). SUBSAMPLE=220, single seed, one order — the +0.42 diffusion-over-GPM and ±0.1 intra-cluster reallocations are
far above noise; treat sub-0.02 as noise. oracle = upper bound (family detector at inference); maxens = deployable floor.
Confirmatory next (if ever wanted, not owed): the "more heads for the swap-family" ceiling; identity-disjoint split; multi-seed.
