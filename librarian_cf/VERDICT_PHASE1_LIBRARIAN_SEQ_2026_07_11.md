# VERDICT — Phase-1: Librarian sequential-domain CF (PREREG `PREREG_PHASE1_LIBRARIAN_SEQ_2026_07_10.md`). Landed 2026-07-11 07:15, $0 local GPU.

> **FILING NOTE (2026-07-14):** this verdict was written retroactively during a staleness audit. The run LANDED 2026-07-11
> (`out/seq_result.json`, `smoke:false`) but no verdict was ever written — the PREREG's status line still read "NOT run full.
> Next: LOCK -> python probe_librarian_seq.py". The result sat unbanked for 3 days. Numbers below are read straight from the artifact.

## RESULT: NON-DESTRUCTIVE OVERLAP — the pre-registered outcome. Sequentially adding domains to the reranker does NOT erode prior domains, even though the new-domain surge lands squarely IN the prior's curvature subspace.

## Setup
Base `cross-encoder/ms-marco-MiniLM-L-6-v2`, 573 real rules chunks, sequence **soccer -> nba -> nfl**, K=64, 4 epochs, lr 2e-5,
retention_frac 0.25, cuda. Metric: per-domain retrieval AUC + recall@3/@5 after each addition; plus `surge_overlap_on_prior` =
fraction of the new domain's update energy landing in the prior domain's subspace V, against a **swap control** (`inSwap`).

## The numbers
### Retention across the sequence (AUC)
  domain    after soccer   after nba     after nfl     joint (all-at-once)
  soccer      0.9572        0.9596        0.9554         0.9645
  nba           --          0.9627        0.9627         0.9662
  nfl           --            --          0.9467           --
Soccer moves 0.9572 -> 0.9596 -> 0.9554 across two subsequent domain additions: a **+0.0024 then -0.0042 wobble = noise, not erosion**.
nba is flat to the 4th decimal (0.9627 -> 0.9627) after nfl lands on top of it. Recall@5 likewise holds (soccer 0.873 -> 0.877 -> 0.877).

### Surge overlap on the prior (the mechanism read)
  after nba : soccer  inV **0.4036**  vs inSwap 0.0003
  after nfl : soccer  inV **0.3489**  vs inSwap 0.0003 ;  nba  inV **0.3819**  vs inSwap 0.0003
The new-domain update puts **35-40% of its energy inside the prior's subspace**, versus ~0.0003 for the swapped control — roughly
**1000x above the control**. The overlap is real, large, and direction-specific.

## Reading it — the dissociation
**Strong directional overlap + intact retention = NON-DESTRUCTIVE OVERLAP.** Landing in the prior's subspace is NOT sufficient to damage it.
This is the PREREG's named outcome and it converges with the rest of the program from an independent substrate (a text reranker, not a
vision readout or an LM):
- It matches cf_cascade's **#2b** headline — what protects is *non-destructiveness*, not avoiding the prior's directions; occupying/overlapping
  the prior's subspace is inert for damage.
- It matches the cosine-geometry read of this same system: `cos(rules, fantasy) = +0.671` = **CONSTRAINED, not contested** -> do NOT route,
  joint training is correct (`VERDICT_PHASE3_FANTASY_RULES_DIRECTION_2026_07_11.md`). Aligned capabilities coexist; overlap without antagonism
  is harmless.
- Joint (all-at-once) retention is nominally the best (soccer 0.9645, nba 0.9662) but the sequential path costs only ~0.005-0.009 AUC —
  i.e. sequential ~= joint here. There is no CF to remediate in this system.

## Caveats
**Single order** (soccer->nba->nfl); the PREREG's order-permutation arm was explicitly deferred and remains unrun, so order-dependence is
untested here. (NB: a later, independent result — `interference_law` Direction C, 2026-07-14 — finds frozen-readout stream forgetting IS
order-dependent/recency-weighted, so the permutation arm is not idle curiosity.) Single seed. Retention is measured from a high base
(AUC ~0.95-0.96, near-ceiling) — the recurring saturation caveat: there is little room to forget, which is itself part of why this reads
non-destructive. No Phase-2 (localize / F2 antagonism) was ever specced or fired; the arc jumped to Phase-3.

## Bottom line
The Librarian reranker shows **no sequential-domain catastrophic forgetting** across soccer -> nba -> nfl, despite the new-domain surge
depositing 35-40% of its energy in the prior's subspace (~1000x the swap control). Overlap is not damage. Production's joint-retrain is
validated a second way, and the "route the domains" question stays vetoed.
