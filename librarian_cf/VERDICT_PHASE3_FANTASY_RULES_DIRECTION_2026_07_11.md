# VERDICT — Librarian Phase-3(2): fantasy-vs-rules direction (PREREG b52e2928, locked 2026-07-11). Landed, $0 local GPU.

## DECISION (pre-registered rule, met cleanly): NOT CONTESTED → HARD-NEG fix, DO NOT route.

## The three diagnostics all agree
**D1 — directional cosine (the detector's predictor, applied here):**
- cross-pole cos(rules, fantasy): base-gradient **+0.671**, fine-tuned Δθ **+0.074** (per-carrier min +0.006, max +0.291 — EVERY
  carrier positive, none anti-parallel).
- reliability scale: within-pole split-half cos rules +0.954 / fantasy +0.998; random-relabel null +0.370.
- Read: the two poles pull the reranker in ALIGNED / mildly-shared directions — the CONSTRAINED regime (like the detector's
  swap↔celebdf +0.59), NOT the anti-parallel antagonism (detector's swap↔diffusion **−0.93**). Threshold for "contested" was < −0.10;
  actual is strongly POSITIVE. Nowhere close.

**D2 — CF surge, both orders:**
- rules→+fantasy: rules 0.992→0.984 (erosion **+0.008**), fantasy lands 0.999. fantasy→+rules: fantasy 0.999→0.998 (erosion
  **+0.000**), rules 0.993. Both erosions sub-0.02 = noise → NO forgetting either direction.
- surge inV 0.437 / 0.451 vs swap ~0.0003 → the surge lands HEAVILY in the prior pole's Fisher curvature YET retention holds =
  **NON-DESTRUCTIVE OVERLAP** — the exact Phase-1 rules↔rules pattern, and the exact detector lesson that high inV ALONE ≠ contested.

**D3 — keeper intrusion (the concrete symptom), joint model:**
- 16 soccer-goalkeeper chunks, 6 fantasy-keeper chunks. Intruder-beats-true-positive rate = **0.0 / 0.0** (n=40 each), mean
  true-positive rank = 0.0 (the correct chunk ALWAYS ranks first). The jointly-trained reranker separates fantasy-"keeper" from
  soccer-"goalkeeper" PERFECTLY when both are candidates. The feared collision does NOT manifest at the reranker.

## Interpretation & remediation (what the decision buys)
1. **Do NOT route fantasy vs rules to separate poles.** They are constrained-coexisting, not antagonistic. One joint reranker is
   correct — production's joint-retrain is validated AGAIN (as in Phase-1). Routing would be over-engineering with no win.
2. **The keeper collision is NOT a reranker-direction problem.** Given both keeper chunks as candidates, the reranker already picks
   the right one 100% of the time. → If a keeper-precision gap is real in production, it lives UPSTREAM of the reranker (first-stage
   retrieval / candidate generation not surfacing the right keeper chunk, or missing cross-domain hard negatives there) — NOT in the
   fantasy-vs-rules reranker direction, and NOT fixable by routing.
3. **The VOIDED Oracle-agent fix is now doubly explained:** it downsampled fantasy 720→90 (data-starvation) to chase a precision gap
   that (a) isn't a directional antagonism and (b) the reranker doesn't even exhibit here. The right lever is cross-domain HARD
   NEGATIVES at the stage that actually confuses keeper (first-stage), never fantasy-downsampling, never pole-routing.

## THE CROSS-DOMAIN PAYOFF (capstone): one diagnostic, two regimes, opposite prescriptions
The SAME discriminative-direction-cosine test cleanly separates the two production systems and prescribes the CORRECT, OPPOSITE
remediation for each:
- **Detector**: cos(swap,diffusion) = **−0.93** → CONTESTED → ROUTE to separate heads → diffusion detection 0.68→0.96 (big win).
- **Librarian**: cos(rules,fantasy) = **+0.67 / +0.074** → CONSTRAINED → DON'T route; joint reranker is right; keeper = upstream hard-neg.
The cosine's SIGN is the portable decision variable (as the CF arc's directional mechanism predicts). It told us to route ONE system
and NOT the other — and both calls are correct on their measured symptoms.

## Caveats (honest)
Self-contained candidate pools (positive + 6 sampled negs), 6-layer MiniLM, single seed, one order per surge. The 0% keeper
intrusion is measured on small candidate sets (pairwise-ish), NOT full-corpus rank-1 — it shows the reranker CAN separate the two
keepers, which is exactly why any residual production gap points upstream, but it does not by itself measure first-stage recall.
Δθ cosine +0.074 is small-positive (near-orthogonal), not strongly aligned — fantasy and rules are largely INDEPENDENT directions,
which is the cleanest possible "safe to share one model" signal. sub-0.02 AUC = noise; don't over-read.
