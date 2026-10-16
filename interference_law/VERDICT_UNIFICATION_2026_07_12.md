# VERDICT — Thread #7: THE UNIFICATION (PREREG 18693b8d, locked 2026-07-12). Landed, $0 local, n=120 (8A×15B), non-saturated.

## RESULT: FALSIFIED as registered. On-axis margin collapse does NOT govern deployed-head forgetting — its sign is even OPPOSITE my prediction. The frozen/plastic dichotomy STANDS. ‖Δθ‖ magnitude remains dominant; the frozen −cos law survives into the plastic regime only as a FAINT within-B whisper (ρ −0.20), which my on-axis reconstruction failed to explain.

## The scoreboard (predictor vs forget_old = DEPLOYED frozen-A-head AUC drop; median forget_old 0.094, IN partial band)
  predictor          pooled ρ   within-B ρ    pre-registered expectation        outcome
  onaxis (P1)         −0.258      −0.311       ≥ +0.50 (direction re-emerges)    FAIL — wrong sign, decisively
  offaxis (P2)        +0.034      −0.002       weak (<0.30)                       ~holds, but P1 fail makes it moot
  total ‖ΔF_A‖        −0.132      −0.184       positive shadow                    FAIL (wrong sign, weak)
  dtheta ‖Δθ‖ (P3)    +0.191    (per-B, n/a)   positive, < onaxis                 dtheta pearson +0.53 — still leads
  cosAB (frozen)      −0.199      −0.202       (the original coordinate)          faint correct-sign whisper
  cosAB → onaxis (P4) −0.038        —          < 0 (loop closure)                 NULL (−0.04) → P4 not real

## Reading it honestly
- **P1 is falsified, and not narrowly.** I predicted on-axis margin collapse along d_A would predict deployed forgetting at ρ ≥ +0.50.
  It came in at −0.26 pooled / −0.31 within-B: the WRONG SIGN. A-classes whose centroid margin along d_A shrinks MORE actually retain
  the deployed head slightly BETTER. My beautiful "one law in the right projection" hypothesis does not survive contact with data.
- **‖Δθ‖ still wins.** Even on the deployed-head metric (which I built specifically to expose direction), global weight-travel magnitude
  is the strongest predictor (pearson +0.53 between-B). #1's crown holds; the plastic regime is magnitude-governed.
- **The frozen law leaves a faint fingerprint.** Within-B, cosAB → forget_old is ρ −0.20: with a FULLY PLASTIC backbone, anti-parallel B
  still causes marginally more deployed-A forgetting. The −cos law whispers into the plastic regime — but weakly, and I did NOT succeed
  in reconstructing its mechanism (on-axis got the sign wrong).
- **P4 loop-closure is a mirage.** cosAB → onaxis is −0.04 (null); the verdict flag tripped on a borderline sign only. No real bridge.

## The most likely reason the test was mis-specified (a caveat, NOT a rescue)
The deployed head decides along its LOGISTIC weight vector w_A, which in a correlated feature space is NOT the centroid-difference
direction d_A I locked as the projection axis. AUC is rank/scale-invariant, so a uniform margin shrink along d_A need not drop the
head's AUC (which lives on w_A). I projected onto the wrong axis for THIS metric. That could mean the unification is untested rather
than false — but swapping d_A→w_A now and re-scoring would be exactly the post-hoc goalpost-move the prereg exists to forbid. It is a
NEW prereg (#7b), not a fix to this one. As registered, #7 is falsified.

## Bottom line for the arc
The frozen/plastic dichotomy is not (by this test) one law in two projections. Two regimes, two governing variables — direction
(frozen readout) and magnitude ‖Δθ‖ (plastic backbone) — remain the honest description. The −cos law does echo faintly into the plastic
deployed-head metric (ρ −0.20), leaving the door open, but the specific bridge I proposed (on-axis centroid drag) is not it.

## Caveats
Single seed; ResNet18; one FT strength (partial-forgetting confirmed, median 0.094). d_A-vs-w_A projection mismatch (above) is the
prime suspect and the natural #7b. Within-B ρ has 7 dof per B × 15 B — modest power; the −0.31 onaxis / −0.20 cosAB signs are stable
across pooled+within-B but magnitudes are soft. dtheta is per-B (15 values) so its within-B ρ is undefined; its lead is a between-B claim.
