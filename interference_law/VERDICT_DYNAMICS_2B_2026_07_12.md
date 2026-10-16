# VERDICT — Direction #2b: INSIDE the collision (PREREG 8b4671be, locked 2026-07-12). Landed, $0 local 201s, 8 B-tasks × 2 LR conditions, fine checkpoints.

## RESULT: The fine grid OVERTURNS #2's "co-timed" headline — forgetting LAGS acquisition by ~15-20 steps. The collision is BALLISTIC (straight-line, drift-cos +0.78, 2b-P1 ✅). Two auto-flags are MISLEADING and corrected below: 2b-P2 "saturation" is a window-anchoring artifact (safe-travel IS present), 2b-P4 "intrinsic" over-claims (magnitude-invariance is confounded by unequal total travel, NOT tested).

## The fine-resolution curves (FAST, LR 3e-4, median over 8 B-tasks)
  step:    0     1     2     3     5     8    12    18    25    40    70   120
  acc_B:  .17   .58   .79   .85   .91   .96   .99  1.00  1.00  1.00  1.00  1.00     ← B LEARNED by ~step 18
  forget:  0   .000  .002  .007  .019  .045  .063  .077  .089  .107  .100  .098    ← still RISING to step 40
  ‖Δθ‖:   0    1.0   1.6   2.1   3.0   3.9   4.8   5.7   6.2   6.7   7.0   7.0
t90_accB = 6.5 steps ; t90_forget = 21.5 steps. Forgetting TRAILS acquisition.

## THE HEADLINE (new, overturns #2): forgetting LAGS acquisition — damage accrues during post-acquisition SETTLING
#2's coarse grid (first checkpoint at t=25) made forgetting and acquisition look simultaneous. At fine resolution they are NOT: acc_B
saturates to 1.00 by step ~18, but forgetting keeps climbing to step ~40 (peak 0.107) while the weights keep settling (‖Δθ‖ 5.7→6.7
from step 18→40). So the backbone FINISHES learning B, then continues to damage A for ~15-20 more steps as it settles into the B-solution.
The damage is done during post-acquisition weight-travel, not at the acquisition moment. Three phases: (1) steps 0-3 acquisition begins,
forgetting negligible; (2) steps ~3-40 BALLISTIC COLLISION, forgetting plunges straight-line, LAGGING acquisition; (3) steps 40+ PLATEAU.

## Predictions (as registered) — with honest corrections to two auto-flags
  2b-P1 BALLISTIC        acq-window drift-cos = +0.780   ✅ CONFIRMED (≥+0.50). The collision is a straight-line plunge (direction set
                         early, fixed). [Caveat: window anchored on t90_accB covers steps 0-8 = the EARLY plunge; the lagged bulk
                         (8-40, where most forgetting happens) staying ballistic is a descriptive follow-up not covered by this window.]
  2b-P3 ACQ-LOCKED       t90 slotgap = 3 (FALSE)         ✅ CORRECTLY FALSIFIED — and this IS the finding: forgetting lags acquisition
                         by ~15-20 steps. (Not "failure" — the falsification is the discovery.)
  2b-P2 SAFE-TRAVEL      damage/travel ratio 0.85 → flag "SATURATION"  ⚠️ FLAG WRONG. The window was anchored on t90_accB (step 6.5),
                         but forgetting LAGS, so the "post-acquisition" window (step 8→120) still contained the active forgetting RISE
                         (0.045→0.107) → ratio ≈ 1. Descriptively safe-travel IS present: step 40→120, ‖Δθ‖ +0.25 while forget −0.01
                         (travel continues, damage stops). The registered metric mis-fired due to the (newly discovered) lag; re-anchor
                         on t90_forget → safe-travel visible. Falsified-as-registered, phenomenon real, metric bug named.
  2b-P4 INTRINSIC        final FAST .098 vs SLOW .033; t90 21.5→95 → flag "INTRINSIC"  ⚠️ FLAG OVER-CLAIMS. Time-dilation is REAL (SLOW
                         spreads the collision ×4.4 in steps — the burst is not an LR artifact, it dilates with learning speed). BUT the
                         "same basin" magnitude clause fired only on the "OR slow lower" branch: SLOW forgot 3× LESS (.033 vs .098)
                         BECAUSE it traveled 2.6× less ‖Δθ‖ (2.73 vs 7.0) — exactly the ‖Δθ‖ law. Magnitude-invariance is NOT tested;
                         it is CONFOUNDED by unequal total travel. To test it, MATCH ‖Δθ‖ across LR (run SLOW longer), not step count.

## What #2b establishes (net, honest)
1. The collision is BALLISTIC (straight-line drift, cos +0.78) — direction fixed by the early gradient. ✅
2. Forgetting LAGS acquisition by ~15-20 steps — damage accrues during post-acquisition settling, NOT at the acquisition moment. This
   corrects #2's coarse-grid "co-timed" claim. ★ new
3. LR sets the collision TIMESCALE (SLOW dilates ×4.4) — the one-shot is time-dilated by learning speed, not created by it. ✅
4. Whether LR changes the collision MAGNITUDE is confounded (SLOW traveled less ‖Δθ‖) → untested; needs a match-‖Δθ‖ run. ✗ (open)
5. Safe-travel (‖Δθ‖ grows, forgetting flat) is descriptively present after step ~40 but my acquisition-anchored metric missed it. (open)

## Follow-up (#2c, NEW prereg if pursued — NOT done here)
Re-anchor the safe-travel window on t90_forget (fixes 2b-P2) + MATCH total ‖Δθ‖ across LR conditions instead of step count (fixes 2b-P4
magnitude confound: run SLOW until ‖Δθ‖=7.0 and ask whether it then forgets ~0.098 = LR-invariant magnitude, or still less = LR-gentle) +
measure whether the LAGGED bulk (steps 8-40) stays ballistic (extend the P1 window past acquisition).

## Caveats
Single seed, ResNet18, CIFAR-100, deployed-head metric, 8 B-tasks. acc_B still fast (0.85 by step 3) even at these grids — the collision
sits mostly in steps 3-40. Two registered metrics (P2 window, P4 travel-match) were mis-specified given the lag, named above; they do NOT
affect the two robust results (2b-P1 ballistic, and the acquisition→forgetting LAG). The lag is the load-bearing new finding.
