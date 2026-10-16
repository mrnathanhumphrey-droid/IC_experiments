# VERDICT — Direction #2: the DYNAMICS of interference (PREREG 9508ee75, locked 2026-07-12). Landed, $0 local 257s, n=120 pairs, 15 B-tasks × 7 checkpoints, non-saturated (final forget 0.099).

## RESULT: Interference is a SINGLE FAST COLLISION co-timed with task acquisition, then a PLATEAU. Forgetting is decided in the first ≤25 steps (early-warning ρ=+0.85) and does NOT grow with further training. My ballistic prior (D2-P1) and monotone prior (D2-P4) were both FALSIFIED by data; early-warning (D2-P2) strongly CONFIRMED; cosine-front-loading (D2-P3) NULL.

## The two curves that tell the story (median over 120 A-B pairs)
  checkpoint t:   0      25     50     100    200    350    500
  forget_old:    0.000  0.091  0.104  0.097  0.094  0.095  0.099    ← jumps by t=25, peaks t=50, FLAT thereafter
  acc_B:         0.17   1.00   1.00   1.00   1.00   1.00   1.00    ← B fully learned by the FIRST checkpoint
Forgetting accrues ENTIRELY during B-acquisition (both complete by t=25) then plateaus. The backbone keeps moving to t=500 (‖Δθ‖ grows)
but forgetting does not — the damage is done at acquisition; later travel is "safe."

## Predictions (as registered)
  D2-P2 EARLY-WARNING  ρ(forget@50, forget@500) = +0.846   ✅ CONFIRMED (≥+0.60). Who forgets is set by t≤50. Cheap early-warning works.
  D2-P1 BALLISTIC      early consecutive drift-cos = +0.008 ❌ FALSIFIED (predicted ≥+0.60) — see resolution caveat.
  D2-P3 COSINE-RATE    ρ(cosAB,forget@50)=−0.139 vs @500=−0.150, Δ=+0.011  ⭕ NULL. Not front-loaded — the −cos residual is
                       CONSTANT-weak across all time, not a transient early mechanism. (My "signal spent early" story was wrong.)
  D2-P4 MONOTONE       rebound fraction = 0.55  ❌ by the metric — but the threshold was mis-calibrated (see caveat); the AUC forget
                       curve is a flat PLATEAU, which is the honest read.

## Two caveats I will NOT paper over
- **D2-P1 is unresolvable at this checkpoint grid, so "falsified" overstates it.** The collision COMPLETES before the 2nd checkpoint
  (t=25): the earliest interval [0,25] already contains the whole event, so the first big increment v0 dwarfs the tiny post-plateau
  jitter increments v1,v2 → their consecutive cosine ≈ 0 by construction. This measures "collision finished before I looked again,"
  NOT "the trajectory rotates continuously." Ballistic-vs-curved DURING the collision needs fine checkpoints inside [0,25] (→ #2b).
- **D2-P4's 55% is threshold noise.** d′ here is ~4 (base AUC≈1.0), so the pre-registered 0.1 rebound cutoff is ~2.5% of d′ — it fires
  on plateau jitter. The AUC forget-curve (flat 0.09→0.10 after the jump) shows the real behavior: a PLATEAU, neither progressive
  forgetting nor meaningful recovery. "Rebounds common" is an artifact of a scale-mismatched threshold, not a B-overfit-frees-A dynamic.

## What this ADDS to the arc (the reconciliation)
The static ‖Δθ‖ law (#1: forget ∝ ‖Δθ‖) is really a statement about the EARLY COLLISION. Evidence: B0 traveled ‖Δθ‖=15.97 and forgot
0.338; all other B-tasks traveled ~7.1 and forgot ~0.09. The forgetting is fixed by acquisition-phase travel; the additional travel to
t=500 (flat forgetting) is non-damaging. So "magnitude governs" (plastic regime) = "the acquisition burst's weight-travel governs," and
this run shows the damage is TEMPORALLY front-loaded (first ≤25 steps) even though it is NOT cosine-directionally front-loaded (D2-P3).
Two-phase picture: (1) fast acquisition collision — forgetting jumps, damage set; (2) plateau — B refines to acc 1.0, ‖Δθ‖ grows, A
holds steady. Early-warning (D2-P2 +0.85) is the actionable payoff: forgetting is predictable from the first 10% of training.

## Follow-up (a NEW prereg if pursued — #2b, NOT done here)
Fine-grained checkpoints inside [0,25] (e.g. every 2 steps) to (a) resolve D2-P1 ballistic-vs-curved DURING the collision, (b) log
‖Δθ‖(t) to directly confirm damage = early-travel (not just endpoint), (c) test whether the d′-trough early-stop is a real intervention
once de-jittered. Also: does the collision timing track acc_B(t) causally (slower-learned B → later collision)?

## Caveats
Single seed, ResNet18, one LR (3e-4), CIFAR-100, deployed-head metric. acc_B saturates to 1.0 by t=25 (fast task) → collision and
acquisition are confounded in time at this grid; a slower/harder B or lower LR would spread the collision out (→ #2b). D2-P1/P4 metric
mis-specifications are named above and do not affect D2-P2 (the robust confirmed result) or the emergent one-shot/plateau finding.
