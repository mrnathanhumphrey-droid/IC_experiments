# PREREG — Direction #2c: is ‖Δθ‖ the SOLE governing variable, or does the PATH matter? + fix #2b's two metric bugs.

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune, deployed frozen-A-head metric (continuity #7/#2/#2b). $0 local. Reuses probe_dynamics_2b machinery + LR sweep.

## What #2b left open (and #2c closes)
#2b found: collision is BALLISTIC (+0.78) and forgetting LAGS acquisition by ~15-20 steps. Two registered metrics MIS-FIRED because of
that lag: (2b-P2 safe-travel) window anchored on t90_accB, but forgetting lags → "post" window still held the rise → false "saturation";
(2b-P4 magnitude) SLOW-LR forgot 3× less but ALSO traveled 2.6× less ‖Δθ‖ → magnitude-invariance CONFOUNDED, untested. #2c fixes both and
promotes the confound to the HEADLINE test: across LR, is converged forgetting a function of converged ‖Δθ‖ ALONE (path-independent,
#1's law is sufficient), or does the PATH (LR/settling) matter beyond ‖Δθ‖?

## Design: LR SWEEP, each run to convergence (forgetting plateau)
Four conditions, each fine-checkpointed to reach its forgetting plateau; per-condition ‖Δθ‖(t), forget(t), acc_B(t), A-centroid p_A(t):
  LR 6e-4 : CKPT [0,1,2,3,5,8,12,20,35,60,100]
  LR 3e-4 : CKPT [0,1,2,3,5,8,12,18,25,40,70,120]        (= #2b FAST, the reference)
  LR 1.2e-4: CKPT [0,3,6,10,18,30,50,90,150,250]
  LR 6e-5 : CKPT [0,5,10,20,40,70,120,200,350,550,800]   (extended to approach convergence)
6 B-tasks, SUB_A=[0,6,12,18,24,30,36,42], NEG 90-99, B_POOL 50-89, 5 cls/B, deployed head w_A frozen at base, seed 0.
Converged (X_conv) = value at the last checkpoint. t90_X = first checkpoint reaching init+0.9·(final−init). Lag = t90_forget − t90_accB
(measured BOTH in steps AND in ‖Δθ‖ traveled between those two checkpoints).

## PRE-REGISTERED PREDICTIONS
2c-P1 — ‖Δθ‖ SUFFICIENCY ACROSS LR (the headline). Converged forgetting is monotone increasing in converged ‖Δθ‖ across all 4 LR
    conditions: Spearman ρ(‖Δθ‖_conv, forget_conv) ≥ +0.80 over the 4×6=24 (condition,B) points. → ‖Δθ‖ remains the governing variable
    ACROSS learning rates (LR matters only via how far it travels). FALSIFIED if ρ < +0.50 or the points scatter badly off a monotone
    curve (two conditions at similar ‖Δθ‖_conv giving very different forget) → the PATH matters beyond ‖Δθ‖, refining #1's law. [Honest
    prior: I EXPECT rough sufficiency (more travel → more forget) but flag a plausible path residual given #2b's hysteresis hint.]
2c-P2 — SAFE-TRAVEL (re-anchored, fixes 2b-P2). Damage-per-unit-travel Δforget/Δ‖Δθ‖ in [0, t90_forget] ≥ 3× that in [t90_forget, end]
    (median over B, LR 3e-4). Anchoring on t90_FORGET (not t90_accB) reveals safe-travel: post-collision weight-travel is ≥3× less
    damaging per unit than collision-phase travel. NULL if ratio < 2 (plateau is saturation, not safe-travel).
2c-P3 — FULL-COLLISION BALLISTIC (extends 2b-P1 past acquisition). Mean consecutive A-centroid drift-cosine over the FULL collision
    window [0, t90_forget] ≥ +0.50 (median over pairs, LR 3e-4). The LAGGED bulk (steps ~8-40, where most forgetting happens) stays
    ballistic, not just the early plunge. FALSIFIED < +0.30.
2c-P4 — THE LAG IS SPATIAL, NOT TEMPORAL (new, elegant-if-true). The acquisition→forgetting lag is more CONSISTENT across LR measured in
    ‖Δθ‖ (weight-travel) than in steps: CV(lag_‖Δθ‖ across 4 LRs) < CV(lag_steps across 4 LRs). → forgetting trails acquisition by a
    roughly fixed amount of WEIGHT-TRAVEL (a ballistic SPATIAL delay in θ-space), which the step-count lag (15 vs 75 in #2b) obscures.
    NULL if the two CVs are comparable (lag has no cleaner unit).

## Outcome grid (locked)
- 2c-P1 holds: #1's ‖Δθ‖ law is the SUFFICIENT cross-LR statistic — LR only sets travel distance. Dynamics reduce to "how far did it go."
- 2c-P1 falsified (path residual): forgetting is NOT a function of ‖Δθ‖ alone — a slower/more-settled path forgets differently at equal
  travel → a genuine refinement of #1 (the SETTLING, not just the distance, damages). Higher-value outcome if clean.
- 2c-P2/P3 confirm: the two-phase picture (ballistic collision → safe plateau) is metric-clean, #2b's flags were artifacts (as claimed).
- 2c-P4 holds: the lag has a natural SPATIAL (‖Δθ‖) unit → "forgetting is a fixed weight-distance behind acquisition" — a crisp law.

## Non-saturation
Higher-LR conditions (3e-4, 6e-4) converge to forget ~0.10 (partial band). Lower LRs forget less; require ≥2 of 4 conditions with
converged forget ≥ 0.03 for a valid ‖Δθ‖-sweep. 6e-5 may not fully converge by 800 steps (‖Δθ‖ still creeping) — flagged; its point is
"near-converged," noted as a caveat, not dropped. d′ not used (AUC-based forget avoids #2-P4's threshold artifact).

## Config (locked)
build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT, wd 1e-4. LRs/checkpoints/tasks as above. Metric = Spearman + the t90 step-vs-‖Δθ‖
lag comparison + damage-ratio. Single seed for the trend. No production system, no network, pure local CIFAR.
