# PREREG — Direction #2: the DYNAMICS of interference. Is the collision decided at t=0, or does it unfold?

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune (plastic backbone), deployed frozen-A-head metric (continuity with #7/#7b). $0 local GPU. Reuses probe_unification_wa
machinery (build_full, deployed head w_A, d′, forget_old) + a checkpointed full-FT.

## The new axis
Every prior probe measured STATIC endpoints (initial d's, final Δθ, final forgetting). But forgetting HAPPENS over training time.
This probe checkpoints A's deployed-head discriminability d′_A(t) and forgetting forget_A(t) THROUGHOUT a B fine-tune (t = 0,25,50,
100,200,350,500 steps), plus B-competence acc_B(t) and the A-feature-centroid trajectory p_A(t). Four questions the statics can't answer:
is the interference direction BALLISTIC (fixed at t=0) or does it ROTATE mid-training? Can final forgetting be predicted from the FIRST
~50 steps? Does the STATIC cosine govern the collision RATE even though it weakly governs the endpoint? Is forgetting monotone or does it
REBOUND?

## Measured per B-task b (N_B=15), per checkpoint t, per A-class a ∈ SUB_A
- forget_A(a,b,t) = AUC(w_A;F_A0,F_neg0) − AUC(w_A;F_A(t),F_neg(t))     deployed-head forgetting CURVE (w_A frozen at base)
- dprimeA(a,b,t)  = discriminability of A vs neg along frozen w_A at step t
- p_A(a,b,t)      = mean A-feature vector (512d) at step t; increment v(t)=p_A(t)−p_A(t−1)
- acc_B(b,t)      = B-task train accuracy at step t
- cosAB(a,b)      = <d_A, d_B> (base space, static)   ;   dtheta(b,t) = ‖θ(t)−θ0‖

## PRE-REGISTERED PREDICTIONS (falsifiable, directional; honest prior stated)
D2-P1 — BALLISTIC-then-CURVED (my genuine prior; accept falsification). The A-centroid drift is approximately STRAIGHT-LINE early:
    mean consecutive-increment cosine cos(v(t),v(t−1)) ≥ +0.60 over the first ~100 steps (t≤100), i.e. the collision DIRECTION is set
    by the early gradient and forgetting accrues by traveling along a roughly fixed direction (consistent with ‖Δθ‖-governed statics).
    FALSIFIED if early consecutive-cosine < 0.40 → the interference direction ROTATES continuously (curved trajectory, a real finding).
D2-P2 — EARLY-WARNING. Who forgets is decided early: ρ(forget_A at t=50, forget_A at t=500) ≥ +0.60 across all (A,B) pairs. Final
    forgetting is predictable from the first 10% of training → cheap early-warning is possible.
D2-P3 — STATIC COSINE GOVERNS THE COLLISION RATE. The −cos law is FRONT-LOADED: cosAB predicts EARLY forgetting at least as well as
    late — ρ(cosAB, forget_A(t=50)) ≤ ρ(cosAB, forget_A(t=500)) (both ≤ 0, early no weaker). Anti-parallel B collides IMMEDIATELY; then
    magnitude ‖Δθ‖ takes over and dilutes the directional signal (explaining why cosAB→final-forget was only −0.18 in #7/#7b — the
    directional signal is spent in the first steps). If early cosAB-effect is clearly stronger than late (Δρ ≤ −0.10), the arc's static
    weak-residual is REVEALED as a strong-but-transient early mechanism. NULL if cosAB is flat across time.
D2-P4 — MONOTONE (descriptive, light). Forgetting is mostly monotone: fraction of (A,B) trajectories where dprimeA REBOUNDS by >0.1
    after its minimum is < 15%. If rebounds are common (>30%), that is itself a finding (B overfits → A partially recovers).

## Outcome grid (locked)
- D2-P1 ballistic + D2-P2 early-warning + D2-P3 front-loaded: interference is a FAST DIRECTIONAL COLLISION set at t=0, then slow
  magnitude drift — a clean two-phase dynamical picture that reconciles the static −cos (early) with ‖Δθ‖-governed endpoints (late).
- D2-P1 falsified (curved): the geometry ROTATES during training → forgetting is a genuinely dynamical process, statics are incomplete.
- D2-P3 flat/null: the cosine does NOT govern the rate → the −cos law is purely a static endpoint property, dynamics are orthogonal.
- D2-P4 rebounds common: forgetting is non-monotone → early-stopping at the d′ trough is an intervention (leads to a #2b).

## Non-saturation (reuse)
forget_old metric IDENTICAL to #7/#7b (already confirmed non-saturated, median ~0.099 at t=500 in partial band [0.05,0.45]). The CURVE
must reach the partial band by t=500 (it does by construction); intermediate checkpoints are strictly smaller → no new saturation risk.

## Config (locked)
build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT, FT_LR=3e-4, checkpoints CKPT=[0,25,50,100,200,350,500], SUB_A=[0,6,12,18,24,
30,36,42], N_B=15, NEG 90-99, B_POOL 50-89, 5 cls/B-task, seed 0. Deployed head w_A frozen at base. Metric = Spearman (Pearson
alongside). Single seed for the trend; 3-seed only if a headline prediction lands borderline. No production system, no network, pure local CIFAR.
