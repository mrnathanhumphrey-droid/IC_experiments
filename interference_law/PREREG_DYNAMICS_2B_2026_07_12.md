# PREREG — Direction #2b: INSIDE the collision. Fine-grained resolution of the one-shot forgetting burst.

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune, deployed frozen-A-head metric (continuity #7/#7b/#2). $0 local. Reuses probe_dynamics machinery + fine checkpoints + ‖Δθ‖(t).

## What #2 couldn't see, and why
#2 (lock 9508ee75) found forgetting is a SINGLE FAST COLLISION co-timed with acquisition (both done by t=25) then a PLATEAU, with strong
early-warning (ρ+0.85). But its coarsest-relevant interval [0,25] already CONTAINED the whole collision, so it could NOT resolve: (a) is
the collision BALLISTIC (straight-line plunge) or CURVED (rotating) DURING the burst? (b) does forgetting track weight-travel ‖Δθ‖(t)
and then DECOUPLE (safe travel), or is the plateau just saturation? (c) is the "one-shot" intrinsic, or an artifact of a fast LR that
learns B in <25 steps? This probe checkpoints INSIDE [0,25] at fine resolution and adds a SLOW-LR condition to de-confound.

## Design: two conditions, fine checkpoints matched to each timescale
- FAST (LR 3e-4, as #2):  CKPT_F = [0,1,2,3,5,8,12,18,25,40,70,120]
- SLOW (LR 6e-5, LR/5):   CKPT_S = [0,5,10,20,40,70,120,200,350,500]
8 B-tasks, SUB_A=[0,6,12,18,24,30,36,42], NEG 90-99, B_POOL 50-89, 5 cls/B, deployed head w_A frozen at base, seed 0.
Per checkpoint t, per (a,b): forget_A(t), d′_A(t), A-centroid p_A(t); per (b,t): acc_B(t), ‖Δθ‖(t)=‖θ(t)−θ0‖.
Define t90_X = first checkpoint where X reaches initial + 0.9·(final−initial). Acquisition window = checkpoints with t ≤ t90_accB.

## PRE-REGISTERED PREDICTIONS (falsifiable; honest prior stated where I have one)
2b-P1 — BALLISTIC DURING THE COLLISION (my prior; genuinely uncertain, accept falsification). Over the acquisition window (t ≤ t90_accB),
    mean consecutive A-centroid drift-cosine cos(v(t),v(t−1)) ≥ +0.50 (FAST): the collision is a roughly straight-line plunge whose
    direction is set by the early gradient. FALSIFIED if < +0.30 → the drift ROTATES even during the burst (curved collision).
2b-P2 — SAFE TRAVEL (damage decouples from travel after acquisition). Damage-per-unit-travel Δforget/Δ‖Δθ‖ in the ACQUISITION window is
    ≥ 3× that in the POST-acquisition window (FAST). i.e. early weight-travel is ~3×+ more damaging per unit than late travel → the
    plateau is NOT saturation, it is SAFE TRAVEL (the backbone keeps moving but stops damaging A). [Cross-checks the LLM-arc Probe-7
    finding that warm-up cut damage-per-unit ~÷3 — same "damage/unit is not constant" phenomenon, here across the training trajectory.]
2b-P3 — COLLISION IS ACQUISITION-LOCKED (de-confound). Forgetting tracks acquisition in time: within each condition, |t90_forget −
    t90_accB| ≤ 2 checkpoint-slots (they rise together); AND the SLOW condition shifts BOTH later in step-count (t90_accB and t90_forget
    both larger than FAST). → the collision is locked to WHEN B is learned, not to an absolute step count. NULL if forget's t90 is
    unrelated to accB's t90 (forgetting has its own clock).
2b-P4 — ONE-SHOT IS INTRINSIC, NOT AN LR ARTIFACT. SLOW spreads the collision out in steps (t90_forget(SLOW) ≥ 2× t90_forget(FAST) in
    absolute steps) while final forgetting stays in the SAME BASIN (|forget_final(SLOW) − forget_final(FAST)| ≤ 0.05, or SLOW lower).
    → the burst is time-dilated by learning speed, not created by it; forgetting magnitude is set by the task pair, not the LR.

## Outcome grid (locked)
- 2b-P1 ballistic + P2 safe-travel + P3 acquisition-locked + P4 intrinsic: CLEAN TWO-PHASE LAW — a straight-line collision locked to
  acquisition sets the (LR-invariant) damage; subsequent travel is safe. The dynamical completion of the arc's static ‖Δθ‖ law.
- 2b-P1 falsified (curved): the collision ROTATES → interference direction is dynamical even within the burst (statics incomplete).
- 2b-P2 falsified (damage/travel flat): the plateau is SATURATION not safe-travel → "safe travel" story dies; forgetting just runs out
  of A to destroy at this task difficulty.
- 2b-P3/P4 null: the one-shot is an LR/absolute-step artifact → generalization of #2's picture is limited (report as scope caveat).

## Non-saturation
FAST final forget ~0.10 (from #2, partial band). SLOW may forget LESS; require SLOW final forget ≥ 0.03 (measurable) — if SLOW ≈ 0
(too gentle to interfere) P4's basin-comparison is moot and reported as such. d′~4 (base AUC≈1.0) noted; t90 uses forget (AUC-based),
not the noisy d′, to avoid #2-P4's threshold artifact.

## Config (locked)
build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT, wd 1e-4. Conditions/checkpoints/tasks as above. Metric = Spearman + the t90
step-locking comparison. Single seed for the trend. No production system, no network, pure local CIFAR.
