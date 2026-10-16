# PREREG — Direction B: the frozen<->plastic phase transition, LoRA rank as ORDER PARAMETER

Author: arc (CF interference). Date: 2026-07-14. Status: PRE-REGISTERED, SHA256-lock before run. CIFAR-100 + ResNet18 conv-LoRA
(layer3+layer4), reuses probe_representational_law machinery. Lambda GPU.

## The question the arc deferred
The dichotomy has two mapped endpoints: frozen linear readout -> forget ~ -cos (rho -0.74, DIRECTION-governed, SS2); LoRA-rank-32
backbone -> rho -0.12, cosine dead, DISRUPTIVENESS-governed (SS6/Thread-6). The RANK SWEEP between them was explicitly NOT run
(VERDICT_REPRESENTATIONAL_LAW line 43: "Not run ... the rank in {4,8,16} sweep"). Direction B maps the crossover: is the frozen->plastic
transition a smooth decay or a sharp threshold in the order parameter (LoRA rank = plasticity)?

## Design
Per rank r in {1,2,4,8,16,32}: for each of N_B fixed B-tasks, fresh LoRA-fine-tune on B, re-probe ALL 50 A capabilities
(frozen-re-probe forgetting = base linear-A-probe AUC minus post-FT AUC). B-tasks and base directions held FIXED across ranks (fair
comparison). Two governance axes per rank:
- DIRECTION: rho(cos(d_A,d_B), forget) pooled + mean WITHIN-B partial rho (alignment signal controlling for which-B).
- DISRUPTIVENESS/MAGNITUDE: eta2_B (fraction of forget variance explained by B-identity) + rho(per-B feature-drift, per-B median forget).

## PRE-REGISTERED PREDICTIONS
B-P1 — DIRECTION DECAYS. |rho(cos,forget)| decreases monotonically with rank, from near the frozen -0.74 at low rank toward ~-0.12 at
    rank 32. [Prior: monotone decay.] Report the curve; no hard pass/fail on monotonicity (measurement noise allowed).
B-P2 — DISRUPTIVENESS RISES. eta2_B (which-B main effect) increases with rank; per-B drift-vs-forget rho becomes more positive. The two
    axes CROSS: at low rank direction dominates (|rho_cos| > eta2_B-driven disruptiveness), at high rank disruptiveness dominates.
B-P3 — CROSSOVER CHARACTER. Identify the rank where direction governance falls below disruptiveness governance. Pre-registered as
    descriptive: is it a smooth crossover (gradual over the rank decade) or a sharp threshold (one rank does most of the flip)? Either is a
    valid result; the claim is the SHAPE of the transition, not a specific critical rank.

## Outcome grid (locked)
- Direction decays AND disruptiveness rises with a visible crossover -> the dichotomy is a CONTINUOUS phase diagram in plasticity; the
  governing variable transitions from -cos to disruptiveness as an order-parameter function. The headline.
- Direction stays strong across all ranks -> the rank-32 -0.12 was a high-rank-specific collapse, not a gradual transition (surprise;
  would mean plasticity must exceed a high threshold before cosine dies).
- Direction already dead at rank 1 -> even minimal plasticity kills the cosine law; the transition is at the frozen/plastic boundary
  itself (any backbone motion decouples it). Also a clean result.

## Non-saturation / headroom (the recurring trap)
Require per-rank median forget in [0.05, 0.40]. Low rank may FLOOR (too little plasticity to forget) and high rank may SATURATE; ranks
outside the band are flagged and EXCLUDED from the governance comparison (only measurable ranks map the crossover). If most ranks floor,
raise B_STEPS/B_LR; if most saturate, lower them. B_STEPS=1000, B_LR=4e-3 per the Thread-6 gate recipe (rank32 gave median ~0.12).

## Config (locked)
resnet18 IMAGENET1K_V1, conv-LoRA on layer3+layer4 conv1/conv2; B-task = 5-way over reserved classes 50-89; A = 50 held-in classes vs
neg pool 90-99; RESIZE 128; PER_CLASS 100; N_B=8 fixed B-tasks; B_STEPS=1000/B_LR=4e-3; ranks {1,2,4,8,16,32}; frozen readout reference
rho=-0.74 (SS2). Metrics = per-rank {rho_cos pooled + within-B, eta2_B, rho(drift,forget), median forget}. probe_phase_rank.py on Lambda a100.
