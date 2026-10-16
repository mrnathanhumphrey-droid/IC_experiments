# PREREG — NEW-1: is the interference matrix ASYMMETRIC? A hidden fragility axis + the dominance structure of capabilities.

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune, deployed frozen-head metric. $0 local. Reuses probe_gradient_alignment / probe_representational_law machinery.

## The crack
The cosine law is SYMMETRIC by construction: cos(d_i,d_j)=cos(d_j,d_i) ⇒ it PREDICTS forget(i←j)=forget(j←i). We have never measured
the DIRECTED matrix. If forgetting is asymmetric (some capabilities are fragile victims, others are bullies), the cosine law is provably
INCOMPLETE and there is a second, per-capability axis (FRAGILITY) we have missed. This probe measures the directed matrix, tests its
symmetry, isolates fragility as a new variable, and characterizes the DOMINANCE structure (transitive pecking-order vs non-transitive cycle).

## Design (square directed matrix over ONE capability set)
CAPS = [0,7,14,21,28,35,42,49,56,63,70,77] (12 spread CIFAR-100 classes). NEG = [90..99] shared. Each capability plays BOTH roles:
- as VICTIM i: deployed binary head w_i (class_i vs neg), direction d_i, base AUC_i — FROZEN.
- as PERTURBER j: full-FT the backbone on the binary task (class_j vs neg), strength gated to partial forgetting.
F[i,j] = AUC_i(base) − AUC_i(after full-FT on j), for i≠j (diagonal undefined). Also record ‖Δθ_j‖ and cos_ij = d_i·d_j.
Decompose F = S + K, S=(F+Fᵀ)/2 symmetric, K=(F−Fᵀ)/2 antisymmetric (off-diagonal). fragile_i = mean_j F[i,j] (row), disrupt_j =
mean_i F[i,j] (col). Note (algebra): the rank-1 potential fit h_i = mean_j K[i,j] = (fragile_i − disrupt_i)/2 — the dominance potential
IS "how much more you're damaged than you damage."

## PRE-REGISTERED PREDICTIONS
New1-P1 — ASYMMETRY EXISTS. ‖K‖_F / ‖S‖_F ≥ 0.15 AND ρ(F[i,j], F[j,i]) < 0.80 across directed pairs. → the cosine (symmetric) law is
    INCOMPLETE; forgetting has a directional component. [Prior: I expect real asymmetry — disruptiveness (‖Δθ‖, per-perturber) and
    fragility (per-victim) are different mechanisms; a strong perturber need not be a fragile victim.] FALSIFIED if ‖K‖/‖S‖ < 0.10 &
    ρ ≥ 0.85 → matrix ~symmetric → cosine law is COMPLETE at this level (also a clean, publishable result).
New1-P2 — FRAGILITY IS A REAL SECOND AXIS. In the additive model forget_ij = μ + row_i + col_j + s·cos_ij fit by least squares, adding
    the ROW (fragility_i) term over a [μ + col_j + s·cos_ij] baseline raises R² by ΔR² ≥ 0.10 — fragility explains forgetting-variance
    that geometry + perturber-strength cannot. AND fragile_i is NOT strongly tied to disrupt_i: |ρ(fragile_i, disrupt_i)| < 0.50 (being
    a bully ≠ being fragile). If both hold → a new factorized law forget ≈ g(cos)·(fragility of victim)·(disruptiveness of perturber).
New1-P3 — DISRUPTIVENESS REPLICATES (continuity/sanity). disrupt_j (column mean) correlates with ‖Δθ_j‖: ρ ≥ +0.50 — the ‖Δθ‖-king (#1)
    reappears as the column structure of this matrix.
New1-P4 — DOMINANCE STRUCTURE: hierarchy vs cycle. The rank-1 potential fit K[i,j] ≈ h_i − h_j explains ≥ 60% of ‖K‖²_F → interference
    has a TRANSITIVE pecking order (a scalar dominance potential h; predators vs prey). FALSIFIED if < 40% → NON-TRANSITIVE / CYCLIC
    interference (capability rock-paper-scissors: i beats j beats k beats i) — the more surprising, higher-value outcome. [Genuinely
    uncertain; either is a real finding.]

## Outcome grid (locked)
- P1 asym + P2 fragility + P4 hierarchy: NEW FACTORIZED LAW — forgetting = symmetric geometry (cosine) × victim-fragility × perturber-
  disruptiveness, with a transitive dominance order over capabilities. The cosine law becomes one of three factors. Top outcome.
- P1 symmetric: cosine law is COMPLETE (fragility/disruptiveness collapse to one per-capability scalar) — clean, tightens the arc.
- P4 cyclic (<40%): NON-TRANSITIVE interference — capabilities have no global pecking order; a genuinely novel structural finding worth
  its own thread (why would continual-learning damage be rock-paper-scissors?).

## Non-saturation
Binary(class_j vs neg) full-FT may be weaker than #1's 5-class perturbation (base already separates class_j from neg). GATE first (4
perturbers): require median off-diagonal F in partial band [0.03, 0.45]. If forget ~0 (perturbation too weak to move a well-separated
task) ESCALATE: perturber = class_j vs the other 11 CAPS pooled (harder, richer negative) — retune before the full 12×12. Do not read
asymmetry off a saturated or dead matrix.

## Config (locked)
build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT, wd 1e-4. Gate FT_LR/FT_STEPS to the partial band (start 5e-4 / 300). 12 caps,
NEG 90-99, PER_CLASS 100, seed 0. Metric = Frobenius ratios + Spearman + least-squares ANOVA + rank-1 potential fit. Single seed for the
structure (the matrix STRUCTURE is the claim, not a 3rd-decimal). No production system, no network, pure local CIFAR.
