# VERDICT — NEW-1b: the asymmetry is a TRANSITIVE PECKING ORDER, not a cycle (PREREG 0f75676e, locked 2026-07-13). Landed, $0 local 1731s, resnet18, CIFAR + DTD, split-half.

## RESULT: CLEAN POSITIVE. Split-half reproducibility settles it identically on both datasets: the net-DOMINANCE potential REPRODUCES (ρ 0.51/0.59) → capabilities have a stable predator/prey identity; the CYCLIC residual does NOT reproduce (ρ −0.01/+0.07) → the ~54% "cyclic residual" from NEW-1 is measurement NOISE, not real rock-paper-scissors. The interference asymmetry is a STABLE, GLOBALLY-ORDERABLE, TRANSITIVE dominance ranking. No non-transitive cycles.

## The numbers (Helmholtz-Hodge decomposition K = gradient(pecking) + curl(cyclic); split-half A vs B on disjoint image halves)
  dataset  medFgt(A/B)  pecking%  cyclic%   ρ(F)   ρ(dominance h)   ρ(curl)    → verdict
  CIFAR    0.21/0.23     ~44%      ~56%     +0.18     +0.512        −0.010     dominance REAL, cycle NOISE
  DTD      0.24/0.23     ~58%      ~49%     +0.31     +0.594        +0.068     dominance REAL, cycle NOISE
  1b-P2 dominance ρ≥.50: PASS both. 1b-P3 curl ρ (≥.30 real / <.15 noise): NOISE both. 1b-P4 dataset-general: YES (identical answer).

## Reading it
- **Dominance is real and reproducible (1b-P2 ✓).** The net-dominance potential h_i = mean_j K_ij = (fragility_i − disrupt_i)/2 correlates
  ρ 0.51 (CIFAR) / 0.59 (DTD) across INDEPENDENT image halves → a capability's predator/prey identity is a stable property, not sampling
  luck. Averaging over a row/col beats the per-entry noise (note ρ(F)≈0.18–0.31 for individual entries, but ρ(dominance)≈0.5–0.6 — the
  aggregate is far more reproducible).
- **The cycle is NOISE (1b-P3, decisive).** curl ρ = −0.01 / +0.07 (both < 0.15) → the cyclic residual is DIFFERENT every resample = noise.
  The ~54% apparent cyclicity (NEW-1 rank-1 fit residual) is a measurement artifact; there is NO reproducible non-transitive structure.
  curl is in fact the LEAST reproducible component (ρ_curl ≈ 0 < ρ_F 0.18–0.31 < ρ_dominance 0.5–0.6) — exactly what you expect if curl is
  what's left after removing the reproducible main effects.
- **The interference is GLOBALLY ORDERABLE.** A transitive pecking order (dominance potential h) captures the reproducible asymmetry;
  capabilities can be linearly ranked from net-predator to net-prey. No rock-paper-scissors.

## The convergence (this closes the structure story cleanly)
The reproducible dominance h_i = (fragility_i − disrupt_i)/2 is the DIFFERENCE of the two durable per-capability main effects (fragility ≈
margin/d′; disruptiveness ≈ ‖Δθ‖). So the ENTIRE reproducible structure of the interference matrix is TWO per-capability scalars:
  • FRAGILITY_i  (how easily i is damaged ≈ base margin/d′)          → drives the row of F
  • DISRUPT_i    (how much i damages others ≈ ‖Δθ‖ it induces)       → drives the column of F
  their SUM-like combination sets overall forgetting; their DIFFERENCE is a transitive net-dominance pecking order.
Everything PER-PAIR beyond these two scalars is NOISE: the interaction has no geometric descriptor (1a''''-d) and no low-rank/completable
structure (NEW-2), and now the antisymmetric cyclic residual does not reproduce (this). Interference reduces to 2 numbers per capability.

## The final structural picture of the interference matrix (fully bounded, positive + null)
  REAL & REPRODUCIBLE:  fragility_i, disrupt_i (2 per-capability scalars) → forgetting magnitude + a TRANSITIVE dominance order (h).
  NOISE / STRUCTURELESS: the per-pair interaction (no descriptor, not low-rank, not completable) and the cyclic residual (doesn't reproduce).
  → F[i,j] ≈ μ + fragility_i + disrupt_j + noise. The only directional content that survives is a globally-orderable pecking order; there
    is no reproducible per-pair or cyclic geometry.

## Caveats
Single backbone seed (split-half is the noise control); 16×16; 50 img/class per half (noisier F — but the design USES that: dominance
rises above the noise, curl does not). Hodge fractions approximate (row-mean potential ⇒ slight non-orthogonality, sum ~1.06; immaterial —
the reproducibility, not the fraction, is the finding). GTSRB excluded (saturated). Predator/prey class lists are within-dataset,
dataset-specific.

## Bottom line
The interference asymmetry is a STABLE TRANSITIVE PECKING ORDER (net-dominance = fragility − disruptiveness), globally orderable, with NO
real cycles. Combined with the descriptor + low-rank nulls, the interference matrix is now fully characterized: F ≈ μ + fragility(victim) +
disruptiveness(perturber) + noise — two reproducible per-capability scalars and nothing else structured. A clean, positive close to the
structure thread, and the answer to "what is even orderable": a 1-D dominance axis. Ready for NEW-3 (transfer) with the structure known.
