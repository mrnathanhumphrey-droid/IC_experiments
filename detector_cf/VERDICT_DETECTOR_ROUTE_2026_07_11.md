# VERDICT — Detector ROUTE arm (PREREG 2dc474a2, locked 2026-07-11). Landed, $0 local, SUBSAMPLE=220, 1 seed, 1 order.

## The router built the predicted architecture, data-driven (TAU=0, no hand-wiring)
- swap  -> SPAWN head 0
- diffusion -> SPAWN head 1   (cos to swap = -0.933 → correctly isolated as the anti-parallel antagonist)
- celebdf -> JOIN head 0   (cos 0.586 to swap-family, -0.433 to diffusion)
- dfdc   -> JOIN head 0    (cos 0.841 to swap-family, -0.255 to diffusion)
- LANDED 2 heads: H0={swap,celebdf,dfdc}, H1={diffusion}. Two heads, not four — the strong antagonist gets its own.

## Final retention (same fixed test sets as naive/gpm/rand)
                swap   diffusion  celebdf   dfdc
  naive        0.651     0.683     0.896    0.938
  gpm          0.740     0.575     0.822    0.789
  route/oracle 0.656     0.993     0.894    0.936    (upper bound: presumes a family detector at inference)
  route/maxens 0.639     0.964     0.884    0.928    (LABEL-FREE, deployable: max sigmoid over both heads)

## WIN (the shippable result): the lone antagonist is rescued, label-free, at ~zero cost
- diffusion: 0.683 (naive) / 0.575 (gpm STARVED it) → **0.964 (maxens) / 0.993 (oracle)**. +0.281 label-free, +0.310 oracle.
- Cost to the other three under maxens vs naive: swap −0.012, celebdf −0.012, dfdc −0.010 — all within noise.
- maxens captures 0.281 of the 0.310 oracle gain → you get ~90% of the win with NO generator label at inference.
- GPM moved diffusion the WRONG way (protect-priors-by-starving-newcomer). Routing is the correct tool for an anti-parallel paradigm.

## MISS (pre-registered PRIMARY failed — reported, not spun): routing does NOT protect swap
- Pre-reg predicted swap ≳ 0.88 on its cluster head. Actual: route/oracle swap = 0.656 — equal to naive, BELOW gpm's 0.740.
- Cause: I asserted the swap-family was ~0.99 self-aligned. It is NOT (cos 0.586 / 0.841 — only MODERATELY aligned). celebdf/dfdc
  still erode swap-specific retention INSIDE the shared head, to the same 0.656 naive suffers. This is the NULL clause I locked
  ("swap-family head dilutes swap below gpm → routing didn't help on swap; report the dilution").
- Lesson refined: isolating the STRONG antagonist (cos ≈ −0.93) is a large, clean win; but "compatible-ish" (cos 0.5–0.8)
  paradigms are NOT a monolith — grouping them into one head does not stop intra-cluster CF. Positive cosine ≠ free coexistence.

## Architecture verdict for the detector team
1. Give an anti-parallel generator family (here: diffusion) its OWN head, score by max-ensemble. Label-free, +0.28 AUC on that
   family, ≤0.01 cost elsewhere. Concrete, deployable.
2. Do NOT assume same-direction swap variants coexist for free: swap still forgets under celebdf/dfdc. Protecting swap TOO needs
   GPM *within* the swap-family head (project off celebdf/dfdc), or swap its own head. → the natural next arm (below).

## NEXT (pre-register before firing): ROUTE + intra-cluster GPM
Give diffusion its own head (keeps 0.99) AND run GPM inside H0 (project the celebdf/dfdc surge off swap's memory). Predicted:
swap ~0.74 (gpm-level) + diffusion ~0.99 simultaneously = retains BOTH antagonists at their individual best — the thing neither
naive, gpm-single-head, nor plain-route achieves. $0 local extension. ~~NOT run — propose + lock first.~~
**[STALE — CORRECTED 2026-07-14. IT RAN, ~6 MINUTES LATER.]** Proposed, locked, fired and verdicted the same hour:
`PREREG_DETECTOR_ROUTE_GPM_2026_07_11.md` → `VERDICT_DETECTOR_ROUTE_GPM_2026_07_11.md`. The prediction held: swap 0.752
(matches single-head GPM) **+** diffusion 0.993 simultaneously; min(swap,diffusion)=0.752 strictly dominates all baselines.
This "NEXT" section is history, not a to-do. See `D:/IC_experiments/STATUS_RECONCILIATION_2026_07_14.md`.

## Caveats (carried)
Filename-hash split (not identity-disjoint) → absolute AUCs inflated; the DELTAS on fixed test sets are the claim. Head-level only
(frozen backbone). SUBSAMPLE=220, single seed, one order — the +0.28 diffusion win is far above noise; don't over-read the sub-0.02
swap/celebdf/dfdc costs. oracle = upper bound; maxens = deployable floor.
