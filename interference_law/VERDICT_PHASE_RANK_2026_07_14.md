# VERDICT — Direction B: the frozen<->plastic PHASE TRANSITION (PREREG cba92440, locked 2026-07-14). Landed, Lambda a100 (~$6.0 across 3 runs incl. 2 false starts), CIFAR-100 / ResNet18 conv-LoRA layer3+4.

## RESULT: the frozen->plastic transition is a SHARP BOUNDARY, not a smooth crossover. The cosine DIRECTION law does not decay gradually with plasticity — it COLLAPSES at the frozen/plastic boundary itself (dead by rank-2 LoRA) and stays dead through rank 32. Pre-registered outcome 3.

## The phase diagram (order parameter = LoRA rank; frozen readout reference rho(cos,forget) = -0.74, SS2)
  rank   |rho_cos|   within-B_rho   eta2_B   rho_drift   med_forget   inband
  (froz)   0.740       -0.740         --        --          --        (ref)
     1     0.052       -0.056       0.390     +0.257       0.014      FLOORED
     2     0.125       -0.068       0.828     +0.943       0.066      yes
     4     0.062       -0.015       0.386     +0.943       0.096      yes
     8     0.029       -0.130       0.295     -0.086       0.110      yes
    16     0.170       -0.212       0.035     +0.086       0.166      yes
    32     0.202       -0.200       0.027     +0.029       0.158      yes
  (primary run: B_STEPS=1000, B_LR=8e-3, N_B=6 — `out/phase_rank_r1000_lr8e3_PRIMARY.json`)

## Reading it
- **B-P1 FALSE as stated (monotone decay): there is NO gradual decay.** At the lowest RELIABLE plastic rank (rank 2, partial forgetting 0.066)
  the clean within-B alignment signal is already **-0.068** vs the frozen **-0.74**. Rank 1 is floored (med 0.014) so its numbers are unreliable,
  but by rank 2 the law is gone. It stays gone through rank 32 (within-B -0.13..-0.21).
- **B-P3 answered: SHARP THRESHOLD at the boundary**, not a crossover over the rank decade. **Any** backbone motion — even a rank-2 LoRA —
  evaporates direction-governance. The cosine law is a razor-thin property of EXACT frozen-ness: it lives at the interface between a FIXED
  representation and a LINEAR decision. This sharpens SS8's "readout-specific" to "DISCONTINUOUS at the frozen boundary."
- Robustness: independent support from Thread-6 (rank 32 at gentler LR 4e-3 -> rho -0.12) and from the deconfound run below. The direction
  collapse is NOT an LR artifact — it holds at LR 4e-3, 6e-3 and 8e-3.
- **Plastic regime is DISRUPTIVENESS-governed**: per-B feature-drift -> per-B forgetting rho ~ +0.7..+0.9 at low-mid rank.

## B-P2 (disruptiveness vs rank): NOT ISOLABLE — and we established WHY (this is the honest sub-result)
Two follow-ups were run to try to isolate the eta2_B-vs-rank trend. Both failed, in instructive ways:
1. **Gentle-LR deconfound (4e-3/1500, `out/phase_rank.json`)**: CORRECTED rank32 eta2_B from a saturated 0.03 to a true **0.26** (confirming the
   primary's near-zero high-rank values were the B_LR=8e-3 headroom bump saturating them). BUT it FLOORED the low ranks (r2/r8 med 0.017/0.025).
2. **Matched-forgetting (`probe_phase_matched.py`, LR 6e-3, per-rank step calibration to forget~0.10, `out/phase_matched.json`)**:
     rank   steps   med_fgt   eta2_B   within-B   rho_drift
       2     2600    0.019    0.440    -0.022     +0.833      <- CAPACITY-CAPPED
       8      692    0.107    0.188    -0.138     +0.762      <- on target
      32      493    0.143    0.055    -0.172     -0.286      <- overshot
   Three obstructions, each a finding: (i) **rank 2 is CAPACITY-CAPPED** — 0.019 forget even at 2600 steps; it physically cannot reach the 0.10
   target, so it cannot be matched. (ii) **rank and forget-magnitude are ENTANGLED** — in all four runs rank 32 forgot MORE than rank 8 at any
   step budget (even 493 steps -> 0.143); it was never brought to an equal-or-lower level. (iii) **eta2_B is CONDITION-SENSITIVE** — rank 32 came
   out 0.03 / 0.26 / 0.055 across the three LR settings.
**Conclusion:** eta2_B is always lower at rank 32 than rank 8, but rank 32 always forgot more — so "falls with rank" is inseparable from "forgot
more." A clean eta2_B(rank) law does not exist to be measured with this design. NOT CLAIMED. The boundary-collapse headline does not depend on it.
One robust extra: drift->forget coupling **decouples at rank 32** (rho_drift ~ -0.3) — per-B drift stops predicting per-B forgetting at high plasticity.

## Caveats
Single seed per rank; CIFAR/ResNet18 only; rank 1 floored (excluded from the direction read; rank 2 carries the low end). Base accA0 ~0.984
(near-ceiling separability, as SS1/Thread-6) — forgetting measured from a high base. Non-saturation flags reported per rank; only in-band ranks
carry the reading.

## Ops lessons banked
- The `out/` dir must be created on a fresh Lambda box (first run crashed writing results after completing rank 1).
- A LOCAL launcher death (exit 127) means the `finally` never fires -> ORPHAN instance cost-bleed (hit once, caught via API list + manual
  terminate). FIX (now in `cloud/lambda_launch_phase_rank.py`): run the probe **DETACHED** (nohup setsid, survives local death) + **POLL** for a
  DONE marker + a **DEAD-MAN self-terminate** on the instance (sleep N; selfterm.py via the Lambda API). Reuse for any >30min run.

## Bottom line
The frozen<->plastic dichotomy is a **discontinuity, not a gradient**. Direction-governance is exclusive to the exactly-frozen readout and dies at
the first increment of plasticity; the plastic regime is disruptiveness-governed, and how disruptiveness itself scales with plasticity is not
cleanly measurable because rank and forgetting-magnitude cannot be disentangled (low rank capacity-caps, high rank overshoots).
Written up: `COSINE_GEOMETRY_OF_INTERFERENCE_2026_07_11.md` SS17.
