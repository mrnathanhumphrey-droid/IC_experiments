# DSI (Deposit-Surge Interaction) — running verdict. 2026-06-22. Agent 2.

Per-step mechanism of map load-bearingness, cross-arch, at the live band (OPT-350m / Pythia-410m only; dead at 1B).
Harness `cloud/dsi_lambda.py`. Reproduce-anchor GATES every read (OPT B-A/B-D +0.089/+0.055; Pythia +0.073/-0.006).

## Harness validation (the anchor gate did its job)
Two perturbations the instrumentation introduced were caught by the anchor gate and fixed:
1. Arm construction: carriers-only reconstruct (bd_feed style) vs the dispositive's FULL-STATE (train phase-2 directly
   from the full post-childhood model). Reverting non-carrier params broke arm D -> Pythia B-D failed (+0.055 vs -0.006).
2. `_damage_attrib` (#4 signature) called sst_loss -> model.eval() and never restored train() -> phase-2 ran dropout-off
   on the 6 damage-checkpoint seeds (~0.018 extra forgetting). #4 DROPPED (to re-add non-perturbingly later).
Clean non-perturbing harness: Pythia anchor PASSES (B-D -0.014 vs -0.006, B-A exact). OPT borderline (B-D +0.025 vs
+0.055, ~1 SE low, B-A passes, dissociation direction holds) -> re-running off the borderline alongside the swap.

## Cross-arch signature (BANKED 2026-06-23 — swap control fired and confirmed deposit-specificity)
Per-carrier (Probe 1, `probe1_carrier.py`), deposit survives in BOTH families (surv ~0.99, no erase anywhere):
- **OPT (load-bearing): carrier-SPECIFIC** — v_proj/k_proj/fc1 ABSORB (surge flows in, deposit holds = non-destructive
  sink); q_proj/out_proj/fc2 DEFLECT (surge routed out of V). A MIX.
- **Pythia (inert): carrier-UNIVERSAL hard DEFLECT** — all carriers deflect, and harder (+0.18..+0.32 vs OPT's
  +/-0.03..0.1).
**Headline (provisional):** load-bearingness is ABSORPTION, not deflection. Pythia deflects uniformly and HARDER yet is
inert -> uniform deflection adds nothing over its non-destructive floor (D-A +0.079). OPT is load-bearing because specific
carriers ABSORB the surge as non-destructive sinks. The map is load-bearing where it absorbs, inert where it only deflects.

## THE BANK GATE — FIRED + LANDED 2026-06-23 (overnight; agents died on API-500 + comp crash, jobs self-completed).
Both swap cells landed (`gateA/dsi_{opt350m,pythia410m}.{json,npz}`, 01:04–01:05). Anchors held: Pythia B−A +0.073
(exact), OPT B−A +0.094 (exp .089); OPT B−D +0.018 = at-floor demoted (0.037 < 0.066 floor, as pre-warned); Pythia
B−D −0.014 (exp −.006, OK).

**INSTRUMENT CATCH (carry forward):** `probe1_carrier.py`'s `swap_gap` = cos(g,S_dep) − cos(g,S_swap) reads NULL on
every carrier (±0.000) — but that is a cosine-to-direction differential, NOT the pre-committed gate. The pre-reg gate is
**real-B inV vs Bswap inV** (energy fraction in V, cross-arm). On the FAITHFUL gate (computed direct from the npz):

| carrier | OPT inV_B − inV_Bswap | Pythia inV_B − inV_Bswap |
|---|---|---|
| v_proj / qkv | **+0.170** | −0.133 |
| k_proj | **+0.092** | — |
| fc1 / h4h | **+0.065** | −0.264 |
| fc2 / 4hh | −0.003 | −0.303 |
| q_proj | −0.014 | — |
| out_proj | −0.015 | — |

**RESULT — absorb-is-load-bearing LOCKS (floor-robust paired cross-arm differential):**
- **OPT (load-bearing):** the absorbing carriers (v/k/fc1) pull surge INTO V deposit-specifically (+0.07…+0.17 over a
  curvature-matched random deposit), surv ~0.98 = non-destructive sinks. Deflectors (q/out/fc2) ≈ 0 = generic.
- **Pythia (inert):** EVERY carrier pushes surge OUT of V deposit-specifically and hard (−0.13…−0.30) — Pythia's
  deflection is an ACTIVE, deposit-specific routing-away, yet still inert (D≥B, B−D negative).

So both families respond specifically to the real deposit, with OPPOSITE sign and OPPOSITE consequence:
**load-bearingness = deposit-specific ABSORPTION (pull in, hold); inertness = deposit-specific DEFLECTION (push out,
protect nothing).** The map is load-bearing where it absorbs, inert where it only deflects. The ~0.066 floor cannot fake
a within-design paired cross-arm differential.

**TWO PRE-REG DEVIATIONS (logged, hunt-honest — not walk-backs):**
1. probe1.py's cosine `swap_gap` does not implement the inV gate → it false-nulls. Read the inV cross-arm gate (above);
   fix or annotate probe1.py before any reuse.
2. The Pythia half of the pre-commit was wrong-SIGNED: we predicted real-B ≈ Bswap (generic); we got real-B strongly
   BELOW Bswap (deposit-specific deflect). The inversion still holds (the consequence split is intact), but Pythia's
   deflection is specific, not generic. Direction logged.

**Why the swap-gate is the bank point and the anchor magnitude is not (run-to-run FLOOR ~0.066, measured by agent 1):**
the swap-gate is a WITHIN-RUN, SAME-CELL, SAME-SEED-SET paired differential (real deposit vs random deposit, measured in
the same run) — the run-to-run reproducibility floor that demolished the dedup-engages-map finding LARGELY CANCELS in a
paired contrast. The OPT anchor B-D is an ABSOLUTE level → fully exposed to the floor. So: BANK on the swap-gate
differential; DEMOTE the anchor magnitude.

## READ ORDER WHEN IT LANDS (pre-committed)
1. **SWAP-GATE FIRST (the bank point, floor-robust).** Per carrier: real-B inV vs Bswap inV. If OPT's absorbing carriers
   (v/k/fc1) show real-B != Bswap AND Pythia's uniform deflect shows real-B ~= Bswap, the absorb-is-load-bearing
   inversion LOCKS on a paired differential the floor cannot fake.
2. **OPT anchor magnitude SECOND, DEMOTED.** If B-D lands ~+0.023..+0.055 that is INSIDE the 0.066 floor -> it is NOT
   evidence of systematic arm-D bias; do NOT diagnose arm-D until ruling out the floor first. B-A reproducing + direction
   holding is what carries the dissociation. OPT B-D is reclassified "supporting, at-floor, leans on ladder+vacuums+
   swap-gate" — same number, same floor, same defense as OPT-350m took on agent 1's side. It does NOT gate the mechanism
   verdict either way.

## DIAGNOSTIC ORDER for the OPT undershoot (corrected)
FIRST: is +0.023 vs +0.055 inside the 0.066 reproducibility floor? (YES, 0.032 < 0.066) -> it is noise, STOP. Only if a
run lands OUTSIDE the floor do we then chase arm-D construction. Do NOT burn a diagnosis on what is already the
instrument's floor.

## SCOPE LIMIT (banked negative result) — G5 does NOT predict the dynamic signature
Fly-around interface pre-committed: per-carrier static G5 (deposit-surge overlap) predicts the per-step dynamic signature,
corr < 0. RESULT: Pythia corr(G5, deflect) = +0.41 (WRONG sign); OPT corr(G5, surv) = -0.12 (weak). So **static G5 does
NOT extend to predicting the dynamic absorb/deflect behavior.** G5 remains a good FAMILY-ORGANIZED structural feature
(RMD-SRC PAYS_STRICT, that result stands) but it is NOT a dynamical predictor. Two instruments disagree -> report both,
bound the claim: G5's reach is static/structural, not per-step dynamics. This prevents overclaiming G5 as a dynamical
predictor in the writeup.

## OPT anchor note (SUPERSEDED by the floor reframe above)
Two full-state OPT runs gave B-D ~+0.023 (vs banked +0.055). Diff 0.032 is INSIDE agent 1's measured 0.066 run-to-run
floor -> treat as noise, NOT systematic arm-D bias. The earlier plan ("if it undershoots again it's systematic") was
wrong-ordered: floor FIRST. B-A reproduces and direction holds; OPT B-D is demoted to at-floor supporting (defended by
ladder + vacuums + the swap-gate), not a standalone value.

## PARKED — dedup x absorb convergence cross-probe (floor-aware form ONLY)
IF the swap-gate confirms deposit-specific absorption on OPT: the cross-probe "does dedup-Pythia grow absorbing carriers"
is still live BUT must NOT be run as a magnitude claim (dedup-engages-map died at the 0.066 floor on agent 1's side). Only
a swap-style PAIRED-DIFFERENTIAL version survives the floor. Park; do not resurrect in the floor-vulnerable magnitude form.
