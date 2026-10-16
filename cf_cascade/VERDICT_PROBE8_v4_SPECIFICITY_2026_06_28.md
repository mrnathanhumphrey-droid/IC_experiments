# VERDICT — Probe-8 v4: directional protection is SPECIFIC in OPT, GENERIC in Pythia. 2026-06-28.

Pre-reg v4 `PREREG_PROBE8_v4_SPECIFICITY_2026_06_28.md` (LOCK 0112d69b). N=100, paired, fp32, dual-cell (OPT a100_sxm4 +
Pythia gh200, ML-CF), heartbeat + smart killswitch. Results `gateA/probe8v4_specificity_{opt350m,pythia410m}.json`.
**Both cells passed ALL validity gates (nondestruct + bite) — this is DISPOSITIVE, not VOID-soft.** Builds on v3 (FINDING 1,
NOT global flatness, STANDS); v4 sharpened the null so "targeting" is a real test, not a Probe-7 re-expression.

## The question
v3 showed the protection is directional (not global). Is that directionality SPECIFIC to the actual phase-2 interference, or
GENERIC to real-task damaging directions? Three magnitude-matched in-V directions, all built to bite: **phase2** (realized
phase-2 displacement), **alttask** (a DIFFERENT held-out task's gradient — TREC — the childhood was never pointed at),
**randtop16** (random in the top-16 Fisher dims). N(d) = childhood's neutralization of direction d = dprop_NoCH − dprop_CH.
S = N(phase2) − N(alttask). The alttask null now BITES (drops task-A 0.26–0.33), so the comparison is real.

## Result → clean architecture dissociation on the specificity axis
| | OPT-350m (load-bearing) | Pythia-410m (inert) |
|---|---|---|
| N(phase2) | **+0.153 ± 0.012 (12.4σ)** | +0.097 ± 0.009 (10.3σ) |
| N(alttask) | **+0.031 ± 0.003 (11.2σ)** | +0.082 ± 0.007 (11.1σ) |
| N(randtop16) | −0.010 (−14.5σ) | −0.008 (−13σ) |
| **S = N(phase2) − N(alttask)** | **+0.122 ± 0.012 (10.1σ)** | **+0.014 ± 0.012 (1.2σ, n.s.)** |
| gates | nondestruct ✓ (0.78), bite ✓ | nondestruct ✓ (0.74), bite ✓ |

- **OPT = SPECIFIC.** Childhood neutralizes the ACTUAL interference direction almost completely (+0.153, phase2 0.147→−0.01)
  but barely touches a different task's direction (+0.031) — even though alttask is the STRONGER damager (0.333 vs 0.147).
  Pre-committed SPECIFIC branch met outright (S>0 ≥2σ AND N(phase2)>N(randtop16) ≥2σ).
- **Pythia = GENERIC.** Childhood neutralizes phase2 (+0.097) and alttask (+0.082) ~equally; S = +0.014 (1.2σ, n.s.). Both
  >> N(randtop16) (so still directional, not global), but NOT aimed. Pre-committed GENERIC branch met.
- **The dissociation:** S_OPT − S_Pythia = +0.108, **~6.5σ.** The load-bearing architecture protects *specifically* against
  the interference it faces; the inert one confers *broad* directional robustness.

## What this means (the arc, closed)
Probe-5 (deposit=debris) → 6 (not pinning) → 7/7b (reorganization, ~100%, dispositive) → 8/v3 (directional, NOT global,
clean) → **8/v4 (the directionality is SPECIFIC in load-bearing OPT, GENERIC in inert Pythia).** One coherent mechanism:
**catastrophic forgetting is directional damage in the task's curvature subspace; a non-destructive childhood protects by
reorganizing the task to be robust in those directions — and in a load-bearing architecture that robustness is aimed at the
specific interference, while in an inert one it is generic.** This refutes the global-flatness account (Mirzadeh'20/Mehta'23)
and adds an architecture-dependent specificity axis that the literature does not have.

## Honest caveats (carry into any write-up)
1. **Direction-construction asymmetry:** phase2 = realized 250-step displacement; alttask = one-shot averaged gradient —
   different *kinds* of direction. The OPT-vs-Pythia contrast holds the construction FIXED, so the dissociation is clean; but
   the absolute "specificity" magnitude has this asymmetry baked in. Fix before print: realized short-training displacement
   for alttask too.
2. **Why is OPT specific if the childhood is blind to phase-2?** Unexplained. Leading hypothesis: specificity tracks
   task-similarity (phase-2=subj is closer to the childhood stream than alttask=TREC). UNTESTED — the highest-leverage next
   probe (vary childhood↔phase-2 similarity, watch S).
3. **Scope:** two small models, one task pairing. DSI showed load-bearingness dies at 1B → scale replication is the #1
   reviewer bound. The dissociation is n=2 on the architecture axis.

## GO-FORWARD (leverage-ranked)
1. **★ Specificity mechanism (highest):** vary the childhood-stream ↔ phase-2-task similarity; test S ∝ similarity. Explains
   the result AND generalizes it in one sweep; cheap (same harness).
2. Scale (≥1B) + a 3rd/4th architecture on the load-bearing↔inert axis (turn n=2 into a trend; bound the claim).
3. Hardening: realized-path alttask (fix caveat 1); a 2nd task pair; confirmatories (7b 1e-5, Pythia 7b reorg).

Artifacts: `gateA/probe8v4_specificity_{opt350m,pythia410m}.json`; harness `cloud/probe8v4_specificity_lambda.py` (LOCK
0112d69b). Instances af16dc84 (OPT) + 50edbf4c (Pythia) terminated clean (no orphan); cost ~$30. Arc:
VERDICT_PROBE5/6/7/8(+v4); CONSOLIDATION_forgetting_flow.md. Lit: refutes global flatness (Mirzadeh'20/Mehta'23),
complements RL's Razor'25 (displacement-decoupling) by localizing the damage to one direction and showing arch-dependent
aimed neutralization.
