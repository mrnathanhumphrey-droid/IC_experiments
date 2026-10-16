# VERDICT — the specificity reckoning (Probe-11/12/13 + Hunt-1): 8v4 dissociation RETRACTED; new dose-response-sign law. 2026-07-09.

Consolidated verdict for the metric-hardening + red-team arc. Preregs/locks: Probe-11 v2 (realized-alt 3-cell) 332e7029,
v3 (gentle) a49e92e0; Probe-12 (scale OPT-1.3b) ef4d761e; Probe-13 (dose-response) 49d14b48. w_A builds: BLOOM-560m (anchor
0.84) + OPT-1.3b (anchor 0.93) via `build_wA_lambda.py`. All runs banked, $0 (all self-terminated). Harnesses:
`cloud/probe11_realizedalt_lambda.py`, `cloud/probe13_doseresponse_lambda.py`, `cloud/build_wA_lambda.py`.

## Headline
**The 8v4 "OPT-specific / Pythia-generic" architecture dissociation is RETRACTED — it was a null-threat-magnitude artifact.**
What survives and STRENGTHENS: childhood reduces forgetting **concentrated on the interference direction** (N_phase2 ≫
N_random), **robust across 3 model families (OPT, Pythia, BLOOM-partial) and scale-stable to 1.3B**. What is NEW and clean:
**the SIGN of the childhood→protection dose-response is architecture-dependent** — POSITIVE for OPT/Pythia (more childhood →
more protection), NEGATIVE for BLOOM (more childhood → less, then harmful). BLOOM is a genuine inverted-response outlier.

## Why the 8v4 dissociation dies (Hunt-1, local, $0)
S = N(phase2) − N(alttask) was the 8v4 specificity metric. It is dominated by which direction does more DAMAGE, not by aiming:
- **N(d) ∝ threat-damage of d**: across 12 (direction × cell) pairs, r(N, dprop_noch) = **+0.81** (the 9b law at the direction
  level — protection buys with threat size).
- **S ≈ sign(dp2_n − dalt_n)**: among properly-dosed cells r = **+0.986**. OPT/Pythia look "specific" only because phase-2
  out-threatens the (weak realized) alttask; OPT-1.3b flips "anti-specific" purely because at scale the alttask out-damages
  phase-2. Same artifact, three faces.
- **Metric-hardening confirms it**: on the realized-alt metric Pythia S = **+0.108** (was +0.014 GENERIC in 8v4) — i.e. Pythia
  is "specific" too; the dissociation evaporated. The one-shot alttask (8v4) saturated; the realized alttask is a weak null
  (bite<0.03). Neither alttask construction cleanly isolates aiming → **S-vs-other-task is not a valid aiming measure.**

The clean, construction-STABLE measure is **N_phase2 vs N_randtop16** (matched magnitude): all good-protector cells concentrate
protection on the interference (N_conc = N_phase2 − N_random > 0), across metrics and scale.

## Probe-13 dose-response map (N=50, N_phase2 vs childhood dose; slope on log10(dose))
| cell | family/scale | N_phase2 gentle → strong | slope | class |
|---|---|---|---|---|
| OPT-350M | OPT | +0.071 → +0.157 | +0.075 | good protector |
| Pythia-410M | Pythia | +0.074 → +0.113 | +0.044 | good protector |
| OPT-1.3B | OPT (4×) | +0.018 → +0.168 | +0.090 | good protector — **scales** |
| BLOOM-560M | BLOOM | +0.044 → −0.022 | −0.049 | **erodes — outlier** |

Three families/scales POSITIVE and scale-robust (OPT-1.3B strongest slope → the DSI "load-bearingness dies at 1B" hint does
NOT appear here). BLOOM NEGATIVE: childhood never concentrates (N_conc ~0 → negative) and gets worse with dose. At matched
dose (2e-6): OPT +0.071 vs BLOOM −0.003.

## Scope / caveats (honest)
- **BLOOM retA_drop is non-monotonic in lr** (0.047 at gentlest 3e-7, ~0.032 mid) → the retA_drop x-axis is unreliable for
  BLOOM; the finding is stated on the **dose (lr) axis**, which is clean. The 3e-7 point looks anomalous.
- N=50 (dose-response), N=100 (Probe-11). n=3 "good protector" archs + n=1 outlier — BLOOM is the only inverted case, so
  "architecture-dependent sign" rests on one negative exemplar. Needs ≥1 more BLOOM-like arch to generalize.
- One task-A (SST-2), one phase-2 (subj), one childhood (mix5). Scope caveat as ever.
- The alttask-null problem is unresolved (Hunt-2, damage-matched null, NOT run) — but it is now moot for the main claim, since
  the surviving measure is N_phase2-vs-random, not S-vs-alttask.

## What this does and does NOT touch
- **UNTOUCHED (stand):** Probe-7/7b reorganization (same movement, less damage — the *how* of good-protector cells), Probe-8v3
  directional-not-global (= the N_phase2 ≫ N_random concentration, now cross-family + scale-robust), Probe-9b (Ω-law refuted),
  Probe-10 (rehearsal-similarity in OPT).
- **RETRACTED/DOWNGRADED:** Probe-8v4 "aimed-specificity vs other-task" architecture dissociation → threat-magnitude artifact.
  The word "specific" should mean "concentrated on the interference vs random," NOT "vs other real tasks."

## New open question
**Why is BLOOM the lone eroder?** (ALiBi positional scheme? BigScience training corpus? something in how its childhood
perturbs the SST-2 deposit?) The next thread if pursued: a 2nd BLOOM-like arch to test whether "inverted dose-response" is a
class or a one-off; and a mechanistic look at what BLOOM's childhood does to the task-A curvature subspace.

### ADDENDUM (Probe-14, 2026-07-09) — the corpus is exonerated; ALiBi is the suspect.
Fired **XGLM-564M** (PREREG_PROBE14, LOCK b30d6579): multilingual like BLOOM, shape-matched (24L/1024h), differs mainly on
**positional (sinusoidal vs ALiBi)**. Result (N=27): XGLM is a **clear PROTECTOR** — N_phase2 +0.020→+0.149 (positive, growing,
z≈9.5 at 7e-6), N_conc grows to +0.152, N_random≈0. Opposite of BLOOM (+0.044→−0.022) at identical doses. So **the multilingual
corpus is NOT the cause of erosion** — BLOOM's inverted response points to its **ALiBi positional scheme** (or post-embed LN).
Mechanism: new `frac_child_in_V` metric shows XGLM's childhood reorganizes **3–4× above chance inside V** (0.37–0.45 vs ~0.10),
directly confirming the protector mechanism. Details: READOUT_bloom_fingerprint_2026_07_09.md. Still open: (a) BLOOM-frac re-run
(does BLOOM fail to reorganize in V?) — in flight; (b) MPT-1b (ALiBi+English) confirm — if it erodes too, ALiBi is the mechanism.

## Status
Probe-11/12/13 CLOSED. Specificity dissociation retracted; concentration+scale-robustness upgraded; dose-response-sign law
new (n=3 vs 1). Files `gateA/probe11_realizedalt_*.json`, `gateA/probe13_doseresponse_*.json`, `gateA/build_wA_*.json`.
OPS BUG NOTED: `lambda_launch_generic.py` intermittently hangs on the scp upload (hit Pythia + OPT-350m this session, ~$20–35
wasted before caught) — needs a scp timeout/retry + run-phase watchdog.
