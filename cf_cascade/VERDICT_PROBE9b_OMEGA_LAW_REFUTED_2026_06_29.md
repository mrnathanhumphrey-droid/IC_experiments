# VERDICT — Probe-9b: the geometric Ω-law is REFUTED; protection is governed by interference magnitude. 2026-06-29.

Pre-reg `PREREG_PROBE9b_PHASE2_LEVER_2026_06_28` (LOCK 869ce94e) + `PREREG_PROBE9b_PYTHIA_ADDENDUM_2026_06_29` (LOCK
fa94862b, locked before the Pythia GPU). Harness `cloud/probe9b_phase2lever_lambda.py`. Two cells, phase-2 task = the Ω
lever, FIXED childhood = mix5, task-A = SST-2, N=100 paired, realized forgetting. Each measure emitted a self-contained
(P, Ω, σ_sem) triple. All instances self-terminated, $0 left running. Artifacts `gateA/probe9b_measure_{opt350m,pythia410m}_*.json`.

## Headline
**The displacement-overlap law does not exist.** The OPT-only ρ(P,Ω) = −0.77 that looked like a clean negative "free-lane"
law was a **forget-magnitude confound + n=6 leverage**. The pre-registered Pythia replication and the pooled
forget-controlled regression dissolve it. **What is robust across both architectures: P ∝ forget_noch — the larger the
interference, the more absolute protection childhood provides. Geometry (Ω) and semantic similarity (σ_sem) add ~nothing.**

## Data (per cell)
OPT (n=6): | task | Ω | forget_noch | σ_sem | P |  — subj 0.028/0.237/0.003/+0.142 · trec 0.038/0.095/0.012/+0.033 ·
emotion 0.120/0.118/0.022/+0.042 · newscat 0.164/0.041/0.008/+0.018 · dbpedia 0.174/0.057/0.010/+0.026 ·
agnews+yahoo 0.199/0.139/0.018/+0.021.
PYTHIA (n=7): subj 0.079/0.210/0.001/+0.090 · trec 0.090/0.080/0.001/+0.005 · subj+trec 0.104/0.210/0.001/+0.137 ·
emotion 0.231/0.043/0.004/+0.009 · agnews 0.277/0.121/0.004/+0.070 · yahoo 0.338/0.086/0.002/+0.045 ·
dbpedia 0.376/0.039/0.001/+0.026.

## Correlations
| | OPT n=6 | PYTHIA n=7 |
|---|---|---|
| ρ(P, Ω) | −0.771 | −0.179 |
| ρ(P, forget) | +0.657 | +0.821 |
| ρ(P, σ_sem) | −0.029 | −0.536 |
| ρ(P, Ω \| forget) | −0.789 | **+0.706** (sign flips vs OPT) |
| ρ(P, forget \| Ω) | +0.686 | +0.912 |
| ρ(Ω, forget) | −0.314 | −0.607 |

**Pooled, within-cell z-scored OLS (n=13), P ~ Ω + forget:** β_Ω = **+0.034**, β_forget = **+0.915**, R² = 0.799.
Pooled partial ρ(P, Ω | forget) = **+0.029**; ρ(P, forget | Ω) = +0.786. Univariate β_Ω(alone) = −0.528 — the raw negative
slope — **vanishes the instant forget enters the model.**

## Why the OPT −0.77 was a mirage
1. **Confound:** in OPT's task set the biggest interferers (subj, forget 0.237) happened to sit at the lowest Ω (0.028),
   so a forget-driven P produced a spurious negative P–Ω slope. ρ(Ω,forget) was only −0.31 in OPT (looked safe), but with
   n=6 that is not enough separation.
2. **Leverage (jackknife):** raw OPT ρ(P,Ω) ranged −0.60..−0.90 across leave-one-out (looked robust), BUT the *partial*
   ρ(P,Ω|forget) collapsed from −0.79 to **+0.00** when agnews+yahoo was dropped — the "Ω independent of forget" claim
   rested on one point.
3. **Replication:** Pythia, with the big-forget task (subj+trec) placed at *mid* Ω, gives raw ρ(P,Ω) = −0.18 and a partial
   that **flips to +0.71**. No consistent geometric law across architectures.
4. **Pooling:** with forget controlled across both cells (n=13), Ω contributes essentially zero (β=+0.03, partial ρ=+0.03).

## The law that survives
**P ∝ forget_noch** (interference magnitude), robust in both architectures (OPT ρ +0.66; Pythia ρ +0.82, partial +0.91;
pooled β +0.92, R² 0.80). Mechanistically mundane and expected: more task-A damage from phase-2 → more headroom for
childhood to reduce it in absolute terms. **σ_sem (Ramasesh task-A↔phase-2 semantic similarity) is dead** (OPT ρ −0.03;
Pythia's −0.54 is itself a forget/Ω-collinearity artifact, not a semantic signal).

## Standing on the locked predictions (recorded, not spun)
- **P1 (ρ(P,Ω) ≥ +0.70 positive, with Ω*):** FALSIFIED on sign in OPT (−0.77), then the magnitude itself REFUTED on
  replication (pooled |partial| ≈ 0). No Ω-law of *either* sign survives. Ω* never claimable (addendum §4 gate: a law must
  exist first; pooled mid-band Ω∈[0.10,0.25] had 6 points, so coverage was adequate — there simply is no crossover to find).
- **P2 (geometric vs Ramasesh head-to-head):** in OPT, geometric "beat" semantic (ρ(P,Ω|σ)=−0.82 vs ρ(P,σ|Ω)=+0.44) — but
  that win evaporates once forget is the real competitor. The honest P2 outcome: **neither geometry nor semantics governs P;
  interference magnitude does.** The locked P2 omitted forget as a contender; that omission is the lesson.
- **P3 / H_ARCH vs H_UNIV (addendum §3):** Pythia ρ(P,Ω) = −0.18 lands formally in the H_ARCH band (|ρ|<0.40). But I do
  NOT claim the pre-committed "architecture-dissociation win," because the pooled regression shows the OPT side dissolves too.
  The defensible read is not "architecture modulates a real Ω effect" — it is **"Ω never robustly drove P in either cell."**

## Walk-backs (explicit)
- **The "free-lane / orthogonal-lane" mechanism is UNSUPPORTED by Probe-9b.** It was a clean story (reorganization buys a
  direction interference doesn't occupy → protection highest at low Ω). The cross-task overlap data do not back it; the
  −0.77 that motivated it does not replicate. The story is retired unless a future design isolates Ω from forget.
- **The "3-substrate Ω law" is NOT assembled.** Ω does not govern protection even within the LLM cells, so placing Librarian
  (Anchor A) and Deepfake (Anchor B) on an Ω axis is moot. Those anchors **stand as independent findings** — Deepfake:
  suppression-only protection in a frozen (no-reorg) backbone; Librarian: protection holds in a free-lane retriever — but
  they do not form an "Ω predicts protection" law.

## What Probe-9b does NOT touch
The mechanism findings upstream are independent of this cross-task overlap result and stand:
Probe-7/7b (protection = REORGANIZATION, damage-per-unit ÷3, not pinning), Probe-8 v3 (DIRECTIONAL not global flatness),
Probe-8 v4 (OPT SPECIFIC / Pythia GENERIC aiming). Probe-9b tested a different claim — whether the *displacement-overlap
geometry predicts protection magnitude across tasks* — and the answer is no.

## Methodological win
OPT-alone (ρ(P,Ω)=−0.77, partial −0.79) would have shipped a false negative-Ω law. It was caught BEFORE becoming a claim by
exactly the pre-registered machinery: addendum locked before the result, forget pre-named as a covariate, the OPT jackknife,
and the Pythia replication on a coverage-optimized ladder. The negative result is the system working.

## Status
Probe-9b CLOSED. Ω-law refuted; P∝interference-magnitude banked. Open question for any successor: is there a geometry that
predicts *fractional* protection once interference magnitude is regressed out? (Probe-9b says displacement-overlap is not it.)
