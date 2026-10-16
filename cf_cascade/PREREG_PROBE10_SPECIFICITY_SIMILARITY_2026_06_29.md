# PRE-REG — Probe-10: is specificity (8v4) DRIVEN BY rehearsal-similarity? 2026-06-29.

To be SHA256-LOCKED (`PREREG_PROBE10_SPECIFICITY_SIMILARITY_LOCK.json`) BEFORE any GPU. Successor to Probe-8 v4
(`VERDICT_PROBE8_v4_SPECIFICITY_2026_06_28`, LOCK 0112d69b), which found OPT AIMS its childhood-protection at the specific
interference direction (S = N(phase2) − N(generic) > 0, 10.1σ) while Pythia is GENERIC (S n.s.). The UNEXPLAINED "why" is the
target here. Hypothesis (from the 8v4 caveat): **specificity is rehearsal-driven — OPT can aim at the interference because the
childhood rehearsed that task (or a same-family task); when the childhood rehearsed nothing like the interferer, only generic
protection remains.** Probe-9b refuted displacement-OVERLAP as a *magnitude* predictor; this is a DIFFERENT claim (does
rehearsal-similarity drive the *aiming* S, not the magnitude P) and uses a CONSTRUCTED causal contrast, not a measured covariate.

## DV (fixes the 8v4 asymmetry up front)
For an interference task T (realized 250-step phase-2 direction from w_A) and a childhood CH:
- N(d) = dprop_NoCH(d) − dprop_CH(d) = childhood's reduction of task-A accuracy-drop when the weights are perturbed along
  direction d at MATCHED magnitude (8v4 estimator; magnitude-normalized, so unlike P it is not forget-confounded).
- **S_T(CH) = N(T-realized-dir) − N(random-top-k-dir).** The random-top-k reference is CONDITION-INDEPENDENT, so S isolates
  the *aimed* component and cancels overall protection-magnitude differences between childhoods. (Both directions realized/
  normalized → the 8v4 phase2-realized / alttask-one-shot asymmetry is removed.) Paired NoCH vs CH per seed, N seeds.

## Design (CONSTRUCTED similarity, swap-controlled — not a regression)
task-A = SST-2 (w_A: OPT 0.823 / Pythia 0.753). V = top-128 task-A Fisher per carrier. fp32, N=100 paired (scope below).
**Four childhoods, identical except one swapped slot; #tasks=3 and total steps=400 held constant; backbone = {trec, subj}:**
| childhood | swapped slot |
|---|---|
| CH_agnews | agnews |
| CH_dbpedia | dbpedia |
| CH_yahoo | yahoo |
| CH_emotion | emotion |
Topic-classification family = {agnews, dbpedia, yahoo, newscat}; affect = {emotion} (SST-2 is task-A, excluded); backbone
{trec=question-type, subj=subjectivity} are non-topic, held in ALL conditions.

**Probe two interference directions** against all four childhoods (realized phase-2 from w_A): **T = agnews** and **T = yahoo**.
Rehearsal-similarity of each childhood to each T:
- T=agnews: CH_agnews = IDENTICAL; CH_dbpedia, CH_yahoo = SAME-FAMILY (topic); CH_emotion = UNRELATED (affect).
- T=yahoo:  CH_yahoo = IDENTICAL; CH_dbpedia, CH_agnews = SAME-FAMILY (topic); CH_emotion = UNRELATED (affect).
(Two interference probes from ONE set of 4 childhoods = the contrast twice, guarding against "agnews is special".)

**Cells:** OPT (test — specificity lives here) + Pythia (NULL control — 8v4 says S≈0; predict FLAT across conditions).

## Pre-registered predictions / verdict
- **P10-main (OPT):** S_T(CH) grades monotone with rehearsal-similarity: **S(IDENTICAL) > S(SAME-FAMILY) > S(UNRELATED)**,
  for both T=agnews and T=yahoo. Operational: Spearman ρ(S, similarity-rank) > 0 with IDENTICAL>UNRELATED gap ≥ 2σ.
- **P10-null (Pythia):** S flat across conditions (no monotone grade; |gap| < 2σ) → specificity is a load-bearing-arch property.
- **Format vs direction (post-hoc, pre-committed):** verify that format-family similarity (topic vs affect) also tracks the
  realized interference-direction overlap between the swapped task's gradient and T's gradient — report which notion of
  "similar" carries S (task-format vs interference-direction). Descriptive, not gating.
- **Verdict map:** GRADES (OPT monotone, Pythia flat) → SPECIFICITY IS REHEARSAL-DRIVEN (answers the 8v4 "why"); FLAT (OPT
  S same across conditions) → rehearsal-similarity is NOT the source (8v4 hypothesis falsified, look elsewhere); Pythia grading
  too → not arch-specific (reframe).

## Pre-named confounds (the 9b lesson — declared before firing)
1. **childhood dose/diversity:** #tasks=3 and total steps=400 fixed across all conditions (only the swapped slot differs).
2. **non-destructiveness:** fixed childhood lr (screen-calibrated per cell); report ret_A per condition; VOID-soft any
   condition with ret_A < w_A − 0.05.
3. **swapped-task ↔ task-A similarity** (emotion is closer to SST-2 than topic tasks are): handled by the random-reference
   subtraction (generic help raises N(random) too); REPORT N(random) per condition so this is visible.
4. **swapped-task interference magnitude:** report each swapped task's own task-A forgetting during childhood; S uses
   magnitude-matched directions so this cannot inflate S directly.
5. **direction construction:** T-direction realized (250 steps); random = top-k(=16) normalized; symmetric magnitude.

## Ops / scope
SHA256-lock before GPU. GPU_PREF excludes a10; killswitch each fire; heartbeat; read off the machine. Harness =
`cloud/probe10_simrehearsal_lambda.py` (reuse the 8v4 N-estimator: vproj_dir / perturb_eval / realized-direction machinery;
loop over the 4 childhood conditions; probe T∈{agnews,yahoo}+random; emit S per (CH,T)). CPU dry-run first.
**Scope: OPT-first**, 4 childhoods × (agnews-dir + yahoo-dir + random) × N seeds. If OPT grades → fire Pythia null. N and
exact cost set at build; expect a single OPT instance (childhoods reused across the two T-probes — efficient).

## Status
Design confirmed by user 2026-06-29. LOCKED c83884a7. Harness `cloud/probe10_simrehearsal_lambda.py` BUILT; CPU dry-run
`cloud/probe10_dryrun.py` ALL PASS; launcher registered (probe10; CHILD_LR/NSEEDS/PROBE_T pass-through). **FIRED OPT
2026-06-29** (iid 1b544d86f99b474d9a4840349b1b60a4, a100_sxm4 @ $1.99, NSEEDS=80, PROBE_T=agnews,yahoo, CHILD_LR=1e-5),
~7–9 h single instance, RESULT PENDING (`gateA/probe10_simrehearsal_opt350m.json`). If OPT grades → fire Pythia null
(CHILD_LR=3e-5) + write VERDICT_PROBE10; if flat → 8v4 rehearsal hypothesis falsified, look elsewhere.
