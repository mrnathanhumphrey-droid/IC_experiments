# PRE-REG — Librarian Phase-3(2): is fantasy-vs-rules a CONTESTED direction (route) or a constrained one (hard-neg fix)? 2026-07-11.

SHA256-lock before any training run. Applies the DETECTOR diagnostic (cos of discriminative directions predicted swap↔diffusion
antagonism, cos −0.93 → route; swap↔celebdf cos +0.59 → constrained/share) to the Librarian's one open gap: PRECISION/#1-rank,
where a fantasy "keeper" chunk intrudes on a soccer-rules "goalkeeper" query (the F2 keeper collision). Decides the remediation:
if fantasy and rules pull the reranker in OPPOSED directions → route to separate poles; if merely constrained/aligned → the
keeper collision is a HARD-NEGATIVE problem, fixable WITHOUT routing (and un-confounds the VOIDED Oracle-agent fix, which
downsampled fantasy 720→90 = data-starvation, not hard-negs).

## HARD ISOLATION (locked, unchanged from Phase-1)
READ-ONLY on `D:/SportsThought`. All inputs COPIED into the sandbox (fantasy corpus_staging/*.jsonl → corpus_fantasy/,
task_a_fantasy.jsonl → data/; rules already staged). Writes ONLY to `./out/`. NEVER touches the production Librarian, its
checkpoints (adapter_a.pt, checkpoints/*), build_artifacts, the Fly app, or git. No rebuild, no deploy, no push. Any production
graduation of a fix = a separate, explicit, mockup-first proposal.

## Data (self-contained, both poles resolve — verified)
ctext = 573 rules chunks (soccer/nba/nfl) + 169 fantasy chunks = 742 combined. TWO POLES:
- RULES pole = soccer+nba+nfl rules tasks pooled (Phase-1's usable Qs).
- FANTASY pole = task_a_fantasy (720 Qs; ALL positives resolve in the 169 chunks; incl. 40 nfl_keeper + 40 nba_keeper).
Self-contained candidate pools: positives = resolved task positives; negatives = sampled from the COMBINED 742-chunk corpus with a
SHARED negative pool across both poles (the detector's shared-reals analog → the only thing differing between poles is the POSITIVE/task).
Model = `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M, all trainable). Metric = AUC / R@5, NEVER P@1. Deterministic hash retention split.

## Three diagnostics
1. **DIRECTIONAL COSINE (the cheap predictor, direct detector analog).** At the BASE model, mean relevance-gradient direction per
   pole g_pole (per carrier + pooled). Report cos(g_rules, g_fantasy) AGAINST the scale set by within-pole split-half consistency
   cos(g_rules_a, g_rules_b) (how well-estimated a direction is) and a random-relabel null. ALSO cos(Δθ_rules, Δθ_fantasy) — the
   actual fine-tuned UPDATE directions (higher-fidelity, computed for free from the surge arms). This is the exact analog of the
   detector's cos(d_swap, d_diffusion).
2. **CF SURGE, both orders.** base→train RULES→retention(rules, incl. keeper subset)→continue-train FANTASY→re-measure rules
   retention; and the reverse (base→FANTASY→+RULES). Erosion (Δ retention) + inV vs inSwap of the surge on the prior pole's Fisher V
   (directional landing). Reuses the Phase-1 fisher_basis / surge_overlap machinery + mandatory random-swap control.
3. **KEEPER INTRUSION (the concrete symptom).** After JOINT training on both poles, for soccer-goalkeeper rules queries measure the
   rank of intruding fantasy-keeper chunks (and vice versa for fantasy-keeper queries), vs a rules-only / fantasy-only baseline
   reranker. Quantifies whether the collision is real and whether it's a cross-pole ranking intrusion.

## Pre-registered DECISION rule (locked before the run)
- **CONTESTED → ROUTE to separate poles**, IF: pooled cross-pole cosine (gradient OR Δθ) is NEGATIVE (< −0.10, opposed) OR the CF
  surge erodes prior-pole retention by > 0.02 AUC in EITHER order WITH directional landing (inV ≫ inSwap).
- **CONSTRAINED / NOT-CONTESTED → HARD-NEG fix (no routing)**, IF: cross-pole cosine ≥ ~0 (aligned/orthogonal) AND surge retention
  HOLDS within noise (|Δ| ≤ 0.02) — even if inV is high (= NON-DESTRUCTIVE OVERLAP, the Phase-1 rules↔rules pattern).
- **EXPLICIT detector lesson carried:** high inV ALONE ≠ contested (positive-cos = constrained coexistence, not antagonism). The
  SIGN of the cosine + ACTUAL erosion decide. Do not call "contested" off overlap magnitude.
- **What the decision buys (so it has teeth):** contested → the Librarian needs pole-routing (fantasy vs rules reranker / intent
  router); joint training makes them fight and keeper is the symptom → borrow the detector's route lesson. Not-contested → keeper is
  a hard-negative problem: add cross-domain hard negatives (fantasy-keeper as explicit NEG for goalkeeper queries & vice versa),
  NOT routing, NOT downsampling — the clean un-confounded fix the VOIDED Oracle-agent recipe missed.

## VOID / caveats
6-layer MiniLM is shallow → headline the cosine + surge erosion + keeper intrusion, not a per-layer depth monotone. Self-contained
pools (task-original full candidate lists dropped, as Phase-1) = deliberate clean scope; negatives sampled, single seed, one order
per surge. Retention sets modest → per-pole, NEVER pooled; sub-0.02 AUC = noise. Base-gradient cosine is a first-order linearization;
Δθ cosine is the faithful update-direction measure — report both, weight Δθ. Smoke settings (1 epoch, tiny) NON-diagnostic.

## Compute / ops
LOCAL GPU (Phase-1 confirmed cuda), ~minutes, $0, no Lambda. New driver `probe_fantasy_rules_direction.py` reusing
probe_librarian_seq.py machinery (fresh/ce_scores/train_on/fisher_basis/surge_overlap/carrier_names) + discriminative_dir + Δθ +
keeper-intrusion readout. py_compile-clean + --smoke before the full run. Output `out/fantasy_rules_direction.json`.

## Status
PREREG written + fantasy inputs staged + resolution verified (720/720 positives resolve; 17 soccer goalkeeper chunks for the
collision). Next: SHA256-lock → build driver → --smoke → full run → apply the decision rule → bank.
