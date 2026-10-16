# PRE-REG — Application Phase-1: does sequentially adding a sport erode the Librarian's prior-sport retrieval? 2026-07-10.

SHA256-lock before any training run. First applied test of the CF/directional-interference framework on a REAL tool (the
SportsThought Librarian reranker). Counterfactual to production: production JOINT-retrains the adapter on all data every rebuild
(sidesteps sequential CF by construction); this probe interrogates the SEQUENTIAL path production avoids.

## HARD ISOLATION CONSTRAINT (locked)
READ-ONLY on `D:/SportsThought`. All inputs COPIED into the sandbox `D:/IC_experiments/librarian_cf/` (rules chunk bodies +
task labels + eval_metrics). The probe writes ONLY to `./out/`. It NEVER touches the production Librarian, its checkpoints
(`adapter_a.pt`, `checkpoints/*`), `build_artifacts`, the Fly app, or git. No rebuild, no deploy, no push. Any production
graduation of a Phase-3 fix would be a separate, explicit, mockup-first proposal.

## Task (self-contained, rules-only)
Rules reranking. `ctext` = all rules chunk bodies from `corpus_staging/rules_{soccer,nba,nfl}.jsonl` (573 chunks). Each sport's
questions (`task_rules_*.jsonl`) keep the rules-resolvable candidates; **negatives naturally include OTHER sports' chunks** →
cross-sport interference is measured directly. Per-sport train/retention split = deterministic hash (RETENTION_FRAC=0.25; MD5(qid)),
no leakage. Model = `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M, all weights trainable), fine-tuned (BCE, pos-weighted, 4 epochs,
lr 2e-5). Metric = **AUC / R@5, NEVER P@1** (saturates). Verified viable: 124/112/169 Qs all usable (≥1 pos + ~30 neg each).

## Instruments (ported discipline: per-(layer,carrier), never pooled; mandatory swap control)
- **Retention AUC/R@5** on each prior sport's frozen retention split, measured after each sequential add.
- **Fisher basis V** per (layer,carrier) = top-K=64 right singular vectors of the per-example gradient matrix (F=GᵀG) for each
  sport at the point just after it is learned. **Vswap** = random orthonormal subspace of EQUAL dim (seed 12345).
- **surge_overlap** = `inV` vs `inSwap` of the NEW sport's training gradients projected onto each PRIOR sport's V (does the surge
  land in the prior sport's curvature?).

## Arms
- **ARM 1 SEQUENTIAL:** soccer → nba → nfl. After adding each sport, measure retention on it + ALL prior sports, and the surge's
  inV/inSwap against each prior sport's V.
- **ARM 2 JOINT:** fresh model trained on all 3 sports together (production's approach). Measure per-sport retention.

## Pre-registered read (locked before the full run)
- **EROSION:** prior-sport retention AUC drops as later sports are added (soccer/nba lower in seq_final than at their own add),
  and seq_final < joint for prior sports. Quantify the **seq−joint gap** per sport. If ~0 → no sequential forgetting on this task.
- **DIRECTIONAL:** surge `inV` ≫ `inSwap`. Combined with erosion → destructive-directional interference (real + aimed, the
  antagonism mechanism). inV≫swap but retention HOLDS → **"non-destructive overlap"** (replicates b2_cf's academic→resolve result
  + the CF-arc mechanism on the sports sequence — curvature shared yet retention robust).
- **JOINT AVOIDS:** joint retention ≥ sequential → production's joint-retrain sidesteps whatever CF the sequential path incurs
  (turns "production avoids CF by construction" into a measured claim).

## VOID / caveats
Small retention sets (~34/31/50 Qs) → report per-sport, NEVER pool into one number; treat sub-0.02 AUC deltas as noise. ONE order
(soccer→nba→nfl) — order-permutation deferred. Rules-only (cross-corpus distractors dropped) = a deliberate clean-task scope.
6-layer MiniLM is shallow → headline the retention curve + inV-vs-swap, NOT a per-layer depth monotone. Smoke settings (1 epoch,
K=16) are NON-diagnostic.

## Compute / ops
Runs LOCAL on the GPU (smoke confirmed `dev=cuda`, fast) — ~minutes, **$0, no Lambda**. Driver `probe_librarian_seq.py`
(py_compile-clean; --smoke validated end-to-end: data load, sequential+joint arms, Fisher surge-overlap all fire; inV~0.26 vs
swap~0). Output `out/seq_result.json`.

## Status
Sandbox built + inputs copied + driver smoke-validated. NOT run full. Next: LOCK → `python probe_librarian_seq.py` (local) →
read seq-vs-joint retention + inV/inSwap → decide Phase-2 (localize / F2 antagonism).
