# PRE-REG — Anchor A: Librarian-Ω (free-lane, real retriever, CPU $0). 2026-06-28.

To be SHA256-LOCKED (`PREREG_ANCHOR_LIBRARIAN_OMEGA_LOCK.json`). Independent-substrate anchor for the Probe-9 Ω law
(`PREREG_PROBE9_AIMING_AXIS_v2_MEASURE9` LOCK 6e269459). Substrate = the SDT **Librarian** retriever (ms-marco-MiniLM-L-6,
BERT, 6 layers × 6 carriers), a real free-lane task. Pure LOCAL CPU analysis over EXISTING artifacts — no GPU, no training,
non-mutating (reads `D:/SportsThought/librarian` checkpoints, writes a NEW json under that project's data dir).

## Honest scope (what this anchor IS and IS NOT)
b2_cf has **no childhood intervention** (it is w_A → surge-on-taskB only). So this anchor does NOT replicate Probe-9's
specificity statistic S = N(phase2)−N(alttask). What it DOES contribute, on an independent real-task substrate:
1. **An Ω-axis datapoint at the free-lane end:** Ω_lib = overlap of the realized task-B displacement with the academic
   (task-A) Fisher curvature subspace V — the same "where the interference lands relative to the task-A subspace" quantity,
   here as a single vector-vs-subspace projection — paired with the real retention outcome.
2. **The K_E90 / inV depth-concentration profile** as its own result (curvature concentrates to near-rank-1 by ~layer 5;
   inV rises 0.24→0.74 L0→L5) — independently lands on Ramasesh's deep-layer finding.
It places a real retriever on the Ω axis so the three-substrate law is not architecture-bound. Reported as such; not as S.

## Inputs (all on disk, confirmed)
- w_A = `librarian/model/checkpoints/b1` (safetensors); w_B = `.../b2` (post-surge). Academic data via `train_b1.load()`
  (`task_a_full.jsonl`). Retention outcome already logged in `training/data/b2_cf_metrics.json`
  (`summary.ret_auc_w_A`/`ret_auc_post`/`ret_auc_delta`; per-checkpoint `forget_trajectory[*].retention.auc`).

## Method (locked — mirrors b2_cf.fisher_basis exactly)
Carriers = the 36 `bert.encoder.layer.*` 2-D `.weight`s. V[c] = top-K=64 right singular vectors of the per-example
academic-gradient matrix at w_A (N_FISHER=120, SWAP_SEED=12345 for the matched random-orthonormal control Vswap),
identical to b2_cf (torch.manual_seed(0), np.random.seed(0)). Recompute V deterministically (it is not saved to disk).
Net displacement per carrier: dW[c] = flatten(w_B[c] − w_A[c]).
- **Ω_lib[c] = ||P_V[c](dW[c])||² / ||dW[c]||²** (cos²θ between the net displacement vector and the academic subspace V).
- **swap control: ΩSwap[c] = ||P_Vswap[c](dW[c])||² / ||dW[c]||²** (matched random subspace — if Ω_lib ≈ ΩSwap the overlap
  is not preferential).
- Aggregate Ω_lib (carrier-mean) + by-layer + by-type; report Ω_lib vs ΩSwap ratio.

## Pre-registered reads (locked)
- **Preferential overlap (gate):** carrier-mean Ω_lib > ΩSwap by ≥10× (b2_cf already saw per-step inV/inSwap ≈ 1373×; the
  net-displacement version is the dispositive form). If Ω_lib ≈ ΩSwap → the displacement does not preferentially enter V →
  report as a free-lane LOW-Ω point.
- **Anchor placement:** record (Ω_lib_mean, retention_delta = ret_auc_post − ret_auc_w_A) as the free-lane substrate point
  on the Ω axis. b2_cf retention delta is small (≈ −0.008) at high per-step inV — i.e. a real retriever that sustains high
  task-B-into-V overlap with little forgetting; the anchor states where on Ω that sits and with what outcome.
- **Depth profile:** by-layer Ω_lib and K_E90 (from metrics json) reported as the Ramasesh-aligned depth result. Shallow
  (6 layers) → report the profile, do NOT headline a monotone significance test (b2_cf's own discipline).

## Validity gates
- w_A retention recompute must match `b2_cf_metrics.json` anchor (reproduce gate). V recompute uses the SAME seeds/N_FISHER/K
  → deterministic. If model/data fail to load → report blocked (do NOT fabricate).

## Deliverable / run
`cloud/librarian_omega.py` (or `librarian/training/librarian_omega.py`) — local CPU, mirrors fisher_basis, computes the
above, writes `librarian/training/data/librarian_omega.json` (NEW file; does not touch b2_cf outputs). ~few min CPU.
This anchor's prereg gate is the Ω_lib>>ΩSwap preferential-overlap check + honest placement; no GPU, no money.
