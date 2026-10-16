# VERDICT — Anchor A: Librarian-Ω (free-lane real retriever). 2026-06-28.

Pre-reg `PREREG_ANCHOR_LIBRARIAN_OMEGA_2026_06_28.md` (LOCK 129df97c). Local CPU/GPU, non-mutating; script
`D:/SportsThought/librarian/training/librarian_omega.py`; result `.../data/librarian_omega.json`. V recomputed
deterministically (mirrors b2_cf.fisher_basis: K=64, N_FISHER=120, SWAP_SEED=12345, seeds 0) from the b1 (w_A) checkpoint;
net displacement w_B−w_A from the b1/b2 checkpoints. Substrate = SDT Librarian (ms-marco-MiniLM-L-6, BERT, 6L×6 carriers).

## Result
- **Preferential-overlap gate PASSES:** Ω_lib(mean) = **0.0460** vs swap **0.000332** → **138×**. The realized task-B
  displacement preferentially enters the academic (task-A) Fisher curvature subspace V, far above a matched random subspace.
  (Absolute overlap is modest — ~5% of the net displacement lands in the top-64 V — much less than the per-step gradient inV
  b2_cf logged (0.448): the cumulative displacement partly cancels out of V, but the *preferential* signal is strong.)
- **Real forgetting outcome:** retention AUC delta = **−0.0078** (0.877→0.869). A real retriever that sustains substantial
  task-B-into-V overlap with near-zero forgetting.
- **Axis placement:** free-lane substrate point = (Ω_lib ≈ 0.046, retention ≈ −0.008) — moderate overlap, protection holds.

## Depth profile (the Ramasesh-aligned result, its own finding)
| layer | Ω_lib | K_E90 |
|---|---|---|
| L0 | 0.036 | 17 |
| L1 | 0.026 | 16 |
| L2 | 0.029 | 15 |
| L3 | 0.044 | 14 |
| L4 | **0.076** | 10 |
| L5 | 0.065 | **4** |
Curvature concentrates to near-rank-1 by L5 (K_E90 17→4) and the displacement-into-V overlap RISES with depth (peaks L4–L5)
— independently lands on Ramasesh's deep-layer concentration finding, on a different substrate/architecture.

## Role in the Ω law
Independent **free-lane** anchor (real retriever, no childhood) on the Ω axis: confirms the "where the interference lands
relative to the task-A curvature subspace" quantity is measurable and preferential on a real task, and ties the depth
concentration to Ramasesh. NOT a replication of Probe-9's specificity S (b2_cf has no childhood) — reported as a
substrate/Ω-placement + depth anchor, not as S. Pairs with Anchor B (deepfake, antagonistic/suppression-only extreme) and
Probe-9 (continuous Ω sweep, OPT/Pythia) for a three-substrate Ω law robust to the n=2-architecture criticism.
