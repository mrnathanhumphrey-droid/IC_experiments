# PRE-REG — Probe 2 clean rank ablation (confound-killer). 2026-06-23.

**Why now:** the vacuums harness piggybacked a "B−D_k" rank read, but it projected the childhood GRADIENT out of V per K,
so childhood learning (`childhood_lossdrop`) was NOT matched across K (OPT 6.66→15.41). Rank was confounded with
learning-amount, and the confound runs co-directional with the OPT high-K erosion → the OPT core-vs-shell read is not
bankable. This probe removes the confound by construction.

**Question (pre-committed):** does map load-bearingness live in the tiny task-A curvature CORE (K_E90 ≈ 14 OPT / ≈ 5–9
Pythia, from `basisinv_*.json`) or in the broad V shell (k up to 128)?

**Design.** Band cells ONLY (opt350m, pythia410m; dead at 1B). Matched dispositive regime (lr_child 3e-5/400,
lr_depth 5e-5/250, held-out subj, N=30 paired seeds, V = top-128 task-A Fisher at w_A). Per seed: childhood run **ONCE**
(full, no projection) → deposit `dfull` per carrier; `d_inV = P_V(dfull)`, `M = ‖d_inV‖_F`, `d_outV = dfull − d_inV`.
Rank arms reuse the SAME childhood: carriers = `w_A + d_outV + renorm(P_{V[:k]}(dfull) → M)`, non-carriers = `wB_full`.
Because childhood is one run/seed, `childhood_lossdrop` is **identical across all k** → the vacuums confound is dead.

**Arms / ladder.** OPT K = {0, 14, 32, 64, 128}; Pythia K = {0, 5, 9, 32, 64, 128}.
- k=0 ≡ arm D (out-of-V deposit only). k=128 ≡ arm B (full deposit = wB_full exactly).
- SWAP arm at the knot: curvature-matched random rank-knot subspace in V, magnitude-matched to M (deposit-specificity AT
  the core rank; the full-V swap-gate already banked deposit-specificity overall).

**Magnitude control (prereg-mandated):** every truncated deposit is re-normalized to the full in-V Frobenius magnitude M
→ rank is isolated from magnitude. (At k=128 the renorm is identity, so k=128 reproduces arm B exactly.)

**Reproduce-anchor (GATES the read):** B−A at k=128 must hit the banked B−A (OPT +0.089 / Pythia +0.073) within 0.03,
else harness invalid. Demote any absolute level inside the measured ~0.066 run-to-run floor.

**Pre-committed outcomes:**
- **Monolithic-in-core** → `Bk−A` saturates by k ≈ K_E (the rank-14/9 arm already delivers ~full protection);
  load-bearingness lives in the genuine 90%-mass curvature core.
- **Broad-shell / distributed** → `Bk−A` keeps climbing 64→128; the load is in the shell ABOVE the tiny core
  (the basisinv "deposit sits in the core" enrichment is then incidental to protection — matches the vacuums hint,
  now confound-free).
- **Flat/null** → rank-k content delivers no protection at any k (protection is the out-of-V `d_outV` term; would echo
  the Pythia "protection is K-invariant / out-of-V" vacuums read).

**Swap read:** real k=knot `Bk−A` ≫ swap `Bk−A` → the core-rank deposit is deposit-specific; real ≈ swap → generic
curvature at the core rank.

**Falsifier discipline:** the read only counts behind the k=128 reproduce-anchor. Differentials (Bk−A, B−Bk) are paired
within-run so the ~0.066 floor largely cancels; do not over-read absolute magnitudes.

Harness `cloud/probe2_rank_lambda.py` + launcher `cloud/lambda_launch_probe2.py`. LOCK below.
