# PRE-REG — Detector sequential-paradigm-surge CF: does adding a fake-generator paradigm erode prior-paradigm detection across an ordered sequence? 2026-07-11.

SHA256-lock before the full run. Extends the detector team's pairwise swap↔diffusion CF result (recorded: swap AUC 0.913→0.224
when a head learns diffusion; GPM fails — "the free lane does not exist," paradigms anti-parallel cos −0.95) to a TRUE ORDERED
N-paradigm surge with a full retention matrix. This is the piece that doesn't exist yet, and it tests whether the antagonism
COMPOUNDS across a real sequence and whether GPM (protecting the ACCUMULATED prior memory) holds as paradigms stack.

## HARD ISOLATION (locked)
Frozen backbone `prod_swap_seed0.pt` COPIED read-only into `D:/IC_experiments/detector_cf/`. Paradigm crops GLOBBED read-only
from `D:/corpus/face_swap_corpus` (too big to copy). Writes ONLY to `./out/`. NEVER retrains/touches the live
`prod_swap_seed*.pt` ensemble or the served detector. $0 LOCAL CPU (frozen 128-d features + a linear head).

## Design (faithful to cl_gpm_diffusion.py)
- **Ordered sequence:** swap (ffpp) → diffusion (deepfakeface) → celebdf → dfdc. swap/diffusion = the known anti-parallel pair;
  celebdf/dfdc = in-wild swaps (domain-shifted, "still swaps") — so the sequence also tests whether a later swap-like paradigm
  RESTORES swap or ERODES diffusion.
- **SHARED real pool** (ffpp/real, label 0) across all paradigm tasks → the only thing differing between tasks is the FAKE
  paradigm (isolates the CF axis; matches cl_gpm_diffusion's shared-reals design).
- Frozen DetectorCNN → 128-d penultimate feature → linear Head. Sequential: train head on paradigm-1, continue-train on 2,3,4;
  after EACH add, measure detection AUC on EVERY paradigm (the retention matrix).
- **3 arms:** NAIVE (plain SGD); GPM (project the new-paradigm gradient orthogonal to the ACCUMULATED prior-paradigms' memory
  subspace = QR-orthonormalized union of each prior paradigm's top-K=20 fake-feature PCA); GPM-RANDCTRL (random subspace of
  matched dim — mandatory control). Metric = AUC.

## Pre-registered read (locked before the full run)
- **EROSION / COMPOUNDING:** in NAIVE, prior-paradigm retention drops as paradigms are added. Reproduce the known swap collapse
  when diffusion lands; then does adding celebdf/dfdc erode further, or does a swap-like paradigm RESTORE swap? Report the full
  matrix; quantify per-step drops.
- **DIRECTIONAL (GPM aimed):** GPM prior-paradigm retention > NAIVE AND > RANDCTRL → the protection is aimed at the accumulated
  memory, not generic. If GPM protects priors only by STARVING the new paradigm (like the pairwise result) → the antagonism holds
  across the sequence (no free lane, compounding). If GPM ≈ NAIVE ≈ RANDCTRL → projection can't help at all.
- **The convergence check:** this should reproduce the arc's finding on a *sequence* — directional antagonism, GPM insufficient
  when paradigms are anti-parallel — now with the compounding structure a single pair can't show.

## Caveats (honest)
- Split = deterministic filename-hash, **NOT identity-disjoint** → absolute AUCs likely inflated (identity leakage), BUT the
  forgetting DELTAS (same fixed test set, before vs after a surge) are robust to absolute-level inflation. Flagged; not fixed here.
- Shared reals span domains (ffpp reals vs diffusion fakes' IMDB-WIKI domain) → some domain signal possible; matches
  cl_gpm_diffusion's design and the random-control guards aimedness. Head-level CF only (frozen backbone), not representational.
- SUBSAMPLE=220/paradigm (CPU budget); single seed; one order. Small; report the matrix, don't over-read sub-0.03 deltas.

## Ops
`probe_detector_seq.py` (reuses cl_gpm_diffusion machinery: load_backbone/feat/extract/continue_train-GPM/pca_basis/rand_basis) —
py_compile-clean; --smoke validated end-to-end (2s: backbone load, glob, feature extract, sequential naive arm; swap 0.726→0.611
when diffusion added → forgetting visible). $0 local CPU, minutes. Output `out/seq_detector.json`.

## Status
~~Built + smoke-validated. NOT run full. Next: LOCK → `python probe_detector_seq.py` → read the retention matrix (naive/gpm/rand)
+ compounding + GPM-aimedness.~~
**[STALE — CORRECTED 2026-07-14 by staleness audit. IT RAN.]** `out/seq_detector.json` exists with `smoke: false` (full 4×4 matrix).
No standalone `VERDICT_DETECTOR_SEQ` was written because the verdict was **absorbed**: `VERDICT_DETECTOR_ROUTE_2026_07_11.md` reports
its rows verbatim (naive swap 0.651 / diffusion 0.683). Do NOT re-run. See `D:/IC_experiments/STATUS_RECONCILIATION_2026_07_14.md`.
