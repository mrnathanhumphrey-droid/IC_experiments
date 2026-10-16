# PRE-REG — Anchor B: Deepfake seed-robustness (suppression-only negative control, CPU ~$0). 2026-06-28.

To be SHA256-LOCKED (`PREREG_ANCHOR_DEEPFAKE_SEEDROBUST_LOCK.json`). Independent-substrate anchor for the Probe-9 Ω law
(LOCK 6e269459). Substrate = physics-detector **deepfake/diffusion** detector (FROZEN DetectorCNN backbone → 128-d feature,
linear head only). Local CPU (~3–5 min/seed). REQUIRES a one-line edit to another project's harness (see Ops) — present
before editing.

## Why this is the right anchor (the Ω-axis antagonistic extreme + suppression-only control)
The frozen backbone means **reorganization is structurally impossible** — only the linear head can adapt, so any protection
must be pure suppression/pinning (gradient projection), not reorganization. The diffusion-vs-swap endpoint is
**antagonistic** (`cos_meanshift_swap_diff = −0.95`, `cos_w_swap_w_diff = −0.273` in `cl_gpm_diffusion_result.json`) — the
extreme-low-Ω / anti-aligned end of the axis. So this substrate is the **negative control**: where reorganization can't
happen, protection should behave exactly as pinning theory predicts, and a RANDOM projection should give zero protection.
This is what makes the three-substrate span (antagonistic → free-lane) robust to the n=2-architecture criticism.

## Current state (confirmed on disk)
- Harness `D:/physics_detector/v2/learned_detector/cl_gpm_diffusion.py`; result `cl_gpm_diffusion_result.json`.
- Arms already implemented: baseline (head on swap), naive (continue-train → forgetting), **ARM3 GPM** (gradient projected
  off the swap-memory subspace = real suppression), **ARM4 GPM-randctrl** (projection off a RANDOM subspace = suppression-
  only negative control). ARM4 gives zero protection (swap AUC 0.913 → 0.223). Single-seed: `SEED = 0` (line 37).
- Antagonism confound already decomposed (`antagonism_confound_result.json`): the −0.95 is the DOMAIN vector
  (`cos_domain −0.909`), not the forgery signal (`cos_forgery +0.669`) — logged, carried as a caveat.

## Method (locked)
Sweep **SEED ∈ {0,1,2,3,4}**. Per seed, run the existing 4 arms unchanged; collect: swap AUC for baseline / naive / GPM /
GPM-randctrl, and the cosines (`cos_meanshift_swap_diff`, `cos_w_swap_w_diff`). Aggregate across seeds (mean ± SE).

## Pre-registered reads (locked)
- **Antagonistic endpoint FIRMS:** cos_meanshift_swap_diff < −0.5 in ≥4/5 seeds (mean ± SE reported); the single-seed −0.95
  was not a fluke. (Magnitude may vary; the SIGN and strong-negative are the claim.)
- **Suppression works, random does not (the negative control banks):** GPM swap-AUC > naive swap-AUC (real projection
  protects) AND GPM-randctrl swap-AUC ≈ naive/chance (random projection = zero protection), each across seeds with
  non-overlapping SE. This banks the **suppression-only negative control**: in a no-reorganization substrate, AIMED
  suppression protects and generic does not — the pinning-theory prediction, the foil to Probe-9's reorganization result.
- **Axis placement:** record this substrate at the antagonistic (anti-aligned) Ω-extreme with "protection = suppression
  only," contrasting the free-lane (Librarian) and the load-bearing-reorg (OPT, Probe-9) points.

## Validity gates
- Seed must actually vary the run (different draws) — verify the 5 result files differ. Report the domain-confound caveat
  (the antagonism is domain-driven) honestly in the verdict.

## Ops (REQUIRES cross-project edit — present before doing)
One-line change to `cl_gpm_diffusion.py`: `SEED = int(os.environ.get("SEED","0"))` (line 37) and seed-stamped output path
`cl_gpm_diffusion_result_seed{SEED}.json` (line ~306); ensure the random-orthonormal control uses the same SEED. Then
`for s in 0 1 2 3 4: SEED=$s python cl_gpm_diffusion.py` (local CPU). Aggregate into
`gateA/anchor_deepfake_seedrobust.json`. No GPU, no money. **Do not edit the other project's harness without an explicit go.**
