# PRE-REG — Thread #4: CURVATURE ENGINEERING — can we design protection for a CHOSEN capability, and is it constrained by the cosine geometry? 2026-07-11.

SHA256-lock before the run. The inverse of the whole arc: instead of MEASURING which capabilities survive interference, DESIGN the
protection. Given a desired "protect set" S, engineer the interference update (project it off span{d_p : p in S}) and ask: is the
REALIZED protection profile = the DESIRED one? Prediction from #3 (entanglement) + the route arm: you can only engineer protection
profiles that RESPECT the cosine cluster structure — protect the swap-family as a UNIT or diffusion alone, but you CANNOT protect
swap while letting its aligned neighbours (celebdf/dfdc) forget. Closes the arc: one cosine geometry governs learn / forget / unlearn
/ PROTECT.

## HARD ISOLATION (unchanged)
Detector sandbox. Frozen backbone (read-only), crops read-only, writes only ./out/. Never touches live prod. $0 local CPU.

## Setup
- Universal FAKE-vs-REAL head H over swap/diffusion/celebdf/dfdc (as #3). Per-paradigm directions d_p (train split).
- **Interference B = "collapse toward diffusion-only":** continue-train H on relabeled data where DIFFUSION stays fake(1) and
  swap/celebdf/dfdc (+ reals) become real(0). This erodes swap/celebdf/dfdc detection — a strong, directional interference.
- **Curvature engineering = protect a chosen set S:** run the SAME interference but GPM-project the update off span{d_p : p in S}
  (orthonormalized). S's detection directions are shielded from the erosion.

## Desired protection profiles to attempt (each = one engineered run)
S ∈ { {} (baseline, no protection), {swap}, {celebdf}, {dfdc}, {diffusion}, {swap,celebdf,dfdc}=family, {swap,diffusion} }.
For each S, measure post-interference retention AUC of ALL FOUR paradigms.

## Pre-registered read (locked before the run)
- **ENGINEERING WORKS (protection is aimable):** protecting S raises retention of S's paradigms vs baseline (S={}) — the update is
  successfully shielded off the chosen directions.
- **CONSTRAINED BY ENTANGLEMENT (the arc-closing claim):** protecting an ALIGNED singleton LEAKS to its neighbours — e.g. S={swap}
  also raises celebdf/dfdc (cos +0.6/+0.8) by a clear margin > 0.03, so you CANNOT protect swap alone. Protecting the FAMILY
  S={swap,celebdf,dfdc} cleanly retains all three. Protecting S={diffusion} (anti-parallel) is INDEPENDENT — it does NOT raise the
  swap-family. → the realizable protection profiles are exactly the cosine CLUSTERS, not arbitrary subsets.
- **FIDELITY curve:** define leakage(S) = mean retention-gain on paradigms NOT in S that are ALIGNED with S (cos>0.3) vs those
  ANTI-PARALLEL (cos<0). Predicted: high leakage to aligned, ~0 to anti-parallel — protection spills along +cos, exactly as
  forgetting (#1) and unlearning-collateral (#3) do. Same geometry, third time.
- **NULL / failure:** if protecting {swap} does NOT leak to celebdf/dfdc (protection IS isolatable) → entanglement doesn't bind
  protection → report (would contradict #3's collateral finding). If GPM-protection doesn't raise retention at all → the erosion
  isn't in the projected subspace; report.

## Caveats (honest)
Head-level only (frozen backbone) — engineering the linear readout's curvature, not representational. Filename-hash split (abs AUC
inflated; deltas/leakage-structure = the claim). GPM here = closed-form projection of the interference gradient (a stand-in for a
designed "childhood curriculum"; the curriculum-data version is a heavier follow-up). Single seed. The directions d_p are estimated
means. This is a mechanism demo that protection obeys the cosine geometry, not a deployable CL system.

## Ops
`probe_curvature_engineering.py` reusing #3/probe_detector_seq machinery (universal head, per-para dirs, continue_train with U=span(S)).
py_compile + --smoke before full. $0 local CPU, seconds. Output `out/curvature_engineering.json` (retention matrix over S-profiles +
leakage-by-cosine).

## Status
PREREG written. Next: SHA256-lock → build → --smoke → full → read the S-profile retention matrix + leakage-along-+cos → bank (arc close).
