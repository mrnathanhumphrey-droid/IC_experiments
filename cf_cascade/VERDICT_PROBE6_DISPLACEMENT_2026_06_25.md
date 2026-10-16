# VERDICT — Probe-6 displacement-absorption. 2026-06-25.

Pre-reg `PREREG_PROBE6_DISPLACEMENT_2026_06_25.md` (LOCK 185910cd). N=100, paired, fp32, a100_sxm4 (ML-CF),
per-step heartbeat + smart killswitch. Result `gateA/probe6_displacement_opt350m.json` (VOID=false).
First probe purpose-built under the post-floor rule (N≥100 + paired). w_A anchor 0.823 reproduced; phase2_eroded=True.

## The question
Probe-5 showed the forgetting damage is **coordinate displacement in V** (snapping task-A coords back to w_A
recovered +0.10). DSI showed load-bearingness = the childhood *absorbing the gradient surge into V*. The bridge
nobody had measured: **does the absorption actually reduce the displacement?** Pre-committed H1 = "absorption =
pinning": childhood should leave task-A coords CLOSER to w_A in V (ΔD>0), protect retention (Δret>0), and the two
should correlate per-seed (r>0). Paired arms off the same w_A: **NoCH** (w_A→phase-2) vs **CH** (w_A→childhood→phase-2).
D = mean_carrier ||P_V(w_end−w_A)||; ret = SST-2 acc at w_end.

## Result (OPT-350m, N=100) → H1 FALSIFIED. Protection is NOT pinning.
| quantity | value | |
|---|---|---|
| Δret = ret_ch − ret_noch | **+0.0882 ± 0.0116 (+7.6σ)** | childhood PROTECTS task-A — load-bearingness reproduces clean |
| ΔD = D_noch − D_ch | **−0.1293 ± 0.0074 (−17.5σ)** | childhood ends FARTHER from w_A in V, not closer |
| r(ΔD, Δret) | **−0.322** | seeds with MORE in-V displacement get MORE protection |
| D_noch / D_ch | 0.489 / 0.618 | |
| ret_noch / ret_ch | 0.578 / 0.667 | |
| Dperp_noch / Dperp_ch (out-of-V) | 0.812 / 1.143 | out-of-V grew MORE than in-V |

All three of H1's pre-committed predictions for "pinning" came back the wrong way except the retention sign:
retention IS protected (+7.6σ), but ΔD is strongly NEGATIVE and the correlation is NEGATIVE. **The pre-committed
SURPRISE/REOPEN branch fired: Δret>0 while displacement does not go positive.** Load-bearingness does not work by
holding the task-A coordinates near w_A.

## The step-count confound (named, and why the conclusion survives it)
The CH arm trains 400 extra steps (childhood) before phase-2, so part of "ends farther from w_A" is trivially more
total motion. So the **absolute ΔD sign is confounded** — we cannot claim "childhood increases *phase-2* displacement."
But the two load-bearing conclusions are confound-FREE:
1. **The +0.088 protection** is real (childhood is non-destructive to task-A by construction; this is the reproduced
   load-bearingness effect, well-powered).
2. **The negative correlation r=−0.322 is the kill shot for pinning:** if protection came from staying near w_A, the
   less-displaced seeds would retain BETTER (positive r). The opposite holds. This is within-condition, step-count-robust.

## What it positively suggests (not yet proven)
- **Reorganization / relocation, not suppression.** Childhood protects task-A while moving *through* its subspace more,
  so the SST-2 function must survive larger excursions — the solution is relocated to a more robust region, or spread so
  no single coordinate is load-bearing. (Coheres with Probe-2: load lives in the broad SHELL, not a pinnable core.)
- **Faint deflection signal:** the FRACTION of motion landing in V drops with childhood (in-V fraction 0.516→0.476;
  out-of-V grew 0.81→1.14 vs in-V 0.49→0.62). Childhood routes a larger share of movement OUT of the task-A subspace —
  the DSI "deflection" picture — but it is small and is NOT what carries the +0.088.

## Reconciliation with Probe-5 (apparent tension, resolved)
Probe-5: snapping back toward w_A in V (proj) IMPROVED retention (+0.10). Probe-6: childhood ends FARTHER from w_A in V
yet retains BETTER. These coexist only if **distance-from-w_A predicts retention WITHIN a fixed path, but childhood
changes the path** — it relocates the robust region. Probe-5 moved within one (childhood-fixed) end-state; Probe-6
varies whether childhood relocates at all. Probe-7 tests this directly.

## GO-FORWARD
- **Probe-7 (next): confound-kill + mechanism.** Measure phase-2-only displacement from each arm's PRE-phase-2 anchor
  (NoCH=w_A, CH=post-childhood) to remove the step-count confound, and test suppression vs reorganization via
  forgetting-per-unit-phase-2-displacement. See `PREREG_PROBE7_*`.
- **Pythia Probe-6 (deferred, cheap):** re-confirm the Δret dissociation is arch-specific (OPT protects, inert Pythia
  ≈0). Lower information than Probe-7 now; the harness already supports the cell.

Artifacts: `gateA/probe6_displacement_opt350m.json`; harness `cloud/probe6_displacement_lambda.py`; prereg LOCK 185910cd.
Instance `6946f461` terminated clean (no orphan); cost ~$14. Related `VERDICT_PROBE5_RESIDUAL_2026_06_25.md`,
`VERDICT_DSI_2026_06_22.md`, `VERDICT_PROBE2_RANK_2026_06_23.md`.
