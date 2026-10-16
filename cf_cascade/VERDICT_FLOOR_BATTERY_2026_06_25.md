# VERDICT — B−D reproducibility-floor battery. 2026-06-25.

Pre-reg `PREREG_FLOOR_BATTERY_2026_06_23.md` (LOCK f1dfa005). Noise characterization, not mechanism. All blocks ran on
a100_sxm4, fp32 or bf16 as specified, with per-step heartbeat + stall-detecting killswitch (the monitoring that was
missing on 2026-06-23). Per-cell, never global. Reproduce-anchor (w_A) held exactly on every block (OPT 0.823, Pythia
0.753).

## The blocks (all landed)
| block | OPT-350m | Pythia-410m |
|---|---|---|
| bf16 hardware floor (5 same-seed repeats) | SD **0.0165** (repeats .059/.023/.029/.017/.039) | SD **0.0000** (5× bitwise-identical −0.0216) |
| fp32 hardware floor (5 same-seed repeats) | SD **0.0087** (.037/.026/.018/.038/.035) | (not run; confirmatory — bf16 already 0) |
| seedsample (5 disjoint N=30 draws) | mean **+0.0128**, SD **0.0168** (.014/.020/−.015/.030/.015) | mean **−0.0159**, SD **0.0159** (−.006/.004/−.033/−.014/−.031) |
| seedcurve (N=100) | mean **+0.0062 ± 0.0102**, per-seed SD **0.1005**, SE<0.02 at N=30 | — |
| drift (a10, platform-confounded) | +0.0066 (a10≠a100 → discard as env-drift measure) | killed (a10 hung) |

## FINDING 1 — the OPT "+0.055" was a low-N seed-luck artifact
Properly seeded, OPT B−D = **+0.006 (N=100) / +0.0128 (5×30)** — i.e. **~+0.01, not +0.055.** Per-seed B−D SD is 0.10,
so a 30-seed mean has SE ≈ 0.018; "+0.055" sits ~3 SE above the true ~+0.01 = a high draw of the first 30 seeds. The
absolute claim "OPT is load-bearing, B−D = +0.055" does not reproduce. OPT is *barely* positive.

## FINDING 2 — the dissociation SURVIVES, at ~half the magnitude, marginally significant
OPT (+0.01) is genuinely above Pythia (−0.016). Dissociation OPT−Pythia:
- apples-to-apples (both seedsample, 150 seeds): +0.0128 − (−0.0159) = **+0.0287 ± 0.0103 → 2.8σ**
- conservative (N=100 OPT +0.006 vs Pythia seedsample): **+0.022 ± 0.012 → 1.8σ**

So the dissociation is **+0.022 to +0.029, ~1.8–2.8σ — real but modest/marginal**, NOT the banked +0.061 (+0.055 vs −0.006).
The headline magnitude drops ~2×. It is not noise (Pythia is reliably negative, OPT reliably ≥0), but it is a small effect
that N=30 was under-powered to pin.

## FINDING 3 — the "0.066 floor" resolved, no unnamed source
Per-cell floor decomposition (per 30-seed estimate):
- OPT: hardware (bf16 0.0165 / fp32 0.0087) ⊕ seed (0.0168) → **total ~0.024 (bf16) / ~0.019 (fp32)**
- Pythia: hardware ~0 ⊕ seed 0.0159 → **total ~0.016**
The difference between two independent 30-seed runs has SD ≈ √2 × 0.024 ≈ 0.033; a one-off observed **0.066 is a ~2σ tail
pair**, not the 1σ floor. RSS reconstructs the real floor (~0.02), and the prereg's "reconstruct 0.066" target was itself
anchored to a tail. **No unnamed source.**

## DECISION GATE (pre-committed) → BOTH, roughly equal
Not bf16-dominated, not seed-dominated: hardware (~0.0165 bf16) and seed-sampling (~0.0168) are **equal** on OPT. fp32
**halves the hardware half** (0.0165→0.0087) but the seed half is irreducible. So **the fix is N, not precision** — fp32
helps a little; resolving a small B−D needs N≥100 (or paired within-run differentials). Heteroskedastic as predicted:
Pythia has zero hardware floor, OPT has a real one.

## What this VINDICATES (the reconciliation)
The fragile quantity is exactly the one the program always flagged as fragile — the **absolute** B−D level. The findings
banked as **paired within-run differentials** are precisely where this floor CANCELS:
- **DSI swap-gate** (real-B inV vs Bswap inV, same run/seeds) — `VERDICT_DSI_2026_06_22.md`
- **Probe-2 rank** (Bk vs B, same childhood/seeds) — `VERDICT_PROBE2_RANK_2026_06_23.md`
These STAND. The mechanism program is **not** mechanizing noise — it explains a real (~+0.027, ~2σ) dissociation. But its
framing must change from "explaining a +0.06 load-bearing effect" to "explaining a +0.03 dissociation," and every quoted
absolute-B−D magnitude must be re-read against the ~0.02 floor.

## GO-FORWARD (binding on future B−D work)
1. **fp32 + N≥100, or paired within-run differentials** — for ANY absolute B−D claim. N=30 is under-powered below ~0.05.
2. Probes 3/4/5 must use paired differentials and/or N≥100, not N=30 (re-spec before firing).
3. Re-read DSI / Probe-2 / depth-ladder magnitudes against the ~0.02 floor; demote any absolute level inside it.
4. Optional clean-ups (low priority): a100 drift re-run to isolate env-drift from the a10 platform difference; Pythia fp32
   (confirmatory).

Artifacts: `gateA/floor_battery_{opt350m,pythia410m}_{bf16,fp32,seedsample,seedcurve}.json`. Harness
`cloud/floor_battery_lambda.py` (instrumented). Cost ~$70 across the battery; a10 detour (2026-06-23) wasted ~a day —
see [[feedback_size_gpu_to_workload_not_cheapest]].
