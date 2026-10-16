# VERDICT — SPIRAL-OPT-1: Pythia depth-funnel does NOT replicate on OPT-350m. 2026-06-21. Parallel agent.

Pre-reg `PREREG_SPIRAL_OPT_2026_06_21` (lock `eadc8832`). CPU-only on existing OPT artifacts
(`breadth_pass_opt.npz`, `crosspair_residual_opt.npz` — faithful OPT swaps of the Pythia generators). Script `opt1.py`,
result `gateA/spiral_opt1.json`. Six tripwires bound the run.

## VERDICT (cross-arch fork): FUNNEL DOES NOT REPLICATE — architecturally different depth-geometry.
OPT's neighborhood map is NOT structureless, but it is **differently organized** than Pythia's. The spiral's reach is
bounded: the funnel signature is Pythia/GPT-NeoX-family-specific, not universal. The morning's close **holds at parity**.
Downstream legs (OPT-4 twist, OPT-3 seed-load-bearing, OPT-CARRIER) are **gated OFF** — they presuppose Pythia's funnel
as the structural prior; without it, twist/seed-load-bearing change meaning. (User decision 2026-06-21: bank + halt.)

## Two corroborating rows (TRIPWIRE 3; per-carrier, never pooled; curvature-controlled = clean read)
**Row 1 — Pythia's signature measure (eff-rank collapse, the "ρ−0.72") is ABSENT on OPT.**
- Pythia (replication target): qkv `b_effrank_curvres` ρ=−0.72 z=−3.47; h4h ρ=−0.65 z=−3.15. (Strong collapse.)
- OPT: every carrier |z|<2, mixed sign — q_proj −0.36/z−1.76, k_proj −0.04 & −0.38, v_proj +0.06/+0.04,
  out_proj −0.41/z−1.92 (closest, still ns), fc1 +0.15 & −0.16, fc2 **+0.51**/z+2.48 (breadth/crosspair). No OPT carrier
  reproduces the collapse.

**Row 2 — where OPT has strong depth-structure it is OPPOSITE-sign and on different carriers.**
- OPT MLP resolution+universality CONTRACT hard with depth: fc1 `a_meanpairsim_curvres` ρ=−0.84 z=−4.02 (crosspair),
  −0.73 z=−3.54 (breadth); fc2 −0.74 z=−3.62 / −0.74 z=−3.51; (c)-universality fc1 ρ=−0.51 z=−2.43, fc2 ρ=−0.66
  z=−3.15 (p=0.0002).
- Pythia's MLP did the OPPOSITE: h4h `a_curvres`≈0 (+0.02/−0.15), h4h (c)=+0.57 (UP). So OPT's strong signal is exactly
  where/how Pythia had none — different carriers (fc1/fc2 vs qkv/h4h), opposite direction.

## Per-carrier summary (never pooled)
| | Pythia funnel | OPT |
|---|---|---|
| signature | eff-rank collapse (b), qkv ρ−0.72 / h4h −0.65 | (b) absent, all \|z\|<2, mixed sign |
| strong OPT signal | — | (a)-resolution + (c)-universality contraction, MLP fc1/fc2 ρ−0.74..−0.84 z−3.5..−4.0 |
| attn | qkv carries (b) | q/k weak-DOWN on (a); v/out anomalous |

## GUARD (reproduce-known-quantity)
My (c) computation (from OPT npz contest arrays, since `crosspair_residual_opt.json` report errored) reproduces
Pythia's cached `partial_per_layer` to **8.9e-16** before any OPT measurement. (c) definition is faithful.

## AMENDMENT (TRIPWIRE 4 — error caught in my own pre-reg, recorded transparently)
The locked `PREREG_SPIRAL_OPT` (eadc8832) stated the OPT-1 direction as **naive coarse→fine** (a-sim DOWN, eff-rank
UP, c DOWN). That is WRONG: the prompt asked to predict Pythia's **inverted** fine→coarse, and Pythia's actual
signature is (b) eff-rank **DOWN** (−0.72), (a)-sim **UP**, (c) **UP**. I encoded the `spiral1.py` docstring's naive
comment instead of reconciling against Pythia's real result. The replication verdict is unaffected — it is OPT-vs-
Pythia-ACTUAL, re-graded against the true target above — but the locked direction line is corrected here. The pre-reg
file is left byte-intact so its lock honestly records the mis-statement; this verdict is the correction of record.

## RECONCILED OPT-vs-Pythia differences (TRIPWIRE 4, all flagged pre-run)
6 OPT carriers (q/k/v/out_proj,fc1,fc2) vs Pythia 3 (fused qkv,h4h,4hh); 3 crosspair task-Bs (no 'code'); breadth
index parsed from `names` (carrier order k,v,q,out,fc1,fc2); (c) from npz (json errored), guard-verified.

## SCOPE / what this settles
Cross-arch parity: the SPIRAL depth-funnel (eff-rank collapse + twist) is NOT a universal property of fragility-carrying
architectures — it is GPT-NeoX/Pythia-family-specific. OPT has its own (opposite-sign, MLP-localized) depth structure;
whether THAT is load-bearing was NOT tested (downstream gated off by user decision). Bounds the spiral's reach.
