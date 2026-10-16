# PREREG — Face-2: frozen↔plastic as DISCONTINUITY vs PHASE TRANSITION (continuous ‖Δθ‖ order parameter)

**Date:** 2026-07-18 · Extends SS17 (VERDICT_PHASE_RANK). SHA-lock before run.

## The hole SS17 left
SS17 swept **LoRA rank** and found the cosine DIRECTION law (frozen ρ_cos = −0.74) already dead by **rank-2** (within-B ρ = −0.068), calling the frozen→plastic transition a **sharp boundary**. But rank is a *broken* order parameter for the transition SHAPE: it's discrete (no rank-1.5), **rank-1 floored** (med_forget 0.014, unreliable) so the **sub-rank-2 region — where the transition actually lives — is unmapped**, rank↔forget-magnitude are entangled, and it was **single-seed**. "Sharp" really means "unresolved below ‖Δθ‖≈34."

## The question
Fix rank; use **‖Δθ‖ (travel-norm) as a CONTINUOUS order parameter**, walking the **real GD trajectory** via snapshots at increasing step counts. Does direction-governance exist at small plasticity?
- **H_linear (reframe):** the cosine law is the LINEAR-RESPONSE limit of forgetting. As ‖Δθ‖→0⁺, ρ_cos revives toward −0.74; as ‖Δθ‖ grows, 2nd-order disruptiveness dominates → ρ_cos→0. Crossover at ‖Δθ‖_c. SS17's rank-2 (‖Δθ‖≈34) was already nonlinear.
- **H_discontinuity (null):** no revival — ρ_cos≈0 the instant forgetting is nonzero. SS17's discontinuity holds at fine resolution; the frozen cosine law is a measure-zero (exact-frozen) property.

## Design
Reuse `probe_representational_law` (CIFAR-100 / ResNet18 conv-LoRA layer3+4, frozen-re-probe forgetting) — **same substrate as SS17**, directly comparable. Fix **rank ∈ {2, 8}** (2 = direct SS17 comparison + deepest linear-response probe; 8 = wider in-band coverage + capacity cross-check). Per (rank, seed, B-task): train the LoRA, **snapshot at steps {2,4,8,16,32,64,128,256,1000}** (a natural ‖Δθ‖ ladder along the true optimization path).

> **Amendment (post-smoke, pre-real-run, 2026-07-18):** the smoke showed ‖Δθ‖ rises fast (≈14 by step 32) and plateaus by ~256, so the original {8,16,…,1000} schedule under-sampled the small-‖Δθ‖ linear-response regime that H_linear is about. Snapshots moved EARLIER to {2,4,8,16,32,64,128,256,1000} to resolve ‖Δθ‖≈1–10 before committing to the H_linear/H_discontinuity read. This is a resolution change only; the pre-committed decision rule is unchanged. At each snapshot re-probe all 50 A-capabilities → per-(A,B) (cos(d_A,d_B), forget). B_LR=8e-3 (SS17 primary recipe). **3 seeds × 6 fixed B-tasks** → CIs on ρ_cos(‖Δθ‖) (fixes SS17's single-seed hole). cache→grid: training is the cost, snapshot re-probes are cheap.

Metrics per (rank, snapshot): mean ‖Δθ‖, mean feature-drift, ρ_cos pooled + within-B (per-seed mean → mean±sd across seeds), median forget, in-band flag [0.05,0.40].

## Pre-committed read (per rank, on in-band + measurable snapshots)
- **H_LINEAR SUPPORTED:** ρ_cos_within_B ≤ −0.30 at the smallest ‖Δθ‖ AND more negative there than at the largest ‖Δθ‖ by ≥0.15. → a linear-response window exists; the transition is continuous, not a boundary discontinuity. **Reframes SS17.**
- **H_DISCONTINUITY SUPPORTED:** ρ_cos_within_B > −0.20 at *every* measurable ‖Δθ‖. → dead as soon as motion is nonzero; SS17 holds at fine resolution. **Hardens SS17 into a true discontinuity.**
- **AMBIGUOUS:** neither — map it honestly.

## Priors (stated to lose)
SS17's rank result plus the "measure-zero frozen manifold" intuition lean me toward **H_discontinuity (~60%)**. But the linear-response framing is mechanistically clean and genuinely untested (nobody has looked below ‖Δθ‖≈34), so **H_linear (~40%)** is a live, respected, and *higher-upside* outcome — it would reframe the whole frozen↔plastic story. Falsifier for H_linear = no revival (ρ_cos > −0.30 at small ‖Δθ‖). I bet against my own more-interesting hypothesis, honestly.

## Cheap-first
SMOKE (rank 2, 1 seed, 2 B-tasks, snaps {32,256,1000}, PER_CLASS 40) validates: delta-scaling trajectory (‖Δθ‖ rises with steps), forgetting turns on, ρ computes, CIFAR loader works on the box — BEFORE the real ~$1–2 spend. Real fires on explicit go.

## Output
`out/phase_continuum.json` → verdict appended to `VERDICT_PHASE_RANK` (or its own VERDICT). ⚠️ watch the CIFAR/PIL ExifTags trap (Face-1b) — smoke will catch it; fall back to torchvision.CIFAR100 loader if it fires.
