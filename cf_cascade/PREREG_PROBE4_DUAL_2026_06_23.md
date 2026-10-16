# PRE-REG — Probe 4: disjoint-subspace dual deposits. 2026-06-23. LOCK below.

**Question (pre-committed):** when two deposits occupy non-overlapping V-subspaces, do they REINFORCE, INTERFERE, or act
INDEPENDENTLY when the phase-2 surge arrives?

**Band (pre-committed):** OPT-350m, Pythia-410m ONLY (live band; dead at 1B).

**Matched dispositive regime:** lr_child 3e-5, lr_depth 5e-5 / 250 steps, held-out = subj, N=30 paired seeds,
V = top-128 task-A Fisher at w_A.

**Two childhood streams (pre-committed, LOCKED, disjoint by construction-attempt):**
- S1 = {agnews, dbpedia, yahoo}   S2 = {newscat, emotion}.
Each stream trained 400 steps. The disjointness of the resulting V-deposit subspaces is VERIFIED, not assumed (gate below).

**DISJOINTNESS GATE (HARD-STOP, pre-committed):** build S_dep1, S_dep2 (each = P_V(Δw) per carrier, top-r=8 directions).
Compute the per-carrier subspace overlap = mean top principal-angle cosine between span(S_dep1) and span(S_dep2). If the
carrier-median overlap > 0.35, the streams are NOT disjoint → the probe HARD-STOPs and reports the overlap (re-pick
streams next round). The reinforce/interfere read only counts behind a passed disjointness gate.

**Arms (per seed):**
- single1 = childhood S1 (400) → phase-2  →  B1−A
- single2 = childhood S2 (400) → phase-2  →  B2−A
- dual    = childhood S1+S2 interleaved (800 total) → phase-2  →  Bdual−A
- single800 = childhood S1 alone (800 steps, COMPUTE-MATCHED to dual) → phase-2  →  B800−A   (controls "dual is just more childhood")

**Reads (endpoint + per-step survival, per seed → mean ± SE):** B1−A, B2−A, Bdual−A, B800−A; per-deposit survival in the
dual arm (‖P_{S_dep_i}(w_t−w_A)‖ normalized, i=1,2) at phase-2 end; childhood_lossdrop per arm; disjointness overlap.

**Reproduce-anchor (GATES the read):** single800 B−A must be in the banked B−A band (OPT ~+0.089 / Pythia ~+0.073) within
0.03 (an 800-step single deposit ≈ the dispositive-scale deposit), else harness/regime invalid.

**Pre-committed outcomes:**
- **Reinforce** → Bdual−A > max(B1−A, B2−A) AND > B800−A; both deposits survive → two disjoint subspaces protect
  additively/super-additively.
- **Interfere** → Bdual−A < B1−A + B2−A (sub-additive) OR one/both deposits erode in the dual arm → the surge plays them
  against each other.
- **Independent** → Bdual−A ≈ B1−A ⊕ B2−A with both surviving, ≈ B800−A → deposits act without coupling.

**Controls:** compute-matched single800 (dual ≠ "just more training"); numerical disjointness gate (above); per-step
survival (interference may appear mid-phase-2 and wash out at endpoint — report the trajectory).

**Floor discipline:** all B−A are within-run paired (same seeds) → dose-to-arm contrasts floor-robust; demote absolutes.

Harness `cloud/probe4_dual_lambda.py` + launcher `cloud/lambda_launch_generic.py probe4`. CELL ∈ {opt350m, pythia410m}.
