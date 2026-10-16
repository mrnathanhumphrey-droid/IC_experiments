# PRE-REG — Hodge harmonic component of task interference = subspace holonomy (cyclic done right)

- **Date:** 2026-07-27
- **Status:** LOCKED — pre-registered. Not fired.
- **LOCK:** `0d49caf9` (sha256[:8] of draft body at lock, 2026-07-27T00:06; lock order 2 of 4)
- **Arc:** interference_law (extends Face-1c cyclic machinery)
- **Priority:** #1 of the 2026-07-27 subspace/eigenspace audit (★★★★★ — best novelty × cheapness × program-fit × has-a-floor)

## One-line hypothesis
The **harmonic (curl) component** of the pairwise interference matrix — the cyclic residual Face-1c found (`face1c_*` : rock-paper-scissors forgetting triples) — corresponds to **non-commuting rotations between task gradient subspaces (holonomy)**. For a cyclic triple (A,B,C), the composed principal-angle rotation `R_AB ∘ R_BC ∘ R_CA` deviates from the identity, and this **holonomy deviation** correlates with the curl magnitude `|curl_{ABC}|` on that triple.

## Why now / the wedge
- Lit re-confirmed (2026-07-27): **ZERO prior art bridging HodgeRank ↔ catastrophic forgetting.** HodgeRank lives in game-ranking (Balduzzi NeurIPS'18, 2601.07158); task-ordering lives in CL (2510.19941, 2604.13460 spectral-but-forward-time-only, 2510.09181 explicitly pairwise/transitive). Nobody has applied Helmholtz–Hodge to a neural interference matrix to isolate a cyclic component **or** given it a subspace mechanism.
- **Program owns the machinery:** Face-1c already built the curl/gradient Hodge split + placebo controls (RANDTEXT/NOISE) that proved curl is **task-specific**, not pipeline/SNR noise. This thread turns the fragile "cycles exist" result (real but Qwen×banking77-specific) into an **eigenspace mechanism**: *why* are cycles pairwise-invisible? Because the loop of subspace rotations doesn't commute.

## Falsifier (pre-registered kill conditions)
- **PRIMARY KILL:** across triples, `|Spearman(holonomy_deviation, |curl_{ijk}|)| < 0.30` (perm-null p>0.01, above the Haar null band) → the harmonic component has **no subspace signature** → curl is unstructured residual → **closes Face-1c** as "cyclic real but geometrically meaningless." Legit honest negative.
- **CONFOUND KILL:** if holonomy-curl correlation vanishes once `|curl|`'s covariance with per-triple forgetting *magnitude* is partialled out → curl-holonomy link is a magnitude artifact.

## Priors to lose (pre-register against)
- 2510.09181 (interference is **pairwise/transitive**, adversarial alignment with top old-task curvature) → predicts curl≈0 and no cyclic subspace structure. A holonomy HIT sides against the transitive-only camp.
- 2604.13460 (spectral characterization, forward-time operator, no cyclic term) → predicts no harmonic component.

## Must-beat baselines
1. **Magnitude baseline:** does `|curl|` merely track total forgetting on the triple? (partial out).
2. **Haar random-rotation holonomy null:** random high-dim subspaces have non-zero composed-rotation deviation → the QX1 finite-N Haar null band (`float_module` technique) sets the real bar, not zero.
3. Scalar min-principal-angle sum around the loop (does the *magnitude* of angles, ignoring non-commutativity, already explain curl? holonomy must beat it).

## Method
- **Substrate:** reuse Face-1c cached fine-tune grid — **Qwen2.5-1.5B × banking77** (the corner where curl was strongest, curl-ρ ~0.37). No new fine-tuning; cache→grid.
- Build antisymmetric interference matrix `M_ij` (directed-F); run the existing Helmholtz–Hodge least-squares split → gradient (transitive) + curl (harmonic).
- Per task, extract top-k **gradient subspace** basis `U_i` (Grassmann point). For each consecutive pair in a triple, compute the principal-angle rotation aligning `U_i→U_j`; compose around the loop; measure **holonomy deviation** = geodesic distance of the composed operator from identity (or `‖log(R_AB R_BC R_CA)‖`).
- Correlate holonomy deviation with `|curl_{ijk}|` across all triples.

## Confound controls (pre-registered)
- **Projection-dependence** (Face-1c lesson: low-d projection can induce cycles) → sweep projection dims {8, 24, 48}; holonomy-curl link must persist, and (per Face-1c) be **strongest at high dim** to rule out low-dim artifact.
- **Curl SNR:** weak-forget cells (0.02–0.06) → use the strongest cell only; report medFgt.
- **Haar null band** for holonomy (mandatory — see baselines).

## Stats
Cross-seed reproducibility gate (curl-ρ≥0.30, already established in Face-1c) as entry condition; perm-null on the holonomy-curl correlation; Haar finite-N null band; report over 3 seeds.

## Cost / lane
~$2–4 Lambda (cached grid + new geometry compute is cheap). SMOKE the holonomy computation on one seed first.

## Standing floor (publishable regardless)
- **Null:** closes the dangling Face-1c ambiguity (is the ~0.20–0.37 curl weak-real-but-meaningless or structured) — a clean result either way.
- **Hit:** first mechanism for cyclic/non-transitive interference = **subspace holonomy**, in a literature that has zero HodgeRank↔forgetting prior art. Fuses directly with PREREG_ANGLE_SPECTRUM_ASYMMETRY (#2) — same subspaces.
