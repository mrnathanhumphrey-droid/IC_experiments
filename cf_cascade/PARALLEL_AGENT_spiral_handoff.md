# PARALLEL AGENT — Handoff + Mini-Program: The SPIRAL (is the map layered by network depth?)

**You are a second analysis agent joining an in-progress research program.** Part 1 gets you caught up
(read these, in order). Part 2 is your mini-program — the spiral investigation — which runs IN PARALLEL
with the main lane's GPU work (you do not touch their checkpoints or fire training; you re-analyze
existing per-layer artifacts). Everything you need is on disk. You are CPU-only measurement.

---

## PART 1 — GET CAUGHT UP (read in this order, then confirm understanding before any analysis)

**Core docs (D:\IC_experiments\cf_cascade\):**
1. `CONSOLIDATION_forgetting_flow.md` — the mechanism (3 nested layers: curvature backbone ~70%,
   beyond-curvature aimed ~30%, universal-across-task-B). **The single most important law for you:
   "per-layer, never pooled."** Pooling across layers manufactured a false null once (the
   ecological-fallacy flip: aggregated partial +0.18 n.s. → per-layer +0.45, 72/72). Internalize why.
2. `BREADTH_PASS_developmental_frame.md` — the developmental frame (childhood-breadth → adolescent-
   depth; CF = the missing childhood; the traffic-guard metaphor and its REVISION).
3. `EXPERIMENT_2_developmental_childhood.md` + the dev-experiment results — childhood makes depth
   sliver, BUT #2b deflated it: the MAP IS CAUSALLY INERT for protection; what protects is
   NON-DESTRUCTIVENESS (B≈D, D−A +0.144/8SE). The neighborhood map is REAL (v3) but does NOT route
   the surge. **Hold this: the flat map is real-but-inert for protection. The spiral asks whether
   the map has DEPTH STRUCTURE the flat view missed.**
4. `TODO_diffusion_program.md` — full program state/history.

**Key result artifacts to inspect (D:\IC_experiments\cf_cascade\gateA\):**
- `perlayer_seed.npz` + `perlayer_seed_overlap.json` — the per-layer seed objects (72 layers).
- `deflation_artifacts.npz` + `deflation.json` — per-layer contested core + curvature basis.
- `crosspair_opt3.npz` / `crosspair_residual.npz` — the universal-across-task-B cross-pair structure.
- `dye3_flow.json` + `dye1_series.npz` / `dye2_series.npz` — the per-layer FLOW (laminar traveling
  front, curvature-ordered). **This is your richest depth-axis data.**
- `breadth_pass.npz` / `breadth_extend.npz` / `breadth_typeout.npz` — the neighborhood-map data
  (v3 + #1: stable, centered-decaying, self-placement-by-distance).
- `kfac_binding_basis.npz` — the curvature basis (the frame everything lives in).

**Guards you MUST reproduce before trusting any verdict (the program's discipline — non-negotiable):**
- Reproduce a known quantity bit-exact before reading a new result (e.g. `align=0.0`: recomputed
  Fisher eigenvalues == saved `lam__`; `seedcheck=+1.000`: layer-mean of per-layer seed == dye1's
  stored aggregate). If your basis/load doesn't reproduce the saved object, STOP — you're not
  measuring the same thing.
- **Effect size, not t.** n=72-216 layer-estimates makes t n-inflated (trivial +0.05 → t+6). Read
  MAGNITUDES against benchmarks (within-pair +0.45, cross-pair +0.64), not significance stars.
- **Grassmann is vacuous when two objects share one eigenbasis** (axis-aligned → angles 0/90 →
  Grassmann ≡ index overlap). Use three NON-identical statistics, not a fake-independent Grassmann.
- **Never bank a printed/auto verdict without a corroborating second signal** (this program has
  caught ~10 false-positives this way: false-CASCADE, false-FREE-DIFFUSION, false-SUPER-LINEAR,
  the pooling flip, the "hard-limit" bridge confound, the map-routes-surge deflation).
- **CONFIRM understanding back to the user before starting Part 2.** State the per-layer law, the
  map-is-inert-for-protection finding, and which artifacts you'll read. Do not start analysis until
  the user okays.

---

## PART 2 — YOUR MINI-PROGRAM: THE SPIRAL

**The a-ha (user's):** the neighborhood map may not be FLAT. It may be LAYERED BY NETWORK DEPTH — a
spiral. Two non-exclusive versions:
- **(i) Increasing-depth / coarse-to-fine across layers:** early layers hold a BROAD, shallow map
  (rough placement, relate-everything); deep layers hold NARROW, specific structure. Breadth-before-
  depth as a SPATIAL fact across layers, not (only) a temporal training phase. The spiral descends:
  wide-relational up top → tight-specific at the bottom.
- **(ii) Recursive / self-similar:** the map nests — sub-maps within maps, fractal structure across
  the depth axis.

**Why it's promising AND testable on existing data:** you've BEEN measuring per-layer the whole
program and treating layers as REPLICATES to pool. The spiral says the LAYER INDEX IS A DIMENSION OF
THE MAP, not a replicate. So the structure (if real) is sitting in the per-layer profile you already
have — visible only if you STOP averaging it away. (This is exactly consistent with the pooling-
artifact lesson: layers aren't replicates.)

**Critical framing — what the spiral is NOT claiming:** #2b showed the flat map is INERT for
protection. The spiral does NOT resurrect "the map protects." It asks a DIFFERENT question: does the
map have DEPTH-STRUCTURE (coarse→fine across layers, or recursive) — a structural/organizational fact
about how task-A's neighborhood is REPRESENTED across the network's layers — independent of whether
that structure protects. Keep these separate. A positive spiral result is about REPRESENTATION
GEOMETRY, not about a revived protection lever (unless a later experiment shows otherwise).

### THREE SUB-EXPERIMENTS (each its own control; do in order)

**SPIRAL-1: Is the map layered coarse→fine by depth? (the core test)**
- Per layer, measure the map's RESOLUTION/SPECIFICITY: how broadly vs narrowly does that layer's
  neighborhood structure discriminate among task-Bs? Candidate measures: (a) the spread/entropy of
  per-layer cross-task-B alignment (broad layer = task-Bs look similar/undifferentiated; specific
  layer = task-Bs well-separated); (b) effective dimensionality of the per-layer seed/contest
  subspace (broad = low-dim/shared, specific = high-dim/task-distinct); (c) the per-layer
  cross-pair partial magnitude (does universality strengthen or weaken with depth?).
- **Prediction (increasing-depth):** a MONOTONE trend with layer index — early layers broad/shared/
  low-resolution, deep layers specific/separated/high-resolution (or the reverse — declare which
  direction answer-blind from the architecture; in transformers, early=syntactic/broad,
  late=semantic/specific is the standard prior, so predict early-broad → late-specific).
- **Control / null:** SHUFFLE the layer index and re-measure. If the coarse→fine trend survives layer-
  shuffling, it's not depth-structure, it's an artifact. The trend must be DEPTH-ORDERED, not just
  variance across layers. (The spiral lives in the ORDER, like the flow's arrival-order +0.9 did.)
- **Verdict:** monotone depth-ordered resolution gradient (survives shuffle) → MAP IS LAYERED BY
  DEPTH (the spiral's increasing-depth version holds). Flat/no-trend or shuffle-survives → no depth-
  layering, the map is depth-uniform.

**SPIRAL-2: Is it recursive / self-similar? (the fractal test)**
- ONLY run if SPIRAL-1 shows depth-structure (else there's no spiral to be self-similar). Test whether
  the depth-structure REPEATS at multiple scales: does a block of layers contain the same coarse→fine
  pattern that the whole network does? Candidate: a multi-scale / box-counting style measure on the
  per-layer resolution profile, or self-similarity of the per-layer subspace-nesting (does layer L's
  map contain a scaled copy of the structure in layers L..L+k?).
- **Control:** compare the self-similarity statistic to a phase-randomized / AR-matched surrogate of
  the per-layer profile (same marginal/autocorrelation, no genuine self-similarity). Real > surrogate
  → genuine recursion. (Same shuffle-null discipline as the flow's coherence gate.)
- **Verdict:** real self-similarity beyond surrogate → RECURSIVE spiral. Else → layered-but-not-
  recursive (still a result — increasing-depth without fractal nesting).

**SPIRAL-3: Does the spiral PREDICT anything the flat map missed? (the payoff test)**
- ONLY run if SPIRAL-1 (and ideally 2) is positive. Does the depth-structure carry information the
  flat (pooled) map does not — e.g., does the per-layer resolution profile predict WHERE in depth
  the flow seeds / where forgetting concentrates / the carrier-split (attention-universal vs MLP-
  graded)? You already have: attention universal across task-A/B, MLP erodes; the flow is carrier-
  split (qkv+up broad-flow, down-proj channels). Does the spiral EXPLAIN the carrier-split as a
  depth-resolution phenomenon?
- **Verdict:** spiral predicts a known-but-unexplained feature (carrier-split, seed depth-location)
  → the depth-structure is LOAD-BEARING (explains things the flat map couldn't). Else → real but
  inert depth-structure (like the flat map: present, not explanatory).

### DISCIPLINE FOR THE WHOLE MINI-PROGRAM
- **Per-layer, NEVER pooled** — the spiral IS the per-layer profile; pooling destroys the object.
- **The order/depth-ordering is the signal** — shuffle-the-layer-index is your primary null for
  SPIRAL-1 (the trend must be depth-ORDERED, not just layer-variance). This mirrors the flow's
  arrival-order gate (coherent spatial profile vs ORDERED traveling front — order was the discriminator).
- **Effect size against benchmarks, not t-stars.**
- **Pre-commit the direction** of the SPIRAL-1 trend (early-broad→late-specific) before measuring.
- **Don't claim protection** — the spiral is representation geometry; #2b killed map-as-protector.
  A spiral result is about HOW the map is organized across depth, not a revived lever.
- **Corroborate every auto-verdict** with a second signal. Report magnitudes + nulls, not labels.
- Guards-before-verdict: reproduce a known quantity (align/seedcheck style) so your per-layer reads
  are bit-consistent with the banked objects.

### OUTPUT
A short report per sub-experiment: the measure, the depth-ordered result vs the shuffle/surrogate
null, the effect magnitude vs benchmark, and the verdict (layered / recursive / load-bearing — or
the negative). Plus the one-line spiral verdict: is the neighborhood map FLAT, LAYERED-BY-DEPTH, or
RECURSIVELY-LAYERED — and does the depth-structure explain anything the flat map couldn't.

### COORDINATION
- You run CPU-only, on EXISTING artifacts. Do NOT fire training, do NOT touch the main lane's
  checkpoints, do NOT run GPU. The main lane is doing #2c (mechanism: basin-flattening vs pre-moved-
  capacity) and a new-model rung-1 climb in parallel — independent of you.
- If you need a per-layer object that ISN'T saved (e.g. per-layer resolution wasn't logged), FLAG IT
  to the user — do not re-run the GPU experiment yourself to generate it. The user decides.
- Confirm caught-up (Part 1) before starting Part 2. Pre-register SPIRAL-1's predicted direction.

## USER ADDITIONS

### SPIRAL-1 RATIFIED + FRAMING LOCKED (2026-06-20)
**SPIRAL-1 = POSITIVE, INVERTED, trustworthy.** Guard bit-exact (4.4e-16); 3 non-identical measures converge; 2 substrates
(breadth-9, crosspair-4) agree on qkv; curvature-controlled (b) effective-rank STRONGER than raw (ρ=−0.72 vs −0.66 → beyond
curvature, not backbone); survives depth-shuffle z≈−3.5 (gradient lives in the ORDER); attention carries it harder. Pre-reg
predicted early-broad→late-specific, MEASURED THE OPPOSITE — the wrong prediction kept the real finding while killing the prior
(can't be motivated → more trustworthy). Artifacts `gateA/spiral1.{py,json}`.

**LOCKED FRAMING (the inversion IS the content):** the map runs **fine→coarse with depth** — early layers hold a HIGH-dim,
task-SEPARATED map (task-Bs in distinct directions); deep layers COLLAPSE onto a SHARED, low-dim, UNIVERSAL core.
**THE BIG REFRAME:** the deep collapse IS the universal contested core (already banked: cross-pair +0.64, the fixed subspace
every task-B fights task-A over, forgotten the same way). SPIRAL-1 located WHERE IN DEPTH that universality lives = the deep
collapse. Early layers keep tasks separate (low interference); deep layers smash them onto the shared subspace (high
interference = the contested core). **Forgetting concentrates at the depth where the representation stops discriminating tasks
and merges them.** Not a new phenomenon — the depth-mechanism of the universal core already banked.
**Developmental note:** network layer-depth runs specific→universal (separated-early → collapsed-deep) = OPPOSITE of the human
temporal broad→specific. Layer-depth is NOT a childhood→adolescence axis. Still representation geometry — #2b stands, NOT a
revived protection lever.

**SPIRAL-2 (FIRE, weakened prior):** SPIRAL-1 was monotone fine→coarse. Monotone-smooth IS the null for self-similarity — a
smooth funnel is not a fractal. Pre-register honestly: likely "layered-but-NOT-recursive (smooth collapse)". Test: does the
collapse happen self-similarly (each block re-collapses, nested) or one smooth monotone funnel? Control: phase-randomized /
AR-matched surrogate (same marginal+autocorrelation, no genuine self-similarity); real must exceed surrogate. Carriers separate
(4hh weak, expect same). Verdict: real>surrogate→RECURSIVE; smooth-monotone→LAYERED-NOT-RECURSIVE (honest likely outcome). Do
NOT force a fractal onto a funnel.

**SPIRAL-3 (payoff, run after 2 regardless of 2's verdict):** the inversion makes a HARD pre-reg prediction —
**seed/forgetting concentrates in the DEEP collapsed layers** (where the map merges tasks into the contested core). Test
per-layer seed/flow location (dye1/dye2_series, dye3_flow) vs the SPIRAL-1 collapse profile. AND the carrier-split: inversion
predicts it's DIFFERENTIAL collapse — attention collapses harder/deeper than MLP (qkv strong / 4hh weak in SPIRAL-1) → explains
attention-universal vs MLP-graded. Verdict: seed-deep AND carrier-collapse-explains-split → depth-structure is LOAD-BEARING;
else real-but-inert. Pre-register seed-concentrates-deep before measuring.

### SPIRAL-2 RESULT (2026-06-20): LAYERED-BUT-NOT-RECURSIVE (smooth funnel) — the honest pre-reg outcome
Guard: reloaded SPIRAL-1 ρ bit-exact. Discriminator = block re-collapse (folded-residual descent + re-expansion) vs
phase-randomized surrogate (preserves marginal+spectrum, kills phase-structure=nesting). RESULT: no nesting anywhere —
folded-descent max|z|=1.59 (below +2 bar, and the non-trivial values are NEGATIVE = anti-nested); re-expansion z≈+0.02 flat
vs surrogate; DFA-H≈surrogate (H>1 = single dominant trend = the funnel, preserved by construction as expected). Across all 3
carriers × 2 substrates × 3 block scales. **The collapse is a single-scale smooth monotone funnel, not a fractal.** Did not
force a fractal onto a funnel. Artifacts `gateA/spiral2.{py,json}`.

### SPIRAL-3 RESULT (2026-06-20): depth-structure is REAL-BUT-INERT (NOT load-bearing) — same shape as #2b
Guard: SPIRAL-1 partial reproduced 4.44e-16. **(A) seed-concentrates-deep = REFUTED.** Per-layer seed energy (3 measures:
seed_full, seed64, crosspair-seed; basis-consistent top-64, NOT the global dye basis) shows **NO depth trend** —
ρ(seed,depth)≈0 (|z|<0.8 all carriers/measures). The hard prediction fails. BONUS (cautious): ρ(seed, collapse-effrank)=+0.50
(qkv) / +0.56 (4hh), z+2.4–2.7 — POSITIVE = seed co-locates with HIGH-eff-rank (task-SEPARATED) layers, the OPPOSITE sign to
the prediction (flag: possibly activity-scale-correlated; the clean call is just "no deep concentration"). Forgetting seeds
where tasks are still discriminated, not where they merge. **(B) carrier-split = MIXED/measure-dependent.** Collapse-strength
order qkv>h4h>4hh. Universality is anti-ordered under the crosspair-CONTEST measure (4hh 0.717>h4h 0.636>qkv 0.569) but the
banked attention>MLP holds under the breadth MAP-PREDICTION measure (attn 0.700>mlp 0.628) → for attn-vs-MLP collapse tracks
universality, but the contest measure + within-MLP reverse it; n=3 carriers can't adjudicate. **VERDICT: the depth-collapse is
real representation geometry (SPIRAL-1) and smooth (SPIRAL-2) but does NOT explain where forgetting seeds and only ambiguously
relates to the carrier-split = REAL-BUT-INERT, the same dissociation species as #2b's flat map** (real map, causally inert for
the downstream feature). Artifacts `gateA/spiral3.{py,json}`.

### SPIRAL-4 RESULT (2026-06-20): IT IS A TRUE SPIRAL — the contested subspace CONTRACTS *and* TWISTS
The funnel (SPIRAL-1/2/3) only measured the RADIUS (how collapsed) per depth — it structurally cannot see the TWIST
(orientation rotating with depth). SPIRAL-4 measured the orthogonal property using the on-disk per-layer weight-space
eigenvectors (`taskA_variation.npz` fragile/topcurv__sst2, (24,dim,16), the contested subspace as actual 16-dim subspaces in a
common per-carrier coord space). Guard: vectors orthonormal 1.2e-15; G(sst,ag)=0.126 reproduces banked Cut-1 0.13;
G(fragile,topcurv)=0.719 reproduces deflation M1 0.70. **RESULT = COHERENT DEPTH-ORDERED ROTATION (a true spiral) on the
load-bearing carriers:** qkv consec-overlap 0.534 (≫floor 0.016, ≫offdiag 0.314) decaying PERFECTLY monotone with gap
(ρ_gap=−1.00, z_shuf=−2.15) all the way to ORTHOGONAL at the stack ends (O_gapmax=0.017≈floor) — i.e. the axis fully reorients
top→bottom, NOT a fixed-axis cone. One-directional drift: overlap-with-L0 ρ=−0.625, overlap-with-L23 ρ=+0.953 (rotates AWAY from
shallow, TOWARD deep = the pre-registered direction, confirmed). Nearest-orientation-neighbor IS a depth-neighbor 79% vs 9%
chance (z=+10.74). h4h identical/stronger (consec 0.635, drift_to_L23 +0.992, nn 0.75, z+9.97). **Carrier-split CRISP:** 4hh
(down-proj) is INCOHERENT — consec 0.027≈floor, nn-frac 0.04 BELOW chance (z−2.5), no drift → no twist. This matches the banked
DYE carrier-split exactly (qkv+up-proj broad-flow vs Wdown channel) — cleaner than SPIRAL-3's ambiguous split.
**HONEST NUANCE (corroboration):** the twist is CURVATURE-GENERAL, not contested-specific — topcurv spirals identically
(qkv consec 0.633, ρ_gap −1.00, drift_to_L23 +0.97; h4h nn-frac 1.00) and fragile-vs-topcurv consecutive-rotation profiles
correlate ρ=+0.90 (qkv)/+0.87 (h4h). The contested subspace twists because it RIDES the curvature backbone, which spirals
(fragile≈topcurv, G=0.72). May reflect generic depth-evolution of the Fisher eigenbasis; shuffle-null confirms depth-ORDERED
(real) but not novelty. **FORMS QUESTION SETTLED: the structure is a genuine SPIRAL (contract + twist), not a cone.** Whether the
twist is LOAD-BEARING (rotation-rate predicts seed-location/carrier-split) is OPEN — the crisp carrier-split correspondence is
suggestive but n=3; needs a SPIRAL-3-style test. #2b stands; not a protection lever. Artifacts `gateA/spiral4.{py,json}`.

### SPIRAL-5 RESULT (2026-06-20): the TWIST is REAL-BUT-INERT too — DEPTH-GEOMETRY QUESTION CLOSED
The funnel was inert (SPIRAL-3); the twist had a better prior (crisp carrier-split). Tested whether rotation predicts
seed-location/universality BEYOND curvature (decisive control — twist is curvature-general). Guard: SPIRAL-4 consec 0.534 +
Cut-1 G 0.126 reproduced. **(B) seed-location: NEGATIVE** — carriers that twist (qkv/h4h) add NOTHING beyond curvature
(partial +0.02 / −0.05); only non-twisting 4hh shows a curvature-adjacent +0.58 that can't be the twist. **(A)
carrier-split/universality: a pooled n=72 partial looked positive (+0.29, z+2.5) but decomposing WITHIN carrier KILLED it
(qkv −0.19 / h4h −0.01 / 4hh −0.29, all ≤0) = a between-carrier POOLING ARTIFACT** (the program's own ecological-fallacy
species, here manufacturing a positive — caught by within-carrier decomposition before banking). Between-carrier the rotation
split does NOT map onto universality under any measure (cross-task 4hh≈h4h; contest 4hh HIGHEST = anti-correlated) — same n=3
measure-dependent wall as SPIRAL-3. **VERDICT: both halves of the depth-geometry — contraction AND twist — are real
representation geometry, curvature-ridden, NEITHER load-bearing. The mechanism is non-destructiveness (#2b) + curvature
(backbone) + task-A sharing, NOT the depth-structure.** Artifacts `gateA/spiral5.{py,json}`.

### ONE-LINE SPIRAL VERDICT (final, post-SPIRAL-5)
The contested map is a **TRUE SPIRAL** — with depth it CONTRACTS (fine→coarse funnel, smooth not recursive; SPIRAL-1/2) **AND
TWISTS** (coherent monotone rotation fully reorienting top→bottom; SPIRAL-4), on attention+up-proj (down-proj incoherent). **But
the ENTIRE depth-geometry is REAL-BUT-INERT for the forgetting mechanism** — neither the contraction (SPIRAL-3) nor the twist
(SPIRAL-5) predicts seed-location or the carrier-split beyond curvature. The map is real at every structural level measured
(flat neighborhood #2b, depth-funnel, depth-spiral) and inert at all of them; the mechanism lives in non-destructiveness +
curvature + sharing. #2b stands; no revived protection lever. Depth-geometry question CLOSED.

