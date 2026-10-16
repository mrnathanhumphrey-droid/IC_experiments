# The Breadth-Pass / Missing-Childhood Frame for Catastrophic Forgetting

**Status:** conceptual frame + lit-map + **v1/v2/v3 RESULT (2026-06-20): the map is REAL and GRADED BY DISTANCE FROM TASK-A.** The
§5 breadth-pass experiment ran (breadth_pass_lambda.py): a map fit from a stream of 9 task-Bs' step-0 seed geometry predicts a
HELD-OUT task-B's pull at curvature-controlled partial 0.652, decomposing into ~0.40 generic model geometry (randtok floor 0.401) +
~0.16 language (nltext 0.561) + a task-specific lift. ★ v3 leave-one-TYPE-out (the decisive press — is the lift cross-type or just
within-type near-duplicates?): cross-type held-out 0.627 (vs language floor 0.561) does NOT collapse; within-type wins (same 0.743 vs
cross 0.650) BUT cross-type generalization is REAL and GRADED by distance from task-A (sentiment) — topic/subj/emotion/sentiment above
floor, code≈floor, ucdp(violence-events) BELOW floor. The neighborhood is CENTERED ON TASK-A and DECAYS with structural distance =
exactly the developmental shape (a childhood map helps NEAR what it saw; novel task-kinds still yank). Bounds #3 to the neighborhood;
a BROADER childhood would extend it. **★★★ #2 DEVELOPMENTAL EXPERIMENT DONE (2026-06-20) = CLAIM LANDS (in-neighborhood): a net
given a (structured) childhood SLIVERS where one without it YANKS, and it's the MAP not the mileage.** Held-out=subj; childhood =
non-sentiment neighborhood (no task-A replay). SST retention A 0.527 / B(real childhood) 0.653 / C(scrambled, matched compute) 0.538:
B−A=+0.126±0.024 (~5SE), B−C=+0.115±0.025 (~4.6SE), task-B not starved → genuine sliver; childhood made depth 3× less destructive.
Scramble verified by loss (real→0.79, scrambled floored 2.33≈log#labels). CAVEAT: master-none partially violated (childhood loss
0.79 = substantial learning → "structured pre-exposure" not "shallow"; C matches compute so structure-not-compute holds); scrambled
control is noise-fitting (a non-destructive no-map control would tighten); single net/task-A/heldout, in-neighborhood only. Full
record: memory APEX block + CONSOLIDATION_forgetting_flow.md. **★★★ #2b NON-DESTRUCTIVE NO-MAP CONTROL DONE (2026-06-20) = DEFLATION:
the MAP does NOT route the surge.** Arm D (same stream as B but childhood grads projected off task-A's top-128 curvature subspace →
non-destructive + no-map by construction) slivers AS WELL AS B (B−D=−0.018±0.018, B≈D; phase-2 damage B=D=−0.070 identical; D−A=
+0.144). Guards passed (D non-destr 0.741≥B 0.722, no-map mapfrac 0.018 vs B 0.341, trained drop 15.3). → #2's "it's the map" is
RETRACTED to "it's NON-DESTRUCTIVENESS not the map" (B>C was C's noise-fit destructiveness). The v3 map is REAL but CAUSALLY INERT
for forgetting-protection. **The traffic-guard-WITH-A-STREET-MAP picture (Mode 2 routing) is DEFLATED — the protection is a
non-destructive warm-up, not route-knowledge.** What survives: any non-destructive prior phase (map or map-free) → task-A ~3× robust
to depth. NEW open Q: the SUB-MECHANISM of that map-free protection (basin flattening? pre-moved complement capacity?). **NEXT =
mechanism of Mode-1 protection · Option 2 (2nd net) · master-none-enforced re-run · broader-childhood spanning-bridge.** Generated in conversation 2026-06-20.

**One-line:** Catastrophic forgetting may be what you get when a network is forced through "adolescence" (depth on one task) with no "childhood" (a low-resolution breadth pass that builds a relational map of the task-neighborhood). The breadth map is the prerequisite that lets new depth *sliver in beside* old knowledge instead of *yanking it out of place*.

---

## 1. The metaphor chain that generated this (each link became a testable claim)

The frame was reached by a sequence of metaphors-as-discovery-instruments, not by reading the CF literature. The chain:

- **Ink in water** → forgetting is a *flow* (diffusion from a localized seed), not a cascade or static concentration. (Confirmed earlier in this program: laminar traveling front, gated.)
- **Thread that yanks vs slivers** → the human thread, pulled by a new demand, *slivers* (splits a close-but-distinct alternate strand that COEXISTS). The network thread can't sliver — it has a fixed set of threads and only one operation: *relocate* the existing thread. The old position is vacated (recoverable, but not coexisting). Forgetting = the absence of a sliver operation.
- **Second spool of thread** → slivering requires *new* parameters that belong to task-B (adapters / LoRA / PackNet / MoE). The characterization already gives the spool's *size* (~10-16 dim contested core) and *location* (the seed, the steep contested attention directions). A *fitted* spool, not a generic one. It only has to catch the seed — divert the source and there's nothing to flow.
- **The traffic guard (after a Knicks win)** [Wilson's metaphor, 2026-06-20 — REPLACES "the seamstress"] → WHO directs the surge? Knicks win = the new task flooding in all at once (the yank's suddenness). The arbiter (#3) doesn't *build* anything (the seamstress framing implied growing new tissue/a spool = allocation); she **routes existing flow in real time** so the streams don't pile up at the intersection = **arbitration, not construction.** #3 reframes from *allocate new parameters* → *route the gradient flow already there, around what's occupied.* The network has no traffic guard — only gravity (gradient) and the road grid (weights); every working method has a HUMAN guard (researcher hand-places the adapter). The unsolved thing (#3) is the network routing for itself. (Roads = curved/load-bearing directions — flat dirs aren't streets, routing there does nothing = the residual negative; the contested core, antagonism −0.23, = the intersection she keeps from gridlocking.)
- **Two-step learning / breadth before depth** → the traffic guard can't route because she's never seen the street grid. The decision (reuse / sliver / new-route) is *comparative* — it requires a learned MAP of what's near task-A (the street grid), which requires seeing many tasks at the surface FIRST. Breadth builds the grid the depth-step's routing reads. [#2 CONFIRMED this in-neighborhood: a net given the structured childhood routes new depth around task-A (sliver) where a no-childhood net plows through (yank); B−A +0.126, B−C +0.115, it's the map not the mileage — see status block + CONSOLIDATION.]
- **The 10-year-old / adolescence** → human development does breadth FIRST (childhood: surface of everything, relate-all, master-none — the kid "thinks they know the world" and that confidence is CORRECT at its resolution; the map is complete at low-res). THEN adolescence drives depth into specific regions, and the prior breadth map is what lets the depth *sliver* (find its slot in the existing relational structure) instead of *overwrite the whole world-model*. Catastrophic forgetting = a network going through adolescence with no childhood. Depth with no prior breadth to sliver into.

**Why the metaphors are load-bearing (not decoration):** they generate *addressed* hypotheses, so their NULLS are informative (Sagan: absence of evidence is not evidence of absence — but a metaphor-ADDRESSED null is evidence of *structure*, it tells you the shape of the hole the answer must fit). Cascade-null located the mechanism. Deflation-null removed the novel-object reading. The metaphor's job is not to be right — it's to be specific enough that being wrong tells you where to dig next.

---

## 2. The developmental claim, stated precisely

Human non-catastrophic learning runs in TWO ordered phases:
- **Phase 1 (childhood / breadth):** low-resolution, relate-everything, master-nothing. Builds a *relational map* of the task-space — where things sit relative to each other. Overconfident but complete-at-its-resolution.
- **Phase 2 (adolescence / depth):** drive mastery into specific regions. The Phase-1 map lets new depth SLIVER (attach to its slot in the relational structure) rather than YANK (overwrite). Calculus doesn't erase your world-model because you spent ten years building the map calculus slots into.

The network skips Phase 1 entirely. It is born straight into a depth obsession with no world to hang it on. Handed a task, it drives depth immediately — no breadth map → the depth has nowhere to attach and nothing to relate to → it overwrites. **Catastrophic forgetting = the missing childhood.**

This is reported from inside the cognition that does breadth-first natively (the user is "patient zero for breadth over depth" — painting: you relate a million things to the red before you ever resolve the stand-oil/mineral-spirits ratio in your cad red; sommelier: place the wine by relating it to a hundred others before its tannin number means anything; Collatz: relate 7/45 across measure theory / Fourier / p-adics / Markov / inverse-tree until the neighborhood of decompositions PLACES it; CF program: mapped SST-2's task-neighborhood by hand before any depth question made sense). One domain-blind cognitive move — relate widely before resolving deeply — which is exactly the Phase-1 the network lacks.

---

## 3. What the characterization program already established (the foundation this builds on)

(Pythia-410M anchor, SST-2 task-A, multiple task-Bs; GPT-2-medium net-axis; all gated, all on disk in this dir.)

- Forgetting is DISPLACEMENT not destruction (fully weight-reversible; restore weights → task-A returns).
- It SEEDS at the shared, steep, ANTAGONISTIC directions (the contested core; antagonism cosine ~ -0.23, head-on; tasks want the same dials set to opposite values).
- It SPREADS as a laminar traveling front, curvature-ordered (high→low), additive, recoverable (gated both ways: spatial coherence z 24-105 + arrival-order spearman ~0.9).
- The fragile subspace is TASK-A's, NOT the architecture's (anchor-swap: SST-2→AG→UCDP, fragile subspace moves ~86%; fragile G ≈ topcurv G, so the shared sliver is just generic top-curvature, no architectural fragile core).
- Universality is ACROSS-task-B, SET-BY-task-A, attention-carried, graded with task-B distance (the beyond-curvature aimed component is UNIVERSAL across 4 task-Bs incl. code: seed +0.46-0.63, attention hard-universal +0.67; MLP erodes).
- The protection lever (GPM-style, full-time) works (~94% prevented) — but it is the KNOWN method; the contribution is the characterization, not the lever.
- The timed+fragile-specific refinement does NOT survive control (7-arm matrix): timed-early < full-time on real-forgetting tasks; "fragile-specific" can't be isolated from "stiff" because fragile ≡ stiffest-16 (control non-constructible, draw_lam 7x less stiff than fragile). One live redesign: the curvature-RESIDUAL version (test the ~30% beyond-curvature seed component, where a stiff-matched control IS constructible) — queued ahead of the breadth test.

**Implication for the spool:** the manipulation lever operates on LEARNED structure (task-A-set), which is MOVABLE by how you train task-A — not on fixed architecture you'd have to surgically edit. That makes the allocation hunt MORE tractable, not less.

---

## 4. LIT-MAP (3 web searches, 2026-06-20) — what's charted vs the gap

**Heavily charted (build on, do NOT claim):**
- Task-similarity drives forgetting; intermediate similarity = worst (teacher-student, Lee/Goldt/Saxe 2107.04384; EWC PNAS 2017).
- Curriculum/ordering OVER tasks; constructing paths through task-space from task-relationship info (Effect of Task Ordering, 2205.13323).
- Coarse-to-fine sequencing (CSCL: token→sentence→monolingual, Yoo 2024) — nearest to "breadth before depth" but it's coarse-to-fine WITHIN representation levels, not survey-all-tasks-shallow then specialize.
- Multi-task / interleaving (joint optimization avoids forgetting — but requires all tasks present at once; "everything simultaneously forever," not breadth-then-depth).
- Capacity / width as the "second spool" (Wide Nets Forget Less, 2110.11526; adapters/LoRA/PackNet/TAPS; SETA).
- Curriculum-CL frameworks synthesize: design by complexity/dependency/similarity, easy-to-hard (emergentmind, Feb 2026).

**The boundary-marking NULL (the key soil sample):**
- **DND "Hebbian Null Result"** (Context Channel Capacity, arXiv 2603.07415): 108 experiments / 12 days trying to make a network NATURALLY PARTITION representation space across tasks via Hebbian + top-k sparse + resonance gating + template matching. Best config: 73.19% ACC, -0.42% forgetting, 8 seeds — ~tied with frozen-random control. **Self-allocation did NOT emerge.**
- READ: this is an ADULT being slammed with a task and told "partition yourself, NOW" — with no childhood. They tried to make allocation EMERGE *during* the depth slam. Your frame predicts exactly this null: you cannot allocate a slot for a new task with no map of the neighborhood to place it in; partitioning IS a placing operation; placing requires the breadth map; the breadth map requires the childhood pass they never ran.

**The pattern across the null + the wins (Sagan-style reading):**
- Self-allocation DURING depth → NULLS (DND).
- Externally-supplied allocation (researcher hand-places adapters in advance) → WORKS (adapters/PackNet). But that's the RESEARCHER's childhood, supplied by hand — not the network's.
- The missing middle nobody tried: let the network BUILD ITS OWN MAP FIRST (a childhood phase), THEN let depth use it. Not emerge-during (nulls). Not hand-supplied (works but external). **Self-built-but-PRIOR.**

**THE GAP (white space):**
1. The breadth-pass as a DISTINCT PRIOR PHASE — broad shallow exposure to many task-Bs, explicitly mastering NONE, only to build the relational allocation-map of task-A's neighborhood — then depth, so depth slivers. NOT a named method.
2. The DEVELOPMENTAL framing (childhood-breadth → adolescent-depth as the missing computational step; CF = the missing childhood). NOT found. The van de Ven CL survey (2403.05175) explicitly CALLS FOR neuroscience/developmental inspiration as an OPEN direction → the angle is unmined, not done.

**Caveat (honest, same as every lit-check):** 3 searches. CSCL (coarse-to-fine) and the curriculum-CL frameworks are close enough that precise novelty needs their full-text read. Two papers to read before any novelty claim: **CSCL (Yoo 2024)** — is coarse-to-fine already the breadth pass? — and **DND (2603.07415)** — exactly what self-allocation approach failed, and does breadth-first dodge its failure mode? (DND looks like emerge-during-depth, which breadth-first explicitly does not do — but confirm full-text.)

---

## 5. QUEUED EXPERIMENT — breadth-pass v1 (behind the residual test)

**Question:** can a network learn the MAP of task-A's neighborhood before learning task-B deep — and does a network that "had a childhood" sliver where one that didn't would yank?

**The test (cheap first version — does the map even exist as a learnable object):**
1. Task-A = SST-2 (established w_A + all measured artifacts on disk).
2. **Breadth pass:** expose the net to a STREAM of many task-Bs (AG, Yelp, Subj, code, + as many more as loadable) — NOT to master them (no deep fine-tune), but to extract, per task-B, its seed/contested-core geometry at w_A (the step-0 objects, cheap, NO training). Produces a SET of neighborhood points: where each task-B would pull.
3. **Learn the map:** fit a representation of task-A's neighborhood from that set — per-direction label {contested-universal / task-A-private / free-lane} derived from the STREAM, not from any single pair.
4. **Test the map predicts:** hold out a NEW task-B (not in the breadth stream). Does the map predict where the held-out task-B will pull (lands on the flagged contested-universal directions, above chance)?

**Pre-committed verdict:**
- Map predicts held-out (above chance) → the breadth map is a REAL learnable object → step-1 is constructible → the seamstress has terrain → allocation becomes decidable from a LEARNED map, not hand-measurement. Path to #3 (network as its own seamstress). The developmental claim gets teeth: a net that "had a childhood" can place a task it never saw.
- Map does NOT predict held-out → the neighborhood isn't a stable learnable structure; "breadth" doesn't generalize to unseen tasks → two-step framing is conceptually right but the map is per-task-B, not learnable-in-advance. Bounds the idea.
- Partial → some of the neighborhood is stable-learnable (likely the universal contested core) + some per-task margin → report the split; the stable part is what a breadth pass could learn.

**Discipline (carry):** the HELD-OUT task-B is the whole control — the map must predict a task it did NOT see in the breadth stream, or "learned the map" collapses to "memorized the stream." Per-layer (never pooled — the ecological-fallacy trap that manufactured a false null earlier), seed-not-contest (convergence-clean), guards reproduce known objects (align=0 / anchor-reference style), chance baseline on the held-out prediction, dataset-identity guard (silent Dolly-fallthrough lesson).

**Future direction (downstream of v1):** if the map is learnable, the developmental experiment proper — give the network an actual CHILDHOOD PHASE (low-res broad pass over the task-stream, master none, build the map), THEN depth fine-tune on a held-out task-B, and measure: does the net that had a childhood SLIVER (retain task-A) where the no-childhood net YANKS? That is the direct test of "CF = missing childhood." The DND null is the boundary marker on the wrong side (emerge-during); breadth-first is the untried side.

---

## 6. Queue position

1. (ahead) Residual test — does the beyond-curvature seed residual protect beyond the curvature lever (the one live redesign from the 7-arm null; stiff-matched control IS constructible because residual ⊥ curvature).
2. (this) Breadth-pass v1 — is the task-A neighborhood a learnable map (held-out prediction test).
3. (downstream) Developmental experiment — childhood phase → does it sliver where no-childhood yanks.
4. (parallel, cheap) full-text read of CSCL (Yoo 2024) + DND (2603.07415) to tighten the novelty boundary.

---

## 7. The standing frame (why this isn't just another CL idea)

Forgetting is DISPLACEMENT, seeded at shared steep antagonistic directions, spreading as an orderly recoverable flow, shaped by basin geometry, OWNED BY THE FIRST TASK. The network can only YANK (relocate the one thread) because it lacks the SLIVER operation (grow a coexisting strand). Slivering needs a second spool (size + location known from the characterization) and a seamstress (doesn't exist — currently the researcher stands in). The seamstress can't self-decide because she has no map of the neighborhood. The map requires a breadth pass. The breadth pass is the CHILDHOOD the network never gets. Every self-allocation null in the literature is an adult slammed with a task and no childhood. The untried, lit-supported, white-space move: give the network a childhood first.

"A metaphor-addressed null is evidence of structure." The metaphors keep earning their tests, and even the nulls draw the map.
