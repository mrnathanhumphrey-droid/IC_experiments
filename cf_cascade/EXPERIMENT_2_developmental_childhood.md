# EXPERIMENT #2 — The Developmental Experiment: does a childhood make depth SLIVER instead of YANK?

**For:** ML code agent. **Status:** queued culmination run. Heaviest of the program (phase-2 is real training).
**Substrate:** Pythia-410M, task-A = SST-2 (established w_A + artifacts on disk in D:\IC_experiments\cf_cascade\gateA).
**Prereqs banked:** breadth-pass v3 + #1 — the task-A neighborhood is a LEARNABLE, STABLE, CENTERED-DECAYING map (self-placement-by-distance confirmed on 5 unseen tasks; topic/subjectivity/emotion IN-neighborhood, STEM/code/conflict out-or-disconnected).

---

## THE CLAIM UNDER TEST
Catastrophic forgetting = the missing childhood. A network given a **breadth phase first** (low-res survey of the task-neighborhood, master none, build the map) should, when subsequently depth-trained on a new task-B, retain task-A BETTER than a network that went straight to depth — because the childhood map lets the new depth SLIVER IN (find its slot, coexist) instead of YANK (overwrite). This is the terminal test of the whole metaphor chain.

## THE LIVE METAPHOR (frame for reading results)
A traffic guard at a busy Manhattan intersection post-Knicks-win.
- **Fixed roads, no new asphalt** = can't grow weights (the thread/no-new-spool result).
- **Real-time arbitration of flow** = the gradient is managed DURING training, not pre-decided.
- **The busy intersection** = the contested core (~10-16 shared antagonistic directions, cosine ~ -0.23 head-on).
- **Post-Knicks-win** = the seeding transient (max pull, cosine strongest at the task-A minimum — when forgetting seeds).
- **Knowing the intersection cold** = the childhood map that makes real-time arbitration possible (breadth result).
- **Rookie guard run over** = no childhood → YANK. **Veteran who learned the corner on quiet Tuesdays** = childhood → SLIVER.
- DND null (self-allocation emerging DURING the rush, no childhood) = guard materialized mid-intersection during post-Knicks chaos, run over. We test the untried side: learn the corner on quiet Tuesdays FIRST.

## SCOPE BOUND (state up front; do NOT over-read)
#1 confirmed the neighborhood is centered on task-A (sentiment) and decays with distance. So **this test is valid ONLY for an IN-neighborhood held-out depth task.** Pick the held-out depth task-B from the IN-neighborhood set — **emotion or subjectivity** (#1 confirmed both reachable; neither is the breadth-stream's center, neither is task-A itself). A positive result says "childhood makes depth sliver for tasks the childhood could reach" — it does NOT say childhood helps for alien kinds (code/conflict); that needs a real spanning-bridge experiment, still open. An in-neighborhood NULL damages the frame (interpretable); an alien-task null would be uninterpretable (don't use one).

---

## THE THREE ARMS (the scrambled-childhood control is the whole experiment)

| arm | phase 1 (childhood) | phase 2 (adolescence) | isolates |
|---|---|---|---|
| **A. no childhood** | none | depth-train held-out task-B from task-A's w_A | baseline YANK (the floor) |
| **B. real childhood** | breadth pass over the neighborhood task-stream (NOT the held-out task), low-res, master none, build the map | depth-train the SAME held-out task-B | does childhood → SLIVER (retain task-A better than A)? |
| **C. scrambled childhood (CONTROL)** | breadth pass over the SAME stream, STRUCTURE-DESTROYED (shuffled labels / permuted task-identity — map is noise, exposure+compute matched to B) | depth-train the SAME held-out task-B | is the benefit the MAP, or just more compute/exposure? |

## DECISIVE CONTRASTS
- **B vs A** — does real childhood retain task-A better than no childhood? (The headline: B retains more task-A after depth → childhood slivers.)
- **B vs C** — is the benefit from the MAP (structured childhood) or just more training (scrambled = matched compute, destroyed structure)? **THIS IS THE CONTROL THAT MAKES IT REAL.** B > C → the map did the work (the developmental claim). B ≈ C → "childhood" was just extra compute, the map did nothing, central claim fails.

## TWO METRICS PER ARM (the sliver is two-sided)
- **task-A retention** after depth-training task-B (vs pre-task-B baseline) — did task-A survive (the sliver)?
- **task-B accuracy** — did childhood COST task-B learning? Sliver means COEXIST: retain A AND learn B. Retain-A-by-starving-B is rigidity, not slivering.

## PRE-COMMITTED VERDICT
- **B > A on task-A retention AND B > C AND task-B not starved** → childhood makes depth sliver, and it's the MAP not the compute → developmental claim LANDS (in-neighborhood). Terminal prediction of the metaphor chain confirmed.
- **B ≈ A** → childhood doesn't help even in-neighborhood → frame is wrong at the experiment it was built for. Clean negative, the big one.
- **B > A BUT B ≈ C** → benefit is compute/exposure, not the map → map-specificity claim dies even if more-training-helps (the DND-adjacent failure mode: exposure without the map doing work).
- **B > A, B > C, BUT task-B starved** → childhood retains A by refusing to learn B → rigidity, not slivering. Does not support the claim.

## ROBUSTNESS (carry the lessons)
- **~6-8 seeds.** Expected effect (childhood retention benefit) likely SMALL; 7-arm noise was ±0.05-0.07; underpowered seeds drown a real B-vs-A or B-vs-C gap. Judge B-vs-A and B-vs-C on ERROR-BAR SEPARATION, not point estimates.
- **Held-out task-B must forget meaningfully in arm A** (pre-check: arm A drops task-A >= 0.10). No forgetting → no dynamic range (the dead-code-substrate lesson). Confirm before committing the grid.
- **Matched compute B vs C.** Scrambled childhood uses the SAME steps/exposure as real childhood. The ONLY difference between B and C is structure-vs-noise in the breadth phase. (Else B>C is a compute confound.)
- **Childhood is BREADTH not DEPTH.** Phase 1 must master NONE of the stream tasks (low-res / shallow-touch / step-0 regime, NOT fine-tuning them to convergence). If childhood accidentally depth-trains the stream, it's just multi-task pretraining (known to help) and the developmental claim is UNTESTED. Confirm stream-task loss does NOT converge in phase 1.

## GUARDS BEFORE VERDICT
- Arm A forgets (floor reproduces).
- Held-out task-B actually learns in all arms (phase-2 converges).
- Childhood phase confirmed low-res (stream tasks NOT mastered).
- Per-seed distributions reported, not just means.
- align=0 / dataset-identity guards (silent-fallthrough lesson).
- Save all arrays before readout.

## COST / EXECUTION
3 arms x ~7 seeds = ~21 runs. Each has a TRAINING phase-2 (heavier than step-0 breadth tests); arms B/C add a phase-1 breadth pass. Heaviest run of the program — estimate before firing. Lambda. **Save childhood-phase checkpoints** so phase-2 arms branch from them without recomputing the childhood (B and C each need their phase-1 saved once, reused across the held-out depth seeds).

## WHY THIS IS THE CULMINATION
Every prior result mapped HOW the yank happens (seed, flow, contested core, fragile subspace, the neighborhood map). This is the first test of whether GIVING the network the missing developmental step changes the OUTCOME — whether a childhood produces slivering. B vs A asks "does childhood help"; B vs C asks "is it the map or the compute." Both required. In-neighborhood scope only. The veteran guard learned the corner on quiet Tuesdays; this asks whether that lets him work the Knicks crowd without getting run over.

## USER ADDITIONS
