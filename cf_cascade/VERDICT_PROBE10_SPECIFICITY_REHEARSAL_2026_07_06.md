# VERDICT — Probe-10: specificity IS rehearsal-driven in OPT; the OPT>Pythia dissociation is GRADED, not binary. 2026-07-06.

Pre-reg `PREREG_PROBE10_SPECIFICITY_SIMILARITY_2026_06_29` (LOCK c83884a7). Harness `cloud/probe10_simrehearsal_lambda.py`.
task-A = SST-2; V = top-128 task-A Fisher per carrier; N=80 paired. DV = **S = N(T-realized-dir) − N(randtop16)** (8v4
estimator, magnitude-matched → not forget-confounded), per childhood. **Constructed** rehearsal-similarity contrast: 4
childhoods identical except one swapped slot (backbone {trec,subj} + {agnews|dbpedia|yahoo|emotion}); probe the realized
interference directions of T ∈ {agnews, yahoo}. Primary statistic = ident−unrel paired gap (tie-immune). OPT fired N=80
2026-06-30; Pythia null re-fired N=80 2026-07-05 at CHILD_LR=1e-5 (first attempt at 3e-5 was VOID-soft: childhoods
destructive; killed). Both cells non-destructive on the banked runs. $0.

## Headline
**Specificity is rehearsal-driven in the load-bearing arch (OPT): a childhood that rehearsed the interference task (or a
topic-sibling) aims protection at that task's direction; a childhood that rehearsed nothing like it gives only generic
protection.** The OPT>Pythia dissociation predicted by 8v4 is **GRADED, not binary** — clean on one probe (yahoo, 5.4σ),
absent on the other (agnews, ~1σ). Pythia is *substantially* weaker but not perfectly generic.

## Numbers (ident−unrelated gap = the aimed-protection premium from rehearsing the exact task)
| probe | OPT gap | Pythia gap | OPT−Pythia |
|---|---|---|---|
| agnews | +0.0183 ± 0.0048 (3.9σ) | +0.0115 ± 0.0045 (2.5σ) | 0.0068 → **~1.0σ (n.s.)** |
| yahoo  | +0.0338 ± 0.0044 (7.8σ) | +0.0040 ± 0.0034 (1.2σ) | 0.0298 → **~5.4σ** |

Full S ladders (S = aimed protection; sim 2=identical, 1=same-family topic, 0=unrelated affect):
- **OPT agnews** (ρ=1.00, clean monotone): ident 0.0425 > yah-sib 0.0326 > dbp-sib 0.0284 > unrel 0.0242.
- **OPT yahoo** (ρ=0.40): ident 0.0861 > agn-sib 0.0760 > unrel 0.0523 > **dbp-sib 0.0491** (dbpedia a WEAK sibling).
- **Pythia agnews** (ρ=0.80): yah-sib 0.0249 ≈ ident 0.0243 > dbp-sib 0.0219 > unrel 0.0128 (weakly graded, identical not top).
- **Pythia yahoo** (ρ=0.20): ident 0.0150 > unrel 0.0110 > agn-sib 0.0014 > **dbp-sib −0.0025** (near-null; same-fam < unrel).

## What this establishes
1. **OPT: rehearsal drives specificity — SOLID.** Both probes show a positive ident−unrel gap (3.9σ, 7.8σ); the agnews
   ladder is perfectly monotone in similarity. Rehearsing the exact interference task roughly doubles the aimed protection
   over an unrelated childhood (which itself provides a generic floor of S ≈ +0.02–0.05). This answers the 8v4 "WHY OPT aims":
   because the childhood rehearsed that task/family.
2. **Arch-dissociation is GRADED (OPT > Pythia), not the binary 8v4 read.** On yahoo the dissociation is clean (OPT 7.8σ vs
   Pythia null 1.2σ; 5.4σ apart). On agnews both arches show a weak gap (3.9σ vs 2.5σ) that are statistically
   indistinguishable (~1σ). So the pre-registered **P10-null (Pythia flat) is PARTIALLY VIOLATED** — Pythia is flat on yahoo
   but weakly rehearsal-graded on agnews. Defensible claim: OPT is *substantially more* rehearsal-specific than Pythia, NOT
   that specificity is uniquely a load-bearing property. This mirrors the existing arc tension — 8v4 read the split as binary,
   but 8v3 already showed the *targeting* is graded (OPT>Pythia, not all-or-nothing); Probe-10 shows the *rehearsal mechanism*
   behind it is graded too.
3. **"Same-family" is unreliable → the operative similarity is interference-DIRECTION, not task-format.** dbpedia is
   topic-classification by format but a WEAK sibling (OPT yahoo: below unrelated; Pythia yahoo: negative). The clean,
   consistent signal lives at the ENDPOINTS (identical vs unrelated); the format-defined middle is noisy. Pre-committed
   post-hoc (format vs direction) OWED — see below.

## Caveats / limits
- **Graded, not binary** (see #2): do not overclaim a clean OPT-only property. The agnews probe does not separate the arches.
- **n=2 architectures**, small models (350–410M), one task-A (SST-2), 2 probe directions. Scope, as with the whole arc.
- **Format-vs-direction post-hoc NOT YET RUN.** It needs the realized swapped-task vs probe-T gradient-direction overlaps,
  which this run did not persist → a small GPU re-derivation ($ small) or a cheap add to a future fire. Until then, "interference-
  direction, not format" is the leading explanation for the weak dbpedia sibling, not a proven one.
- Pythia first attempt VOID-soft (childhoods destructive at the mix5-calibrated 3e-5; the fixed-screen-lr assumption breaks
  for 3-task childhoods) — re-run at 1e-5 was clean. Lesson: recalibrate/verify non-destructiveness when the childhood
  composition changes, don't transfer a screen lr across childhood designs.

## Relation to the arc
Answers the ★-highest open thread from the 8v4 brief ("why is OPT specific?") on the mechanism side: **rehearsal-similarity.**
Does NOT touch the upstream chain (7/7b reorganization, 8v3 directional, 8v4 specificity) — it explains the *source* of 8v4's
aiming. Net: the capstone (8v4) now has a mechanism (rehearsal), with the honest amendment that OPT/Pythia is a graded
dissociation. Still the one claim to defend; still needs breadth (scale, 3rd arch) + the format-vs-direction post-hoc for a venue.

## Format-vs-direction post-hoc — DONE 2026-07-06 (LOCK 080dbe88): INCONCLUSIVE, hypothesis NOT supported
Realized each task's 250-step in-V interference direction (OPT, K=8), cosine-overlap matrix vs the agnews/yahoo probe dirs
(`gateA/probe10post_dirsim_opt350m.json`). Result **does NOT confirm "direction beats format":**
- **Directions are noise-dominated:** within-task self-consistency (cosine among the 8 realizations of the SAME task's dir)
  ≈ 0.18 — barely above cross-task overlaps (0.14–0.24). At 250 steps a task's in-V "direction" is not a tight object.
- **ρ(S, direction-overlap) = +0.36 < ρ(S, format) = +0.69** — format tracks S *better* (opposite of the pre-registered
  DIRECTION-WINS; though "format" here largely = "unrelated-affect is lowest," the endpoint effect already known).
- **dbpedia anomaly UNEXPLAINED:** dbpedia→yahoo overlap 0.216 ≈ agnews→yahoo 0.239 (within noise), yet their S as yahoo-
  siblings differ a lot (0.049 vs 0.076). Neither format (both topic) nor direction-overlap (near-equal) explains the
  within-topic difference at this resolution.
- Faint real structure: topic↔topic overlap ~0.23 > topic↔affect ~0.15, but near the 0.18 noise floor.
**Read:** per the locked map this is closer to FORMAT-WINS, but the direction measurement is noise-limited so it neither
proves format nor refutes direction — genuinely inconclusive. Does NOT touch the main Probe-10 result. The middle-rung
"weak-sibling" noise remains unexplained; a cleaner test would need a tighter direction estimate (more steps/averaging) —
NOT pursued (low value). Files `gateA/probe10post_dirsim_opt350m.json`, prereg LOCK 080dbe88.

## Status
Probe-10 CLOSED. OPT rehearsal-specificity SOLID; OPT>Pythia GRADED (clean on yahoo, n.s. on agnews); P10-null partially
violated. Format-vs-direction post-hoc INCONCLUSIVE (direction hypothesis unsupported, measurement noise-limited). Files
`gateA/probe10_simrehearsal_{opt350m,pythia410m}.json`, `gateA/probe10post_dirsim_opt350m.json`. No further post-hoc owed.
