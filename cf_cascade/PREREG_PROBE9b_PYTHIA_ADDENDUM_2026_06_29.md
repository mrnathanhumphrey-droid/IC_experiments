# PRE-REG ADDENDUM — Probe-9b Pythia arm + P1-sign falsification on the record. 2026-06-29.

To be SHA256-LOCKED (`PREREG_PROBE9b_PYTHIA_ADDENDUM_LOCK.json`) BEFORE the Pythia GPU fires. Amends
`PREREG_PROBE9b_PHASE2_LEVER_2026_06_28` (LOCK 869ce94e) with: (1) the OPT result, (2) a recorded falsification of P1's
directional sign + the mechanism that explains it, (3) pre-committed interpretations of the Pythia result, (4) Ω* discipline.
This is written and locked BEFORE reading the Pythia measure — so the verdict is dispositive either way and nothing is post-hoc.

## 0. OPT result that triggers this addendum (banked, n=6, $0 self-terminated)
| task | Ω | forget_noch | σ_sem | P |
|---|---|---|---|---|
| subj | 0.028 | 0.237 | 0.003 | +0.142 |
| trec | 0.038 | 0.095 | 0.012 | +0.033 |
| emotion | 0.120 | 0.118 | 0.022 | +0.042 |
| newscat | 0.164 | 0.041 | 0.008 | +0.018 |
| dbpedia | 0.174 | 0.057 | 0.010 | +0.026 |
| agnews+yahoo | 0.199 | 0.139 | 0.018 | +0.021 |

ρ(P,Ω) = −0.77 ; ρ(P,forget) = +0.66 ; ρ(P,σ_sem) = −0.03. Partials: ρ(P,Ω|forget) = −0.79 ; ρ(P,forget|Ω) = +0.69 ;
ρ(P,Ω|σ_sem) = −0.82 ; ρ(P,σ_sem|Ω) = +0.44. (Files `gateA/probe9b_measure_opt350m_*.json`.)

## 1. P1 directional prediction: FALSIFIED (recorded, not silently swapped)
Locked P1 (869ce94e) predicted **ρ(P,Ω) ≥ +0.70** with a crossover Ω* above which P>0. **The OPT data give ρ = −0.77.**
The MAGNITUDE matched; the **SIGN is opposite.** P1 is recorded as a **falsified directional prediction.**

**Why the flip is the mechanism, not a wound.** The positive guess encoded the *suppression/pinning* intuition (GPM/EWC:
protect the directions the old task occupies → armor thickest where the hit lands → more overlap = more protection). But the
whole arc (Probe-7/7b reorganization, Probe-8 v4 aimed-specificity, Probe-9 screen orthogonality) says childhood does NOT
suppress — it **reorganizes**, buying task-A a **free lane**: a direction the interference does not occupy. A free lane is by
definition *orthogonal* to where the childhood moved A's mass. So protection is **highest when interference lands orthogonal
to the childhood displacement (low Ω)** and dies as interference aligns with it (high Ω). **ρ(P,Ω) = −0.77 is the free-lane
mechanism showing its true sign.** Positive-Ω-law = pinning model; negative-Ω-law = reorganization model; the data picked
reorganization. P1's positive sign was the pinning model leaking into the prereg.

## 2. Three-substrate reconciliation (one sign, not a contradiction)
The negative law unifies the substrates rather than breaking them:
- **OPT (Probe-9b):** reorganization → free lane → P highest at low Ω (orthogonal hit). ρ(P,Ω) = −0.77.
- **Deepfake (Anchor B):** frozen backbone, NO reorganization, directions anti-parallel (cos_ms −0.99) → **no orthogonal lane
  exists** → suppression-only, protection collapses to randctrl. Same statement: *when interference cannot be made orthogonal
  to the deposit, P dies.*
- **Librarian (Anchor A):** free-lane available (moderate Ω 0.046), protection holds.
"Negative-Ω-law" and "antagonism kills the free lane" are **one law**: protection ∝ availability of an orthogonal lane.

## 3. Pythia arm — pre-committed interpretations (lock BEFORE reading)
Cell pythia410m, CHILD_LR = 3e-5 (screen-calibrated, ret 0.722). **Coverage-optimized 7-task ladder** (chosen now, pre-result,
for Ω spread + to populate OPT's empty middle when pooled + to clear n=7; NOT the OPT-identical 6):
subj (Ω 0.079), trec (0.090), subj+trec (0.104), emotion (0.231), agnews (0.277), yahoo (0.338), dbpedia (0.376).
(subj/trec/subj+trec at 0.08–0.10 land in OPT's empty 0.04–0.12 gap when pooled; agnews/yahoo fill 0.28–0.34; the redundant
top cluster newscat 0.407 / agnews+yahoo 0.402 is dropped.)

Pre-committed verdict map on **Pythia ρ(P,Ω)** (n=7, one-tail crit |ρ|≈0.71):
- **H_ARCH — ρ(P,Ω) ≈ 0 (|ρ| < 0.40):** architecture is the governing axis. The free lane is a property of the
  LOAD-BEARING arch (OPT); Pythia is inert/generic (Probe-8 v4 GENERIC) → its padding does not care where the hit lands →
  no Ω dependence. This is a **dissociation** (consistent with v4's SPECIFIC/GENERIC). A win.
- **H_UNIV — ρ(P,Ω) ≤ −0.60 (negative, same sign as OPT):** Ω is the **universal** axis; the architecture dissociation
  collapses; the free-lane law is substrate-general. The bigger, cleaner result. A win.
- **AMBIGUOUS — −0.60 < ρ < −0.40:** underpowered/mixed; report as such, do not force a side.
Both H_ARCH and H_UNIV are pre-registered wins; the result adjudicates *which*, it does not get narrated after the fact.

## 4. Ω* (crossover) discipline
Ω* (the overlap above which P→0) may be claimed **only if** the pooled OPT+Pythia curve has **≥2 populated points in the
middle band Ω∈[0.10, 0.25]** AND a bootstrap CI on Ω* that **excludes both endpoints** of the observed Ω range. A tidy
two-cluster (low/high) plot with an empty middle is **NOT** a crossover and must not be narrated as one.

## 5. OPT jackknife (reported alongside, $0 local, run now)
Leave-one-out Spearman ρ(P,Ω) on the OPT 6, especially drop-subj, to quantify how much of −0.77 is the single leveraged
corner point. Reported with the Pythia verdict so the law/▼subj question is answered before interpretation.

## Status
Addendum locked BEFORE Pythia fire. Pythia 7-task measure fan-out next (CHILD_LR=3e-5). Then pooled 3-substrate Ω law:
Spearman per cell + pooled, partials (Ω vs σ_sem vs forget), jackknife, Ω* only under §4.
