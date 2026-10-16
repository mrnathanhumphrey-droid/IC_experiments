# VERDICT — NEW-1: the interference matrix is ASYMMETRIC; FRAGILITY is a new dominant axis (PREREG ebdbba5a, locked 2026-07-12). Landed, $0 local 119s, 12×12 directed matrix, non-saturated (median off-diag 0.206).

## RESULT: The directed forgetting matrix is STRONGLY ASYMMETRIC — forget(i←j) is UNCORRELATED with forget(j←i) (ρ=−0.02). The symmetric cosine law is DECISIVELY INCOMPLETE. The dominant governing variable is a NEW per-capability scalar — victim FRAGILITY (ΔR²=+0.23, vs cosine's R²=0.07) — not the geometry. Dominance is ~46% transitive (partial pecking order + non-transitive residual).

## The four predictions
  New1-P1 ASYMMETRY        ‖K‖/‖S‖ = 0.285 (≥.15) ; ρ(F_ij,F_ji) = −0.016 (<.80)   ✅ CONFIRMED, decisively.
  New1-P2 FRAGILITY 2nd-axis dR²_frag = +0.233 (≥.10) ; ρ(fragile,disrupt) = −0.24  ✅ CONFIRMED, and it's the DOMINANT factor.
  New1-P3 disrupt↔‖Δθ‖     ρ(disrupt_j, ‖Δθ_j‖) = −0.077                             ❌ FAILED — but for a design reason (below), not a contradiction of #1.
  New1-P4 dominance        rank-1 potential explains 45.8% of ‖K‖²                     ◐ MIXED (neither clean hierarchy ≥60% nor clean cycle <40%).

## The headline: forgetting is ASYMMETRIC, and it's about the VICTIM
- **ρ(F[i←j], F[j←i]) = −0.02.** Knowing how much capability i forgets when you train j tells you NOTHING about how much j forgets when
  you train i. The cosine — symmetric by construction — CANNOT produce this. Our central law is provably incomplete for directed forgetting.
- **Variance decomposition (directed forgetting):** cosine alone R²=0.071 → +perturber(col) 0.238 → +victim-fragility(row) 0.470. The
  VICTIM's identity (fragility) adds ΔR²=0.233 — MORE than cosine (0.07) and more than perturber-strength. **The single biggest
  determinant of how much a capability forgets is WHO IT IS (its fragility), not the geometry to the perturber nor the perturber's power.**
- **Fragility ≠ bully.** ρ(fragile_i, disrupt_i) = −0.24: capabilities that are easily damaged do slightly LESS damage. Two genuinely
  distinct per-capability axes — how easily you're hurt vs how much you hurt others.
- **New factorized law (plastic regime):** forget(i←j) ≈ fragility(victim i) ⊕ disruptiveness(perturber j) ⊕ small geometry(cos_ij).
  The cosine, which the whole arc built, is the SMALLEST of the three terms once the backbone is plastic.

## Why P3 failed (honest — a design compression, not a contradiction of #1)
#1 crowned ‖Δθ‖ as disruptiveness-king across 5-class B-tasks whose ‖Δθ‖ ranged widely (7→16). Here every perturber is binary(class_j
vs neg) and they ALL travel ~equally (‖Δθ‖ 11.5–15, mostly ~12) → ‖Δθ‖ has almost no variance to explain the column differences. So
disruptiveness here (col mean 0.14–0.27) is set by the DIRECTION of the equal-magnitude perturbation, not its length. Reconciles with
#1: ‖Δθ‖ governs WHEN magnitude varies; at fixed magnitude, perturber direction re-enters. Not a contradiction — a different (equal-travel) slice.

## P4: the dominance structure is PARTIALLY transitive (a label-corrected reading)
Rank-1 potential h_i = (fragile_i − disrupt_i)/2 explains 45.8% of the antisymmetry ‖K‖². So ~half the directional forgetting is a clean
scalar pecking order; ~half is non-transitive residual (potential capability rock-paper-scissors). NOTE — the probe's print label
"predator=+/prey=−" is INVERTED: h>0 means fragile>disrupt = PREY (damaged more than it damages); h<0 = PREDATOR. Correct reading:
  PREDATORS (h<0, bully-net): class 0 (−0.072, biggest bully: col-damage 0.265 highest, but itself robust), 42, 63, 77, 28, 70.
  PREY (h>0, fragile-net):    class 21 (+0.044, most fragile-net), 35, 7, 14, 56.
The h-values/hier_frac are computed correctly; only the printed annotation is swapped.

## What this OPENS (the new ground — genuinely new questions)
1. **WHAT IS FRAGILITY?** It's now the dominant axis and completely uncharacterized. Candidate predictors of fragile_i: base-margin/AUC
   headroom, within-class feature spread along w_i, isolation of d_i (mean cos to other caps), ‖w_i‖/conditioning, how far i's features
   travel under a GENERIC perturbation. A probe regressing fragile_i on these = a mechanistic theory of what makes a capability easy to destroy.
2. **IS THE NON-TRANSITIVE RESIDUAL REAL?** 54% of ‖K‖² is off the pecking-order. Cyclic interference (i→j→k→i) would be a striking
   structural finding. Test: seed-stability of the residual cycles + a directed-triangle imbalance statistic vs a symmetric null.
3. **Does fragility TRANSFER across substrates?** Is a capability's fragility intrinsic (about the class) or backbone-relative? Cross-backbone
   fragility correlation would say whether "fragile capabilities" are a real, portable category.

## Caveats
Single seed; ResNet18; CIFAR-100; deployed binary heads; 12 caps (66 directed pairs). ‖Δθ‖ compressed by the binary-vs-neg design (drove
P3's null — a match-magnitude perturber set would re-test disruptiveness). Cosine R²=0.07 is LOW here but EXPECTED: this is the PLASTIC
regime where #7/#7b/#1 already dethroned direction — the news is not "cosine is weak" (known) but "victim FRAGILITY is the dominant,
previously-unmeasured axis, and forgetting is asymmetric." P4's 45.8% is single-seed; the cyclic-residual reality is #2 above. Label bug
in one print statement noted (data correct).
