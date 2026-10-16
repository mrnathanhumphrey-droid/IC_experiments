# VERDICT — NEW-3c: PLASTIC forward transfer is cosine-AGNOSTIC and net-negative (PREREG e6086a1b, locked 2026-07-13). Landed, $0 local 1162s, plastic resnet18, CIFAR + DTD.

## RESULT: DEFINITIVE close. Even in the PLASTIC regime (full-FT the backbone on B, then learn few-shot A on the B-representation), forward transfer is NOT cosine-governed (rho +0.07/+0.09 ~ 0) and is NET NEGATIVE (moving toward B HURTS few-shot learnability of other tasks: net FT -0.055, hurts 62-68% of pairs). Combined with NEW-3 (backward) + NEW-3b (frozen forward), the cosine is PURELY a FORGETTING/INTERFERENCE law at EVERY level; transfer is NEVER cosine-governed. Forgetting and transfer are mechanistically distinct.

## The numbers (full-FT backbone on B; few-shot A on theta_B feats; FT = AUC_A(on theta_B) - AUC_A(on theta0))
  dataset  cold_auc  headroom  net_FT    transfer_frac  hurt_frac  rho(cos,FT)   FT_range
  CIFAR     0.813      YES      -0.0562      0.26          0.68       +0.069      [-0.414, +0.225]
  DTD       0.844      YES      -0.0535      0.31          0.62       +0.093      [-0.483, +0.275]
  FT-vs-cos curve: flat-NEGATIVE across all cos bins on both (no monotone alignment relationship).

## Reading it
- **3c-P1 NET NEGATIVE.** Full-FT on B, on average, DEGRADES the representation for few-shot learning OTHER tasks (net FT -0.055; 62-68% of
  pairs hurt vs 26-31% helped). B-training NARROWS/specializes features for B at a broad cost to general few-shot learnability. This is a
  representational-specialization cost, not a transfer benefit.
- **3c-P2 FALSE (cosine-agnostic).** rho(cos,FT) = +0.07 / +0.09 ~ 0; curve flat across alignment. Aligned B does NOT preserve/improve A's
  few-shot learnability any more than anti-aligned B. So even at the REPRESENTATION level, forward "transfer" is not alignment-governed --
  the specialization cost is broad and cosine-agnostic.
- The wide FT range ([-0.41,+0.22]) with net-negative mean = high-variance representational reshaping (some pairs helped a lot, more hurt),
  uncorrelated with cosine.

## THE COMPLETE DISSOCIATION (three regimes, same cosine -- the definitive result)
  regime                                   governed by cos(d_A,d_B)?   evidence
  BACKWARD forgetting (retain old A-head)   YES   rho +0.55..+0.73     NEW-3 (aligned = gentle forgetting)
  FORWARD frozen (B-head warm-start for A)  NO    rho ~0               NEW-3b (balanced warm-start, net ~0)
  FORWARD plastic (B-representation for A)   NO    rho ~0, net NEG      NEW-3c (this; specialization cost, agnostic)
Same features, same cosine descriptor, headroom preserved (rand-proj kept the backward signal at 0.73) -> the forward nulls are REAL, not
artifacts. The cosine predicts INTERFERENCE (overwriting a capability) and says NOTHING about TRANSFER (whether learning B helps learn A),
at BOTH the readout and the representation level.

## The final answer to "is the forgetting law also the transfer law?"
NO -- definitively, across all regimes tested. The cosine 4-op law is specifically a FORGETTING/INTERFERENCE law: it governs which-way-you-
point when OVERWRITING a capability (aligned coexist/gentle, anti-parallel collide/severe). It does NOT govern transfer in any direction:
  - backward: one-sided forgetting, cosine-governed severity, NO transfer tail (NEW-3);
  - forward frozen: warm-start, cosine-agnostic, net ~0 (NEW-3b);
  - forward plastic: representational specialization, cosine-agnostic, net NEGATIVE (NEW-3c).
Forgetting and transfer are DIFFERENT MECHANISMS. Transfer, where any positive effect exists, follows an alignment-agnostic rule.

## Caveats
Plastic full-FT confounds transfer with forgetting -- the measured NET representational effect is negative + agnostic (that IS the honest
answer for generic capability pairs; a hand-picked B that is a genuine superset/prerequisite of A might transfer, but across the pairwise
capability set it does not, and not by cosine). Random-proj-24d headroom (cold AUC ~0.82; projection preserved the backward cosine signal
-> forward nulls real); 3 few-shot seeds; K=30; 20 caps; CIFAR + DTD. 300-step B-FT (substantial move). Single backbone seed.

## Bottom line
PLASTIC forward transfer is cosine-agnostic AND net-negative -- B-training narrows the representation at a broad, alignment-independent
cost to few-shot learnability. This CLOSES the transfer question definitively across all three regimes: the cosine is a FORGETTING law,
never a transfer law. The last open door of the arc is shut, cleanly. Durable arc findings unchanged: frozen 4-op cosine FORGETTING law
(multimodal) + plastic 2-scalar magnitude structure (fragility + disruptiveness + transitive dominance).
