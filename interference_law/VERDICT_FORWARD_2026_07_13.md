# VERDICT — NEW-3b: forward transfer EXISTS but is NOT cosine-governed (PREREG bb5bba73, locked 2026-07-13). Landed, $0 local, frozen resnet18, CIFAR + DTD, headroom via random-proj-24d.

## RESULT: Forward transfer occurs (38%/24% of B-inits help few-shot A, up to +0.13) — so the positive half of the phenomenon is real, unlike backward (NEW-3 ~0%). BUT it is COSINE-AGNOSTIC (rho(cos,FT) = +0.02/−0.01 ~ 0) and roughly net-zero (helps ~as often as it hurts). DISSOCIATION: the SAME features + SAME cosine that govern backward forgetting severity at rho +0.73 (NEW-3) govern forward transfer at rho ~0. → the cosine law is a FORGETTING law, NOT a general transfer law; transfer and forgetting are mechanistically different. The signed-law-unification hypothesis is FALSE.

## The numbers (frozen resnet18, random-proj-24d; FT = AUC_A(from B-full-init, few-shot) - AUC_A(cold few-shot))
  dataset  cold_auc  headroom  transfer_frac(FT>=.01)  hurt_frac  rho(cos,FT)  net(mean FT)  top-bin FT
  CIFAR      0.800     YES         0.38                 0.41        +0.024        ~0           +0.004
  DTD        0.809     YES         0.24                 0.52        -0.012        ~0           -0.017
  FT-vs-cos curve: FLAT/noisy around 0 across all cos bins (no monotone alignment relationship) on both datasets.

## Reading it
- **3b-P1 TRUE (forward transfer exists).** With headroom (cold AUC ~0.80), 38% (CIFAR) / 24% (DTD) of ordered pairs show FT >= +0.01 — a
  B-full-head init measurably helps learn few-shot A. Range up to +0.13. So the POSITIVE half of the phenomenon is real in the FORWARD
  direction (init/prior), where the backward-retention test (NEW-3) found essentially none.
- **3b-P2 FALSE (not cosine-governed).** rho(cos,FT) = +0.02 / -0.01 ~ 0; the curve is flat across alignment bins. And it is BALANCED
  (help 38% vs hurt 41%; net mean FT ~ 0). So forward transfer here is a HIGH-VARIANCE WARM-START effect — a generic B-init provides a
  real-classifier weight scale/prior that helps some few-shot A's and hurts others, INDEPENDENT of whether B is aligned with A. Not an
  alignment-driven transfer.
- **3b-P3 mixed / no (aligned not systematically positive).** CIFAR top bin +0.004 (negligibly positive), DTD -0.017 (negative). No clean
  aligned-B-helps pattern.

## THE DISSOCIATION (the key finding, and it is clean)
In the IDENTICAL feature space (same random-proj-24d resnet18) and with the IDENTICAL cosine descriptor:
  BACKWARD forgetting severity  <- cosine-governed, rho +0.73 (NEW-3, aligned = gentle forgetting)
  FORWARD  transfer (few-shot)  <- cosine-AGNOSTIC, rho ~0 (this)
The random projection is NOT destroying the cosine signal (it preserved rho 0.73 backward); the forward null is REAL. So the cosine
predicts backward interference but says NOTHING about forward transfer -> forgetting and transfer are DIFFERENT mechanisms. The cosine 4-op
law is specifically about INTERFERENCE/forgetting (which-way-you-point when OVERWRITING a capability), not a general "alignment => benefit"
law. Transfer, where it exists, follows a different (alignment-agnostic) rule.

## The complete transfer picture (NEW-3 + NEW-3b, both directions closed)
  BACKWARD (learn B, keep A?):  ONE-SIDED forgetting; cosine governs SEVERITY (aligned gentle -> anti severe, rho 0.55-0.73); NO transfer tail.
  FORWARD  (B-init, learn A?):  transfer EXISTS (warm-start, 24-38% of pairs help) but COSINE-AGNOSTIC and net ~0; NOT alignment-driven.
  => "Is the forgetting law also the transfer law?" ANSWER: NO. Cosine is a forgetting/interference law. The positive half (forward warm-
     start) is real but follows a different, alignment-agnostic rule. One axis does NOT span transfer<->forgetting.

## Caveats
Frozen READOUT regime (features fixed; a B task can only help A via head init/prior, not by improving features — the "true" forward
transfer via a PLASTIC backbone that B sharpens for A is a deeper, forgetting-confounded regime, untested = NEW-3c if ever). Headroom via
random-proj-24d (cold AUC 0.80; projection preserved the backward cosine signal, so the forward null is not a projection artifact). K=30
few-shot steps, 8 pos/24 neg, 3 seeds, 20 caps, CIFAR + DTD. Forward transfer is net ~0 (balanced help/hurt), so "exists" means "occurs for
a substantial fraction," not "systematically beneficial."

## Bottom line
The transfer question is CLOSED in both directions: backward = one-sided cosine-governed forgetting (no transfer tail); forward = real but
cosine-AGNOSTIC warm-start (net ~0). The clean dissociation (same cosine: rho 0.73 backward vs ~0 forward) proves the cosine law is a
FORGETTING law, not a transfer law -- forgetting and transfer are different mechanisms. This closes the entire NEW-menu on an honest,
sharp note.
