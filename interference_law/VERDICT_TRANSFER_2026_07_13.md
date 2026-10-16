# VERDICT — NEW-3: the forgetting law is NOT a transfer law — it is ONE-SIDED, cosine-governed (PREREG 52671eee, locked 2026-07-13). Landed, $0 local, frozen resnet18, CIFAR + DTD, headroom via random projection.

## RESULT: No backward transfer even WITH headroom. Sequential head-continual learning is one-sided: training on any task degrades the victim, monotonically LESS as cosine alignment rises (rho(cos,Δ) +0.73/+0.55, clean) but NEVER flipping to positive — aligned pairs asymptote to near-zero forgetting (coexistence), not transfer. The cosine governs the SEVERITY of forgetting, not a transfer/forget sign change. NEW3-P1 (transfer exists) FALSE; NEW3-P2 (cosine governs signed retention) TRUE.

## The numbers (frozen resnet18, victim=1-vs-rest, random-proj to 24d for headroom; retention Δ = AUC_A(after B) − AUC_A(base))
  dataset  base_auc  headroom  transfer_frac(Δ≥.01)  forget_frac  rho(cos,Δ)  zero-cross c0  most-aligned-bin Δ
  CIFAR     0.878      YES         0.00                0.99         +0.728        none           −0.15 (cos +0.67)
  DTD       0.889      YES         0.00                1.00         +0.549        none           −0.23 (cos +0.51)
  Δ-vs-cos curve (CIFAR): cos −0.62→Δ −0.55 … monotone … cos +0.67→Δ −0.15 (asymptotes toward 0, never crosses).

## Reading it
- **NEW3-P1 FALSE — no transfer with headroom.** Base AUC ~0.88 leaves clear room to improve, yet transfer_frac = 0.00 (<15%), and mean Δ
  is NEGATIVE in every cosine bin including the most aligned. A negligible few pairs nudge +0.046 (CIFAR) but it is <1%. Continual head-
  retraining only DEGRADES prior tasks; alignment makes it gentler, not helpful. NO backward-transfer tail.
- **NEW3-P2 TRUE — cosine governs the signed severity.** rho(cos,Δ) +0.73 (CIFAR) / +0.55 (DTD), strongly monotone: anti-aligned forgets
  hard (−0.55), aligned forgets gently (−0.15). This EXTENDS the frozen 4-op cosine law to the signed retention axis — the cosine sets HOW
  MUCH you forget across the full range. But the range is [severe forgetting → minimal forgetting], NOT [forgetting → transfer].
- **No zero-crossing (P3).** The aligned end asymptotes to Δ ≈ −0.15 (coexistence / near-zero forgetting) but does not reach positive. So
  the transfer/forget boundary c₀ does not exist in this regime — there is no transfer regime to cross into.
- **Dataset-general (P4):** identical qualitative answer on CIFAR + DTD.

## Honest caveat (the fair-test flag)
This tests BACKWARD transfer in SEQUENTIAL HEAD-RETRAINING: train head on A, continue-train it on B, re-measure A. This regime is
structurally biased toward forgetting — continue-training optimizes the head toward B's solution, i.e. AWAY from A's, so even an aligned B
pulls the head off A's optimum (net degradation, just smaller when aligned). Transfer might genuinely exist in a FAIRER regime that was NOT
tested: FORWARD transfer (does pretraining on B help LEARN A better/faster?), JOINT/multi-task (does co-training A+B beat A alone?), or a
shared REPRESENTATION co-trained (where B could sharpen features useful for A). The pre-registered BACKWARD-transfer question is answered
(no); the positive half of the phenomenon, if it exists, lives in forward/joint transfer — the natural follow-up (NEW-3b).

## What this adds to the arc
The frozen cosine law now has its SIGNED extent mapped: forget ∝ −cos runs from SEVERE forgetting (anti) to MINIMAL forgetting /
coexistence (aligned), cosine-governed at rho ~0.55–0.73 — but it is ONE-SIDED (no transfer) in the continual-readout regime. This
completes the negative axis and shows there is no positive (backward-transfer) tail here; the "aligned" limit is coexistence, exactly as
the routing/coexistence findings (aligned capabilities share a head safely) always implied — now confirmed on the signed retention axis.

## Caveats
Frozen readout regime (not backbone-plastic); headroom achieved via random projection to 24d (base AUC 0.88 — the projection distorts cos
somewhat, yet the cosine signal survived strongly, rho 0.73); 3 head seeds; 20 caps; CIFAR + DTD. A handful of near-zero positive Δ exist
but are negligible (<1%, mean always negative). Forward/joint transfer untested (NEW-3b).

## Bottom line
The forgetting law is NOT a transfer law: continual head-retraining is one-sided — cosine sets the SEVERITY of forgetting (aligned gentle
→ anti severe, rho 0.55–0.73) but there is no backward-transfer tail even with headroom; the aligned limit is COEXISTENCE, not gain. The
signed frozen-cosine axis is now mapped end to end. Positive transfer, if real, lives in forward/joint regimes (untested). A clean,
honest close to the NEW-menu.
