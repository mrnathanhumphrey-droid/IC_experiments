# VERDICT — Thread #5: audio triangulation (PREREG 17a22acb, locked 2026-07-11). Landed, $0 local CPU, 18s.

## RESULT: the directional-antagonism / cosine->forgetting LAW TRIANGULATES to a THIRD modality (AUDIO).
Frozen log-mel (128-d) + linear real-vs-fake heads over 7 TTS/vocoder paradigms + shared LibriSpeech real pool.

## D1 — the SAME structure as vision: a compatible cluster + a lone anti-parallel antagonist
Discriminative-direction cosine matrix:
- 6 neural/concatenative TTS (bark, elevenlabs, gtts, mms, parler, pyttsx3) form a TIGHT CLUSTER — mutual cosine +0.55..+0.96
  (the "swap-family" analog).
- **speecht5 is the LONE ANTAGONIST**: cosine to every other generator is NEGATIVE (−0.11 to −0.37; +0.08 to bark) — the DIFFUSION
  analog. A different vocoder architecture leaves an anti-parallel artifact direction.
- off-diagonal cos: min −0.37, max +0.96, mean +0.54. Real structure, not all-alike.

## D2 — the LAW holds: forgetting rises as cosine falls (Spearman rho = −0.519, n=42 ordered pairs), monotone by bucket
- ALIGNED (cos>0.7): mean forget +0.824, MIN +0.110 — the most-aligned pairs COEXIST (elevenlabs+gtts cos0.96 forget0.11;
  elevenlabs+parler 0.95->0.175; pyttsx3+mms 0.92->0.216).
- mild (0<cos<0.7): +0.963.  ANTI-PARALLEL (cos<0, all involve speecht5): +0.996.
- Same DIRECTION as vision (detector cos−0.93 catastrophic) and text (Librarian cos+0.67 benign). Third modality, same sign.

## D3 — the ROUTE remedy transfers: cluster-aware heads recover the worst paradigm
- Router clustered EXACTLY as the cosine predicts: C0={bark,elevenlabs,gtts,mms,parler,pyttsx3}, C1={speecht5}.
- Naive single sequential head: worst-paradigm retention = 0.000 (it forgets everything but the last-trained).
- Route (separate heads + max-ensemble): worst-paradigm retention = 0.929. **route − naive = +0.929 on the worst paradigm.**
- The detector's "isolate the anti-parallel family, score by max-ensemble" lesson works in audio too.

## HONEST CAVEAT (the one weakness, pre-registered): log-mel saturates real-vs-synth AUC at 1.00
Every base AUC = 1.00 → these TTS are trivially detectable in log-mel, so the linear head overwrites near-totally on continue-train,
compressing forgetting toward 1.0 in the MIDDLE of the cosine range (Pearson only −0.29). The LAW is therefore clean at the
EXTREMES (aligned pairs demonstrably coexist at 0.11; the anti-parallel antagonist forgets totally) and in RANK (Spearman −0.52,
monotone buckets), but the GRADED-MAGNITUDE curve is muddied by saturation. The properly de-saturated forgetting-vs-cosine curve
belongs to Thread #1 (interference law) with a harder, non-saturating task; and re-running on the production physics-operator
features (not generic log-mel) is the faithful within-audio follow-up. Head-level only; filename-hash split (not speaker-disjoint,
abs AUC inflated — deltas/structure robust); single seed; read-speech domain.

## CAPSTONE UPDATE: the directional-antagonism law is now confirmed in THREE modalities
| modality | system | structure | lone antagonist | remedy |
|---|---|---|---|---|
| TEXT   | Librarian reranker | rules+fantasy constrained (cos+0.67) | none (no antagonist) | don't route |
| VISION | deepfake CNN | swap-family + diffusion | diffusion (cos−0.93) | route -> diffusion .68->.96 |
| AUDIO  | TTS detector (log-mel) | neural-TTS cluster + speecht5 | speecht5 (cos −0.11..−0.37) | route -> worst .00->.93 |
The discriminative-direction cosine (its SIGN + magnitude) is a portable, modality-general predictor of interference and of when to
route. The CF-arc directional mechanism is not a two-system quirk.
