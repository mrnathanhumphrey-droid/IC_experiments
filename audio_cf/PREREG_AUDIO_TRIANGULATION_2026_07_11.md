# PRE-REG — Thread #5: does the directional-antagonism / cosine→forgetting LAW hold in a THIRD modality (AUDIO)? 2026-07-11.

SHA256-lock before the run. Triangulation: the CF arc's directional law is confirmed in TEXT (Librarian reranker: cos+0.67
constrained → don't route) and VISION (deepfake CNN: cos−0.93 contested → route, diffusion 0.68→0.96). This tests whether the SAME
law — the discriminative-direction cosine predicts forgetting, and anti-parallel paradigms must be routed — holds in AUDIO deepfake
detection. If yes, the mechanism is modality-general, not a quirk of two systems.

## HARD ISOLATION (locked)
READ-ONLY on `D:/physics_detector`. Audio globbed read-only from `audio_corpus/`. Writes ONLY to `D:/IC_experiments/audio_cf/out/`.
NEVER touches the production audio detector, its LDA/KDE checkpoints (v2/audio_v1_models/*.pkl), operators, or any app. $0 LOCAL CPU.

## Design (faithful to the detector probes; matched-utterance = the cleanest shared-reals yet)
- **7 fake paradigms** = the TTS/vocoder generators: bark, elevenlabs, gtts, mms, parler, pyttsx3, speecht5 (200 each; gtts 400).
- **Shared REAL pool** = LibriSpeech real speech (label 0). The synth clips are NAMED BY LibriSpeech utterance IDs → the SAME
  sentence/speaker content is synthesized by every generator AND exists as real → real-vs-synth is content-matched; the ONLY thing
  differing across paradigms is the GENERATOR (the strongest shared-reals control in the arc so far).
- **Frozen feature extractor** (the "backbone"): generic, deterministic log-mel — librosa 64-band log-mel @16k (gtts 24k→16k),
  per-band mean+std over time = 128-d fixed vector. NO training in the extractor. Deliberately a GENERIC representation (not their
  physics-operator bank): the LAW is about the geometry of task directions and should hold in ANY reasonable fixed space; if it holds
  in log-mel that is a real third-modality datapoint, and re-running on their operator features is the faithful follow-up.
- Frozen features → linear head (real-vs-fake), exactly the detector paradigm. Metric = rank-AUC. Deterministic filename-hash split.

## Three diagnostics (mirror the established probes)
1. **7×7 DISCRIMINATIVE-DIRECTION COSINE MATRIX.** d_p = normalize(mean(feat synth_p) − mean(feat real)). cos(d_i, d_j) for all
   pairs → which generators are anti-parallel (contested) vs aligned (constrained). Richer than the arc's 2 points — a whole matrix.
2. **FORGETTING vs COSINE (the LAW test — seeds Thread #1).** For paradigm pairs spanning the cosine range: train head on A → AUC_A;
   continue-train on B → AUC_A'. forgetting = AUC_A − AUC_A'. Scatter forgetting vs cos(d_A,d_B) across pairs; report Spearman ρ.
3. **ROUTE structure.** Cluster paradigms by cosine (TAU=0), separate heads + max-ensemble vs naive single sequential head over an
   ordered surge → does routing retain the worst paradigm (the detector lesson transferring to audio)?

## Pre-registered read (locked before the run)
- **LAW HOLDS (audio joins vision+text)** IF: forgetting rises as cos(d_A,d_B) falls (Spearman ρ < −0.4, negative & significant),
  same DIRECTION as vision/text — anti-parallel pairs forget more, aligned pairs coexist. The 7×7 matrix shows ≥1 clearly
  anti-parallel (cos < −0.3) and ≥1 clearly aligned (cos > +0.5) pair (structure, not all-alike).
- **ROUTE TRANSFERS** IF: cluster-aware heads + max-ensemble beat naive-single on the worst (most-forgotten) paradigm by > 0.03 AUC,
  as in vision.
- **NULL / doesn't transfer** IF: ρ ≈ 0 (cosine doesn't predict forgetting in audio) OR all paradigms mutually aligned (no
  antagonism to route) OR forgetting ≈ 0 everywhere (log-mel too coarse to carry paradigm-specific signal). Report honestly; a null
  would mean the law is NOT representation-general (or log-mel is the wrong space → operator-bank follow-up).

## Caveats (honest)
Log-mel is a GENERIC frozen representation, NOT their production operator bank (deliberate $0 scope; operator-features = follow-up).
Head-level only (frozen features). Filename-hash split (not speaker-disjoint) → absolute AUCs inflated; the forgetting DELTAS +
the cosine STRUCTURE are the claim, robust to inflation. Content-matched utterances = controlled but narrow domain (read speech).
Single seed; SUBSAMPLE per paradigm capped for CPU. sub-0.02 AUC = noise; report the matrix + scatter, don't over-read one cell.

## Ops
`probe_audio_seq.py` reusing the detector-probe structure (extract/_hash_split/train_head/continue_train-GPM/pca_basis/auc/
discriminative_dir/run_route_arm) with a librosa log-mel `feat()`. Cache features to out/*.npy (extract once). py_compile + --smoke
before the full run. $0 local CPU, minutes. Output `out/audio_seq.json` (cosine matrix + forgetting-vs-cosine + route arm).

## Status
PREREG written; data verified (7 paradigms readable, 3091 real, matched utterance IDs, librosa/soundfile present). Next: SHA256-lock
→ build → --smoke → full run → read the 7×7 cosine matrix + forgetting-vs-cosine ρ + route transfer → bank.
