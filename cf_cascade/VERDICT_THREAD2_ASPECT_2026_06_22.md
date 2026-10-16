# VERDICT — Thread 2: disentangle WIDTH / DEPTH / ASPECT-RATIO from FAMILY. 2026-06-22.

Hypothesis going in (mine): the OPT-vs-Pythia dissociation might COLLAPSE into aspect-ratio (d_model/n_layers) — plan was
to build width-surgery to test it. **Result inverts the hypothesis, with NO new GPU** — the size ladder + depth-surgery
already span width×depth, and crucially two OPT/Pythia cells are BYTE-IDENTICAL in architecture. Analysis `aspect_grid.py`
/ `gateA/aspect_grid.json`.

## The grid (dims measured from checkpoints)
| cell | family | d_model | layers | aspect | G5 | B-D |
|---|---|---|---|---|---|---|
| opt125m | OPT | 768 | 12 | 64.0 | 0.344 | +0.110 |
| pythia160m | PYTHIA | 768 | 12 | 64.0 | 0.525 | −0.005 |
| opt350m | OPT | 1024 | 24 | 42.7 | 0.410 | +0.055 |
| pythia410m | PYTHIA | 1024 | 24 | 42.7 | 0.505 | −0.006 |
| opt1_3b | OPT | 2048 | 24 | 85.3 | 0.439 | pending |
| pythia1b | PYTHIA | 2048 | 16 | 128.0 | 0.490 | pending |

## 1. MATCHED-SHAPE FAMILY CONTROL (the killer — byte-identical architecture)
- **shape (768,12,3072): opt125m vs pythia160m** → G5 gap **−0.180**, B-D gap **+0.115**
- **shape (1024,24,4096): opt350m vs pythia410m** → G5 gap **−0.095**, B-D gap **+0.060**

Two independent OPT/Pythia pairs with IDENTICAL d_model, n_layers, d_ff. Both G5 AND B-D still separate strongly by family.
**The dissociation does NOT collapse into shape — it survives byte-identical architecture.**

## 2. Magnitude ranking — family dominates geometry
- Family offset at matched shape: **0.095–0.180** (G5) — DOMINANT.
- Within-family width (d_model) move: 0.035 (Pythia) – 0.095 (OPT).
- Same-aspect width gap: 0.046. Depth move (fixed width): 0.043.
The matched-shape family gap is the largest single effect and it is geometry-independent.

## 3. Secondary structure (real but second-order)
- **Within-family WIDTH modulation, opposite-signed across families:** OPT G5 RISES with d_model (0.344→0.410→0.439,
  spearman +1.0), Pythia G5 FALLS (0.525→0.505→0.490, spearman −1.0). A recipe×width interaction, ~½ the family offset.
- **Aspect-ratio per se does NOT drive G5:** opt350m-d12 (1024/12, aspect 85.3) G5=0.393 vs opt1_3b (2048/24, aspect 85.3)
  G5=0.439 — same aspect, 0.046 apart → tracks ABSOLUTE width, not the ratio.
- **Depth does NOT drive G5:** flat across d6–d24 (range 0.043), consistent with Thread-1 depth-side mediation.

## VERDICT
The cross-architecture CF dissociation (G5 deposit↔surge AND the B-D protection effect) is **NOT aspect-ratio, NOT width,
NOT depth** — it survives byte-identical SHAPE. **Width-surgery is unnecessary for the family question** (banked: check
existing infra before pouring concrete).

### HONEST CEILING (sharpened) — "matched shape" ≠ "matched architecture"
The two control pairs match d_model / n_layers / d_ff EXACTLY, but OPT and Pythia still differ in DESIGN: OPT = ReLU +
learned absolute positions + sequential attn→FFN; Pythia(NeoX) = GeLU + RoPE + parallel attn‖FFN. So the precise claim is
**irreducible to SHAPE (dims/aspect-ratio)** — NOT "irreducible to architecture", and NOT "it's the data". The gap lives
in a CONFOUNDED BUNDLE:
  {design knobs: position-encoding, activation, attn-topology} ∪ {recipe knobs: data-mix, tokenizer, objective, optimizer,
  training-length}.
These are NOT separable on the OPT/Pythia checkpoints alone — no matched-everything-but-one-knob variants exist across
families. That is the honest limit of what the grid can say.

### Prying the bundle apart (what IS / ISN'T runnable)
- ISOLABLE (within-Pythia controlled variants, each = wa_trainer + geom): training-LENGTH via Pythia step-checkpoints
  (same arch+data, vary optimization amount); a DATA knob via Pythia-deduped (same arch+length, vary dedup). These pry
  recipe knobs by ELIMINATION at fixed design+corpus.
- NOT ISOLABLE on existing checkpoints: design knobs (RoPE/GeLU/parallel-attn), corpus identity, tokenizer — would need
  controlled pretraining we do not have.

### Resolves the saddle-vs-basin open question
The dissociation-layer's open puzzle ("why does OPT-350m sit in a rung-1 saddle while Pythia-410m is in a basin — is it
width?") is answered: opt350m and pythia410m are IDENTICAL shape (1024/24/4096), so the saddle-vs-basin is ALSO not
geometry — it is pretraining. Width is not the axis.

## Knob-isolation (within-Pythia controlled variants — prying the recipe side by elimination)
Each cell = wa_trainer (matched-acc w_A) + geom_cell (G1-G6). Scripts: `cloud/geom_cell_lambda.py` +
`cloud/lambda_launch_geomcell.py`; wa-trainer now accepts a `revision` arg (Pythia step-checkpoints).

| knob | cell | G5 | vs pythia-160m std (0.525) | % of 0.181 family gap |
|---|---|---|---|---|
| DATA (dedup) | pythia-160m-deduped (acc 0.803) | 0.485 | −0.039 (3× boot-SE, OPT-dir) | ~22% |
| **LENGTH** | pythia-160m @ step16000 (acc 0.820) | **0.437** | **−0.088 (OPT-dir)** | **~49%** |
| (full gap) | opt-125m, same shape | 0.344 | −0.180 | 100% |

**Knob verdict — LENGTH is the dominant isolable recipe knob, DATA secondary:**
- **Training length closes ~49% of the family gap** alone (2× data's effect). G5 *rises* through training (0.437 @ step16k
  → 0.525 @ final): the protective low-G5 state (deposit out of the surge's path) is an EARLY-training property that
  ERODES with more optimization. Length also strongly moved G2 (1.57 vs 2.62), G3 (0.377 vs 0.552), G6 (0.318 vs 0.47).
- Data (dedup) is real but secondary (~22%), broad-but-modest (G4 −0.05, G6 −0.06).
- Data + length span ~70% of the gap magnitude IF additive (caveat: measured independently vs the final-Pythia baseline;
  OPT differs in both + corpus/tokenizer/design). Residual + non-additivity live in the unisolable bundle.

**Falsifiable prediction (being tested now, cell `pythia160m_step16k` B-D):** low G5 = protection = high B-D, so a
LESS-trained model should be MORE CF-protective → **extended pretraining erodes catastrophic-forgetting protection.**
Test: does step16000 B-D exceed final-pythia-160m B-D (≈ −0.005)? If yes, the length→G5 geometry shift has teeth in the
actual protection outcome; if B-D is flat while G5 moved, geometry and outcome decouple on the length axis.

## What (optionally) remains
- The within-family WIDTH modulation (OPT-rises / Pythia-falls G5 with d_model) is a genuine recipe×width interaction at
  n=3/family. A width-surgery sweep (slice pretrained d_model at fixed depth, matched-acc) could confirm it IS the only
  thing left that width-surgery would add — second-order, user's call whether worth the GPU.
- pending B-D for opt1_3b / pythia1b sharpen the within-family B-D width trend (feeds the agent's G5->B-D mediation).
