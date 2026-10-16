# READOUT — BLOOM's inverted fingerprint, from banked Probe-13 rows ($0, no new fire). 2026-07-09.

Re-analysis of the 4 banked `gateA/probe13_doseresponse_*.json` summaries (script:
scratchpad `bloom_readout.py`). No GPU. Two jobs: (1) confirm BLOOM's negative slope isn't one bad
point; (2) extract a dumpable mechanistic hypothesis for the confound-breaker fire.

## Dose-response, N_phase2 (protection of realized interference dir) vs N_random (matched-mag control)
| cell | Np gentle→strong | Np slope | Nr gentle→strong | Nr slope | concentration (Np−Nr) g→s |
|---|---|---|---|---|---|
| OPT-350M | +0.071 → +0.157 | +0.075 | −0.007 → −0.005 | +0.001 | +0.078 → +0.162 |
| Pythia-410M | +0.074 → +0.113 | +0.044 | −0.007 → −0.007 | +0.000 | +0.082 → +0.120 |
| OPT-1.3B | +0.018 → +0.168 | +0.090 | −0.011 → −0.009 | −0.002 | +0.029 → +0.177 |
| **BLOOM-560M** | **+0.044 → −0.022** | **−0.049** | **+0.005 → +0.001** | −0.002 | **+0.040 → −0.023** |

## Two findings
1. **BLOOM's Np is clean-monotone-decreasing on the DOSE axis** (0.044, 0.004, −0.003, −0.016, −0.022),
   crossing zero at ~1.5e-6. The 3e-7 anomaly the verdict flagged is on the *retA_drop* axis only; on the
   dose axis the negative slope is NOT one bad point. Verdict stance validated.
2. **NEW — the N_random sign is itself diagnostic.** All three protectors run N_random slightly *negative*
   (−0.005…−0.011): they pay a tiny generic cost to buy big concentrated protection ON the interference
   direction. BLOOM runs N_random slightly *positive* (+0.001…+0.005) while its interference-direction
   protection goes negative. BLOOM's childhood smears mild generic protection AND actively harms the
   interference direction — the exact inverse of the concentration signature.

## Dumpable hypothesis for the fire
Protectors reorganize task-A curvature *inside* the interference subspace V (Probe-7/8). BLOOM's childhood
appears to reorganize curvature *generically / anti-aligned with V*. The geometric quantity to DUMP on the
next fire (not in banked artifacts — needs GPU): **overlap between the childhood-induced curvature change
and V** (does the childhood displacement / Fisher-shift land inside V?). Predict: high for protectors,
low/negative for BLOOM. This is the mechanism behind the sign law.

## Confound-breaker cells (compat-checked)
- **XGLM-564M** (`XGLMForCausalLM`, HF-native, 24L/1024h — shape-matched to BLOOM, multilingual CC100-XL,
  **sinusoidal** positional, OPT-named submodules → OPT carrier list drop-in). XGLM-vs-BLOOM ≈ isolates
  **ALiBi vs sinusoidal** at matched size + matched multilinguality. **First fire — cheapest, cleanest.**
- **MPT-1b-redpajama** (ALiBi, English) — isolates ALiBi across corpus, but repo is gated (needs HF_TOKEN)
  + old custom `MosaicGPT` code + 1.3B (pricier). Hold as confirmatory 2nd cell.
- **mGPT** (multilingual, learned-abs) — GPT-2 `Conv1D` weights are stored `[in,out]`, transposed vs the
  harness's `nn.Linear [out,in]` convention → needs a Conv1D fix to be valid. Redundant with XGLM. DROP.

## ★ RESULT — XGLM breaks the confound (Probe-14, N=27, LOCK b30d6579). H1 CONFIRMED. 2026-07-09.
Fired XGLM-564M: buildwa anchor 0.895 (base 0.608), probe13 dose-response N=27 (early-stopped, clean, $0). w_A gate 0.893 ✓,
144 carriers (24×6). DOSES = BLOOM's grid, for direct overlay.

| dose | XGLM N_phase2 (n=27) | XGLM N_conc | XGLM frac_child_in_V | BLOOM N_phase2 (n=50) |
|---|---|---|---|---|
| 3e-7 | +0.020 ±0.019 | +0.020 | 0.447 | +0.044 |
| 1e-6 | +0.084 ±0.014 | +0.082 | 0.434 | +0.004 |
| 2e-6 | +0.125 ±0.013 | +0.119 | 0.416 | −0.003 |
| 3e-6 | +0.112 ±0.016 | +0.114 | 0.399 | −0.016 |
| 7e-6 | **+0.149 ±0.016** | +0.152 | 0.365 | **−0.022** |

**XGLM is a clear PROTECTOR** — N_phase2 positive & growing (7e-6: +0.149, z≈9.5), N_random≈0 (concentrates, not generic),
N_conc grows to +0.152. Opposite of BLOOM at identical doses. **frac_child_in_V = 0.37–0.45 vs chance ≈0.10** → XGLM's
childhood reorganizes 3–4× above chance INSIDE V (the protector mechanism, directly confirmed).

**Confound broken:** XGLM is multilingual like BLOOM, shape-matched (24L/1024h), differs mainly on positional (sinusoidal vs
ALiBi) — and it PROTECTS. So **the multilingual corpus is NOT what makes BLOOM erode → the suspect is BLOOM's ALiBi positional
scheme (or post-embed LN).** Data: `gateA/probe13_doseresponse_xglm564m.json`; ckpt `gateA/xglm564m_sst2_wA.pt` (anchor 0.895).

## RESOLVED (see `VERDICT_PROBE14_15_CONFOUND_BREAK_2026_07_09.md`)
- **BLOOM-frac re-run** (N=25) DONE: frac_child_in_V = 0.435→0.276 — **HIGH, ≈ XGLM & Falcon.** Registered "BLOOM-low" prediction
  **REFUTED.** frac_in_V does NOT discriminate protect from erode → the mechanism is DIRECTION within V, not energy. (N_phase2
  reproduced the eroder shape +0.048→−0.032.) Orig N=50 backed up `probe13_doseresponse_bloom560m_N50_orig.json`.
- **MPT-1b DEAD, pivoted to Falcon-rw-1b** (ALiBi+English, HF-native): the `anas-awadalla` MPT mirror's 2023 mosaic_gpt custom
  code crashes on current transformers (`all_tied_weights_keys`) — ruled out by CPU smoke test, $0. `tiiuae/falcon-rw-1b`
  (alibi=True, BLOOM-scheme carriers, 72) is the clean vehicle.
- **Falcon-rw-1b (Probe-15, LOCK 623aa5bd, N=9): PROTECTS** — N_phase2 +0.04→+0.21, frac 0.20–0.46. → **ALiBi ALONE does not
  erode.** Combined with XGLM (multilingual alone protects) → the eroder is an **ALiBi×multilingual CONJUNCTION** (or BLOOM's
  post-embed LN), not either knob. See the verdict for the full 2×2.
