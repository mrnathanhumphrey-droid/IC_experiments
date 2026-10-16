"""SPIRAL-OPT-4-v2: does OPT's MLP funnel TWIST (consecutive-subspace rotation), or contract-only?
Per (fc1, fc2) SEPARATELY (never pooled). fragile__sst2__<ct> (24,D,16) from taskA_variation_opt. Pre-reg
PREREG_SPIRAL_OPT_V2 (59ebf808). TRIPWIRE 3: twist POSITIVE iff consecutive-coherence present (vs depth-shuffle)
AND ρ_gap (overlap decays with depth-gap) clears its depth-shuffle null. Within-OPT prediction: coherent rotation
decaying with gap = true spiral (contract+twist); flat = contract-only.
"""
import numpy as np, json
from scipy.stats import rankdata
rng = np.random.default_rng(0)
NLAY = 24; GMAX = 12
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]; MLP = ["fc1", "fc2"]
t = np.load("gateA/taskA_variation_opt.npz", allow_pickle=True)

def overlap(A, B):  # mean cos(principal angles) between orthonormal bases (D,k)
    return float(np.linalg.svd(A.T @ B, compute_uv=False).mean())

def spear(a, b):
    ra, rb = rankdata(a), rankdata(b); return float(np.corrcoef(ra, rb)[0, 1])

out = {"prereg": "59ebf808", "carriers": {}}
for ct in CARRIERS:
    V = t[f"fragile__sst2__{ct}"]  # (24,D,16) orthonormal per layer
    # consecutive coherence (adjacent-layer subspace overlap)
    consec = np.array([overlap(V[L], V[L + 1]) for L in range(NLAY - 1)])
    coh_real = float(consec.mean())
    # gap-overlap profile: mean overlap at each depth-gap g
    gap_ov = np.array([np.mean([overlap(V[L], V[L + g]) for L in range(NLAY - g)]) for g in range(1, GMAX + 1)])
    rho_gap = spear(np.arange(1, GMAX + 1), gap_ov)   # <0 = decays with gap = twist
    # depth-shuffle null: shuffle layer order, recompute coh_real and rho_gap
    nl = 2000; coh_null = np.empty(nl); rg_null = np.empty(nl)
    for i in range(nl):
        perm = rng.permutation(NLAY); Vs = V[perm]
        coh_null[i] = np.mean([overlap(Vs[L], Vs[L + 1]) for L in range(NLAY - 1)])
        go = np.array([np.mean([overlap(Vs[L], Vs[L + g]) for L in range(NLAY - g)]) for g in range(1, GMAX + 1)])
        rg_null[i] = spear(np.arange(1, GMAX + 1), go)
    z_coh = (coh_real - coh_null.mean()) / (coh_null.std() + 1e-30)
    z_gap = (rho_gap - rg_null.mean()) / (rg_null.std() + 1e-30)
    twist = bool(z_coh > 2 and rho_gap < 0 and z_gap < -2)   # coherent AND decays with gap, both vs shuffle
    out["carriers"][ct] = {"coh_consecutive": coh_real, "coh_shuffle_mean": float(coh_null.mean()), "z_coh": float(z_coh),
                           "rho_gap": float(rho_gap), "rho_gap_shuffle_mean": float(rg_null.mean()), "z_gap": float(z_gap),
                           "gap_overlap_profile": gap_ov.tolist(), "twist": twist}

json.dump(out, open("gateA/spiral_opt4.json", "w"), indent=1)
print("=== SPIRAL-OPT-4-v2 twist (per carrier, never pooled) ===")
print("POSITIVE iff: consecutive coherence vs shuffle (z_coh>2) AND gap-overlap decays (rho_gap<0, z_gap<-2)\n")
for ct in CARRIERS:
    r = out["carriers"][ct]; tag = "**MLP**" if ct in MLP else "attn"
    print(f"  {ct:9s} {tag:6s} coh={r['coh_consecutive']:.3f}(shuf {r['coh_shuffle_mean']:.3f} z={r['z_coh']:+.2f}) "
          f"rho_gap={r['rho_gap']:+.3f}(z={r['z_gap']:+.2f}) -> TWIST={r['twist']}")
print("\nMLP read (the OPT funnel carriers):")
for ct in MLP:
    r = out["carriers"][ct]
    verdict = "TWIST (spiral: contract+twist)" if r["twist"] else "contract-only (smooth funnel, no twist)"
    print(f"  {ct}: {verdict}")
print("\nsaved gateA/spiral_opt4.json")
