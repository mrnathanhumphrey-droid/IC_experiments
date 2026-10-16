"""SPIRAL-OPT-4 twist vs DEPTH (layer-surgery 6/12/18/24): does the fc1 spiral (coh=0.702, rho_gap=-1.0, TWIST) and
fc2 contract-only persist / change as decoder depth drops? Faithful slice of opt4.py per depth. Per-carrier never pooled.
GMAX = min(12, nlay-1) per depth (coh_consecutive is the depth-comparable scalar; rho_gap range is per-depth, noted).
REPRODUCE-ANCHOR (T3): d24 fc1 coh == banked 0.7024 (coh/rho_gap deterministic; rng only moves z).
"""
import numpy as np, json, os
from scipy.stats import rankdata

CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]; MLP = ["fc1", "fc2"]
DEPTHS = [('d6', 'gateA/taskA_variation_optd6.npz'),
          ('d12', 'gateA/taskA_variation_optd12.npz'),
          ('d18', 'gateA/taskA_variation_optd18.npz'),
          ('d24', 'gateA/taskA_variation_opt.npz')]

def overlap(A, B): return float(np.linalg.svd(A.T @ B, compute_uv=False).mean())
def spear(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

def carrier_twist(V, rng, gmax):
    nlay = V.shape[0]
    consec = np.array([overlap(V[L], V[L + 1]) for L in range(nlay - 1)])
    coh_real = float(consec.mean())
    gap_ov = np.array([np.mean([overlap(V[L], V[L + g]) for L in range(nlay - g)]) for g in range(1, gmax + 1)])
    rho_gap = spear(np.arange(1, gmax + 1), gap_ov) if gmax >= 2 else float('nan')
    nl = 2000; coh_null = np.empty(nl); rg_null = np.empty(nl)
    for i in range(nl):
        perm = rng.permutation(nlay); Vs = V[perm]
        coh_null[i] = np.mean([overlap(Vs[L], Vs[L + 1]) for L in range(nlay - 1)])
        if gmax >= 2:
            go = np.array([np.mean([overlap(Vs[L], Vs[L + g]) for L in range(nlay - g)]) for g in range(1, gmax + 1)])
            rg_null[i] = spear(np.arange(1, gmax + 1), go)
        else:
            rg_null[i] = np.nan
    z_coh = (coh_real - coh_null.mean()) / (coh_null.std() + 1e-30)
    z_gap = (rho_gap - np.nanmean(rg_null)) / (np.nanstd(rg_null) + 1e-30) if gmax >= 2 else float('nan')
    twist = bool(z_coh > 2 and rho_gap < 0 and z_gap < -2) if gmax >= 2 else None
    return {'coh': coh_real, 'z_coh': float(z_coh), 'rho_gap': float(rho_gap), 'z_gap': float(z_gap),
            'gmax': gmax, 'twist': twist}

out = {'measure': 'fragile__sst2 subspace twist (consecutive coherence + gap decay)', 'depths': {}}
present = []
for lbl, path in DEPTHS:
    if not os.path.exists(path):
        out['depths'][lbl] = {'status': 'NOT_LANDED'}; continue
    t = np.load(path, allow_pickle=True)
    nlay = t['fragile__sst2__fc1'].shape[0]; gmax = min(12, nlay - 1)
    rng = np.random.default_rng(0)
    row = {'nlay': int(nlay)}
    for ct in CARRIERS:
        row[ct] = carrier_twist(t[f'fragile__sst2__{ct}'], rng, gmax)
    out['depths'][lbl] = row; present.append(lbl)

anchor_ok = None
if 'd24' in present:
    fc1_24 = out['depths']['d24']['fc1']['coh']
    anchor_ok = abs(fc1_24 - 0.7024) < 0.005
    out['anchor'] = {'d24_fc1_coh': fc1_24, 'banked': 0.7024, 'ok': bool(anchor_ok)}

json.dump(out, open('gateA/spiral_opt4_depth.json', 'w'), indent=1)

print("=== SPIRAL-OPT-4 twist vs DEPTH (fragile__sst2 consecutive coherence) ===")
if anchor_ok is not None:
    print(f"ANCHOR: d24 fc1 coh = {out['anchor']['d24_fc1_coh']:.4f} vs banked 0.7024 -> {'REPRODUCED' if anchor_ok else 'MISMATCH!!'}\n")
print(f"{'carrier':9s}" + "".join(f"{l:>22s}" for l, _ in DEPTHS))
for ct in CARRIERS:
    tag = '*' if ct in MLP else ' '
    cells = []
    for lbl, _ in DEPTHS:
        dd = out['depths'].get(lbl, {})
        if dd.get('status') == 'NOT_LANDED' or ct not in dd:
            cells.append(f"{'--':>22s}")
        else:
            v = dd[ct]; cells.append(f"coh{v['coh']:.2f}/zc{v['z_coh']:+.1f}/rg{v['rho_gap']:+.2f}".rjust(22))
    print(f"{tag}{ct:8s}" + "".join(cells))
print("\ncoh = mean adjacent-layer fragile-subspace overlap (depth-comparable); zc = vs depth-shuffle;")
print("rg = rho_gap (overlap vs depth-gap, <0 = decays = twist; range = 1..min(12,nlay-1), per-depth).")
print("MLP carriers = fc1/fc2. fc1 banked = spiral (coh0.70,rg-1.0,TWIST); fc2 = contract-only.")
print("saved gateA/spiral_opt4_depth.json")
