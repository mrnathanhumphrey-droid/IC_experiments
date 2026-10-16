"""SPIRAL-OPT-1 vs DEPTH: does the OPT MLP funnel (crosspair3 a_meanpairsim_curvres, banked fc1 rho=-0.837)
strengthen / weaken / hold as decoder depth drops via layer-surgery (6/12/18/24)?
CPU-only, crosspair substrate ONLY (the funnel signal lives there; no breadth needed). Per-(layer,carrier) never pooled.
Faithful slice of opt1.py run_substrate (a_meanpairsim_curvres branch) applied per depth.
REPRODUCE-ANCHOR (T3): full-24 fc1 a_meanpairsim_curvres must == banked -0.837 (rho is deterministic; rng only moves z/p).
"""
import numpy as np, json, sys
from scipy.stats import rankdata

CARRIERS = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
CP_TASKS = ['agnews', 'yelp', 'subj']
DEPTHS = [('d6', 'gateA/crosspair_residual_optd6.npz'),
          ('d12', 'gateA/crosspair_residual_optd12.npz'),
          ('d18', 'gateA/crosspair_residual_optd18.npz'),
          ('d24', 'gateA/crosspair_residual_opt.npz')]  # full-24 anchor

def prank(x): return rankdata(x)
def pcorr(a, b): return np.corrcoef(prank(a), prank(b))[0, 1]
def partial_spear(x, y, z):
    rx, ry, rz = prank(x), prank(y), prank(z)
    rxy = np.corrcoef(rx, ry)[0, 1]; rxz = np.corrcoef(rx, rz)[0, 1]; ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2) + 1e-30)
def trend_null(measure, rng, niter=5000):
    d = np.arange(len(measure)); real = np.corrcoef(prank(measure), prank(d))[0, 1]
    null = np.array([np.corrcoef(prank(measure), prank(rng.permutation(d)))[0, 1] for _ in range(niter)])
    z = (real - null.mean()) / (null.std() + 1e-30)
    p = (np.sum(np.abs(null) >= abs(real)) + 1) / (niter + 1)
    return float(real), float(z), float(p), float(null.std())

def funnel_curvres(cr, carr, nlay):
    """a_meanpairsim_curvres series over layers for one carrier: mean over task-pairs of partial_spear(seed_i,seed_j|lam)."""
    seed = {t: cr[f'seed__{t}__{carr}'] for t in CP_TASKS}     # (nlay,64)
    lam = cr[f'lam__{carr}']                                   # (nlay,64)
    series = np.zeros(nlay)
    for d in range(nlay):
        M = np.vstack([seed[t][d] for t in CP_TASKS]); ctrl = lam[d]
        res_p = [partial_spear(M[i], M[j], ctrl) for i in range(len(CP_TASKS)) for j in range(i + 1, len(CP_TASKS))]
        series[d] = np.mean(res_p)
    return series

out = {'measure': 'crosspair3 a_meanpairsim_curvres (MLP funnel)', 'depths': {}}
present = []
for lbl, path in DEPTHS:
    try:
        cr = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        out['depths'][lbl] = {'status': 'NOT_LANDED'}; continue
    nlay = cr['lam__fc1'].shape[0]
    rng = np.random.default_rng(0)  # same per depth; rho deterministic regardless
    row = {'nlay': int(nlay)}
    for carr in CARRIERS:
        series = funnel_curvres(cr, carr, nlay)
        real, z, p, ns = trend_null(series, rng)
        row[carr] = {'series': series.tolist(), 'rho_vs_depth': real, 'z_shuf': z, 'p': p, 'null_std': ns}
    out['depths'][lbl] = row
    present.append(lbl)

# ---------- REPRODUCE-ANCHOR ----------
anchor_ok = None
if 'd24' in present:
    fc1_24 = out['depths']['d24']['fc1']['rho_vs_depth']
    anchor_ok = abs(fc1_24 - (-0.837)) < 0.01
    out['anchor'] = {'d24_fc1_rho': fc1_24, 'banked': -0.837, 'ok': bool(anchor_ok)}

json.dump(out, open('gateA/spiral_opt1_depth.json', 'w'), indent=1)

# ---------- lede-first report ----------
print("=== SPIRAL-OPT-1 funnel vs DEPTH (crosspair3 a_meanpairsim_curvres) ===")
if anchor_ok is not None:
    print(f"ANCHOR: d24 fc1 rho = {out['anchor']['d24_fc1_rho']:+.3f} vs banked -0.837 -> {'REPRODUCED' if anchor_ok else 'MISMATCH!!'}\n")
hdr = "carrier   " + "".join(f"{l+'(n)':>14s}" for l, _ in DEPTHS)
print(hdr)
for carr in CARRIERS:
    cells = []
    for lbl, _ in DEPTHS:
        dd = out['depths'].get(lbl, {})
        if dd.get('status') == 'NOT_LANDED' or carr not in dd:
            cells.append(f"{'--':>14s}")
        else:
            v = dd[carr]; cells.append(f"{v['rho_vs_depth']:+.2f}/z{v['z_shuf']:+.1f}".rjust(14))
    print(f"{carr:9s} " + "".join(cells))
print(f"\n(n = nlay; rho = spearman(curvres-funnel, layer-depth) within that variant; z = vs layer-shuffle)")
print("MLP funnel = fc1/fc2 rows. Read: does the within-variant funnel hold as total depth drops?")
print("saved gateA/spiral_opt1_depth.json")
