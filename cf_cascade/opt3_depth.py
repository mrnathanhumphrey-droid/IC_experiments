"""SPIRAL-OPT-3 seed-concentration vs DEPTH (layer-surgery 6/12/18/24): does the task-A forgetting-seed still
concentrate DEEP (banked load-bearing signal: fc1 rho_seed_depth=+0.648, fc2=+0.923) as decoder depth drops, and does
the swap control (generic-curvature lam) still stay inert? Faithful slice of opt3.py per depth. Per-carrier never pooled.
Contraction series per depth = spiral_opt1_depth.json (the depth funnel I already computed). Per-layer never pooled.
REPRODUCE-ANCHOR (T3): d24 fc1 rho_seed_depth == banked +0.648 (deterministic; rng only moves z).
"""
import numpy as np, json, os
from scipy.stats import rankdata

CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]; MLP = ["fc1", "fc2"]
DEPTHS = [('d6', 'gateA/perlayer_seed_optd6.npz'),
          ('d12', 'gateA/perlayer_seed_optd12.npz'),
          ('d18', 'gateA/perlayer_seed_optd18.npz'),
          ('d24', 'gateA/perlayer_seed_opt.npz')]
S1D = json.load(open('gateA/spiral_opt1_depth.json'))   # depth funnel series (per depth, per carrier)

def spear(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def shuffle_null(x, y, rng, niter=5000):
    real = spear(x, y); idx = np.arange(len(x))
    null = np.array([spear(x, y[rng.permutation(idx)]) for _ in range(niter)])
    z = (real - null.mean()) / (null.std() + 1e-30)
    pval = (np.sum(np.abs(null) >= abs(real)) + 1) / (niter + 1)
    return real, float(z), float(pval)
def sharpness(series):
    s = np.array(series); d = np.zeros_like(s)
    if len(s) >= 3: d[1:-1] = np.abs(s[2:] - s[:-2]) / 2
    d[0] = abs(s[1] - s[0]); d[-1] = abs(s[-1] - s[-2]); return d

out = {'measure': 'forgetting-seed concentration vs depth + vs contraction-sharpness; swap=lam', 'depths': {}}
present = []
for lbl, path in DEPTHS:
    if not os.path.exists(path):
        out['depths'][lbl] = {'status': 'NOT_LANDED'}; continue
    p = np.load(path, allow_pickle=True)
    nlay = p['seed__fc1'].shape[0]
    rng = np.random.default_rng(0)
    row = {'nlay': int(nlay)}
    for ct in CARRIERS:
        seed = p[f'seed__{ct}']; lam = p[f'lam__{ct}']
        seed_conc = seed.sum(1); lam_conc = lam.sum(1)
        a_cp = S1D['depths'][lbl][ct]['series']          # depth-matched contraction series
        sharp = sharpness(a_cp)
        r_sharp, z_sharp, p_sharp = shuffle_null(seed_conc, sharp, rng)
        r_depth, z_depth, _ = shuffle_null(seed_conc, np.arange(nlay).astype(float), rng)
        r_lam, z_lam, _ = shuffle_null(lam_conc, sharp, rng)
        swap_survives = bool(abs(z_sharp) > 2 and abs(z_lam) < 2)
        positive = bool(z_sharp > 2 and swap_survives)
        row[ct] = {'rho_seed_sharp': r_sharp, 'z_seed_sharp': z_sharp, 'p': p_sharp,
                   'rho_seed_depth': r_depth, 'z_seed_depth': z_depth,
                   'rho_lam_sharp': r_lam, 'z_lam_sharp': z_lam,
                   'swap_survives': swap_survives, 'positive': positive}
    out['depths'][lbl] = row; present.append(lbl)

anchor_ok = None
if 'd24' in present:
    fc1_24 = out['depths']['d24']['fc1']['rho_seed_depth']
    anchor_ok = abs(fc1_24 - 0.648) < 0.02
    out['anchor'] = {'d24_fc1_rho_seed_depth': fc1_24, 'banked': 0.648, 'ok': bool(anchor_ok)}

json.dump(out, open('gateA/spiral_opt3_depth.json', 'w'), indent=1)

print("=== SPIRAL-OPT-3 seed-concentration vs DEPTH ===")
if anchor_ok is not None:
    print(f"ANCHOR: d24 fc1 rho_seed_depth = {out['anchor']['d24_fc1_rho_seed_depth']:+.3f} vs banked +0.648 -> "
          f"{'REPRODUCED' if anchor_ok else 'MISMATCH!!'}\n")
print(f"{'carrier':9s}" + "".join(f"{l:>26s}" for l, _ in DEPTHS))
for ct in CARRIERS:
    tag = '*' if ct in MLP else ' '
    cells = []
    for lbl, _ in DEPTHS:
        dd = out['depths'].get(lbl, {})
        if dd.get('status') == 'NOT_LANDED' or ct not in dd:
            cells.append(f"{'--':>26s}")
        else:
            v = dd[ct]
            cells.append(f"dep{v['rho_seed_depth']:+.2f} shp{v['rho_seed_sharp']:+.2f}/z{v['z_seed_sharp']:+.1f}".rjust(26))
    print(f"{tag}{ct:8s}" + "".join(cells))
print("\ndep = rho(seed_conc, depth) [banked load-bearing signal: seed concentrates DEEP];")
print("shp = rho(seed_conc, contraction-sharpness)/z [the parity test]; swap = lam-sharp must stay inert.")
print("MLP carriers fc1/fc2 = SportsThought parity gate.")
print("saved gateA/spiral_opt3_depth.json")
