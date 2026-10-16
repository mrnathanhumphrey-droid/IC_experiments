"""THREAD 2 — disentangle WIDTH / DEPTH / ASPECT-RATIO from FAMILY, on the grid already on disk (no new GPU; banked
'check existing infra before paralleling'). The size ladder spans width (d_model) and the depth-surgery spans depth at
fixed width, and two OPT/Pythia cells are BYTE-IDENTICAL in shape -> a natural architecture-matched control.

Question: does the cross-arch dissociation (G5, B-D) COLLAPSE into aspect-ratio/width, or survive shape-matching?
If it survives identical architecture -> it is the PRETRAINING recipe/data, not network geometry (and width-surgery is
unnecessary for the family question).
"""
import json, numpy as np, os
from scipy.stats import rankdata

def spear(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) > 2 else float('nan')

# ---------- grid: dims (measured from checkpoints) + G5 (geom feature_matrix) + B-D ----------
DIMS = {  # family, d_model, n_layers, d_ff
    'opt125m': ('OPT', 768, 12, 3072), 'opt350m': ('OPT', 1024, 24, 4096), 'opt1_3b': ('OPT', 2048, 24, 8192),
    'pythia160m': ('PYTHIA', 768, 12, 3072), 'pythia410m': ('PYTHIA', 1024, 24, 4096), 'pythia1b': ('PYTHIA', 2048, 16, 8192),
}
fm = json.load(open('gateA/geom_matrix.json'))['feature_matrix']
G5 = {t: fm[t]['G5'] for t in DIMS if t in fm}
if os.path.exists('gateA/geom_opt1_3b.json'):   # feature_matrix predates the 1.3b landing; pull real G5
    _b = json.load(open('gateA/geom_opt1_3b.json'))
    _s = _b.get('opt1_3b', _b).get('summary', {}) if isinstance(_b, dict) else {}
    if 'G5' in _s: G5['opt1_3b'] = _s['G5']['median']

def _findbd(o):
    if isinstance(o, dict):
        if 'B_minus_D' in o: return o['B_minus_D']
        for v in o.values():
            r = _findbd(v)
            if r is not None: return r
    if isinstance(o, list):
        for v in o:
            r = _findbd(v)
            if r is not None: return r
    return None
def bd_of(tag, *srcs):
    for s in srcs:
        p = f'gateA/{s}.json'
        if os.path.exists(p):
            r = _findbd(json.load(open(p)))
            if r is not None: return float(r[0])
    return None
BD = {'opt125m': bd_of('opt125m', 'bd_opt125m'), 'opt350m': bd_of('opt350m', 'opt_dispositive'),
      'pythia160m': bd_of('pythia160m', 'bd_pythia160m'), 'pythia410m': bd_of('pythia410m', 'pythia_dispositive'),
      'opt1_3b': None, 'pythia1b': None}

def aspect(t): return DIMS[t][1] / DIMS[t][2]

# ---------- 1. MATCHED-SHAPE family control (the killer) ----------
shape_key = lambda t: DIMS[t][1:]  # (d_model, n_layers, d_ff)
matched = {}
for t1 in DIMS:
    for t2 in DIMS:
        if t1 < t2 and shape_key(t1) == shape_key(t2) and DIMS[t1][0] != DIMS[t2][0]:
            matched[f'{t1}|{t2}'] = {
                'shape': shape_key(t1),
                'G5': {t1: G5.get(t1), t2: G5.get(t2), 'gap': (G5.get(t1, np.nan) - G5.get(t2, np.nan))},
                'BD': {t1: BD.get(t1), t2: BD.get(t2),
                       'gap': (BD[t1] - BD[t2]) if (BD.get(t1) is not None and BD.get(t2) is not None) else None}}

# ---------- 2. within-family WIDTH (d_model) trend for G5 ----------
width_trend = {}
for fam in ('OPT', 'PYTHIA'):
    cells = sorted([t for t in DIMS if DIMS[t][0] == fam and t in G5], key=lambda t: DIMS[t][1])
    pts = [(DIMS[t][1], round(G5[t], 4)) for t in cells]
    width_trend[fam] = {'points_dmodel_G5': pts, 'spearman_G5_vs_dmodel': spear([p[0] for p in pts], [p[1] for p in pts])}

# ---------- 3. ASPECT-RATIO control: same aspect, different absolute scale ----------
# opt350m-d12 (1024/12 -> 85.3) vs opt1_3b (2048/24 -> 85.3). G5(d12) from geom_depth.
g5_depth = {int(p[0]): p[1] for p in json.load(open('gateA/geom_depth.json'))['depth_trend']['G5']['points']}
aspect_ctrl = {'opt350m_d12 (1024/12, aspect 85.3)': round(g5_depth[12], 4),
               'opt1_3b (2048/24, aspect 85.3)': round(G5.get('opt1_3b', np.nan), 4),
               'same_aspect_G5_gap': round(g5_depth[12] - G5.get('opt1_3b', np.nan), 4)}

# ---------- 4. DEPTH flatness (fixed width 1024, OPT surgery) ----------
depth_flat = {f'd{d}': round(g5_depth[d], 4) for d in (6, 12, 18, 24)}
depth_flat['range'] = round(max(g5_depth[d] for d in (6,12,18,24)) - min(g5_depth[d] for d in (6,12,18,24)), 4)

out = {'dims': {t: dict(zip(['family', 'd_model', 'n_layers', 'd_ff'], DIMS[t])) for t in DIMS},
       'G5': G5, 'BD': BD, 'aspect': {t: round(aspect(t), 1) for t in DIMS},
       'matched_shape_control': matched, 'within_family_width_trend': width_trend,
       'aspect_ratio_control': aspect_ctrl, 'depth_flatness_fixed_width1024': depth_flat}
json.dump(out, open('gateA/aspect_grid.json', 'w'), indent=1)

# ---------- report ----------
print("=== THREAD 2: WIDTH / DEPTH / ASPECT vs FAMILY (existing grid, no new GPU) ===\n")
print("grid:")
print(f"  {'cell':12s}{'fam':7s}{'d_model':>8s}{'layers':>7s}{'aspect':>8s}{'G5':>8s}{'B-D':>9s}")
for t in DIMS:
    bd = BD[t]; bds = f"{bd:+.3f}" if bd is not None else "  pend"
    print(f"  {t:12s}{DIMS[t][0]:7s}{DIMS[t][1]:>8d}{DIMS[t][2]:>7d}{aspect(t):>8.1f}{G5.get(t,float('nan')):>8.3f}{bds:>9s}")

print("\n--- 1. MATCHED-SHAPE FAMILY CONTROL (byte-identical architecture) ---")
for pair, m in matched.items():
    t1, t2 = pair.split('|')
    print(f"  shape {m['shape']}  [{t1} vs {t2}]")
    print(f"     G5: {m['G5'][t1]:.3f} vs {m['G5'][t2]:.3f}  -> gap {m['G5']['gap']:+.3f}")
    if m['BD']['gap'] is not None:
        print(f"     B-D: {m['BD'][t1]:+.3f} vs {m['BD'][t2]:+.3f} -> gap {m['BD']['gap']:+.3f}")

print("\n--- 2. within-family WIDTH (d_model) trend for G5 ---")
for fam, w in width_trend.items():
    print(f"  {fam}: {w['points_dmodel_G5']}  spearman(G5,d_model)={w['spearman_G5_vs_dmodel']:+.2f}")

print("\n--- 3. ASPECT-RATIO control (same aspect 85.3, 2x absolute scale) ---")
for k, v in aspect_ctrl.items(): print(f"  {k}: {v}")

print("\n--- 4. DEPTH flatness (fixed width 1024, OPT surgery) ---")
print(f"  {depth_flat}")

print("\nVERDICT: if matched-shape gaps are large (>> within-family width/depth moves), the dissociation does NOT")
print("collapse into shape -> it is PRETRAINING recipe/data, not geometry. width-surgery then unnecessary for family.")
print("saved gateA/aspect_grid.json")
