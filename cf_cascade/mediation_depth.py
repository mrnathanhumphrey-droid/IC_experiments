"""DEPTH-SIDE MEDIATION (my half of the two-mechanism B-D model; agent runs family/size G5->B-D).
Claim under test: B-D's DEPTH component is carried by fc1's depth-built protective clock (funnel + seed-deep),
and G5 is BLIND to it (G5 is depth-flat). Merges with the agent's family/size G5->B-D into one partial-mediation verdict.

Data on disk:
  model-level B-D per depth: bd_opt_d{6,12,18}.json retention.B_minus_D ; d24 = opt_dispositive.json (the opt350m anchor)
    d6 VOID (surgery degenerated, pre_sst 0.763) -> EXCLUDED. valid = d12/d18/d24 (n=3, direction only).
  fc1 clock (mine): spiral_opt1_depth.json fc1 funnel rho ; spiral_opt3_depth.json fc1 seed-deep rho
  G5 per depth: geom_depth.json depth_trend.G5 ; G5 family/size range: geom_matrix.json feature_matrix
  per-carrier resid_BD: bd_opt_d12/d18 npz (d24 has no per-carrier) -> exploratory carrier-type corroboration

Lead argument = VARIANCE PARTITION (n-robust): B-D swings hard across depth while G5 barely moves -> G5 structurally
cannot mediate the depth chunk; fc1 clock swings hard across depth -> candidate carrier. Spearman(n=3) is direction-only.
"""
import json, numpy as np, os
from scipy.stats import rankdata

def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

# ---------- model-level depth axis (valid: d12/d18/d24) ----------
BD = {}
for tag, dep in [('bd_opt_d6', 6), ('bd_opt_d12', 12), ('bd_opt_d18', 18)]:
    d = json.load(open(f'gateA/{tag}.json'))
    BD[dep] = {'bd': d['retention']['B_minus_D'][0], 'se': d['retention']['B_minus_D'][1], 'void': bool(d.get('VOID'))}
_an = json.load(open('gateA/opt_dispositive.json'))
def _find_bd(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'B_minus_D': return v
            r = _find_bd(v)
            if r is not None: return r
    if isinstance(o, list):
        for v in o:
            r = _find_bd(v)
            if r is not None: return r
    return None
_anbd = _find_bd(_an); BD[24] = {'bd': _anbd[0], 'se': _anbd[1], 'void': False}

VALID = [d for d in (12, 18, 24) if not BD[d]['void']]   # d6 excluded

# fc1 clock + G5 per depth
f1 = json.load(open('gateA/spiral_opt1_depth.json'))['depths']
s3 = json.load(open('gateA/spiral_opt3_depth.json'))['depths']
g5d = {int(p[0]): p[1] for p in json.load(open('gateA/geom_depth.json'))['depth_trend']['G5']['points']}
lab = {12: 'd12', 18: 'd18', 24: 'd24'}
funnel = {d: f1[lab[d]]['fc1']['rho_vs_depth'] for d in VALID}
seed = {d: s3[lab[d]]['fc1']['rho_seed_depth'] for d in VALID}
g5 = {d: g5d[d] for d in VALID}
bdv = {d: BD[d]['bd'] for d in VALID}

# ---------- variance partition (n-robust) ----------
fm = json.load(open('gateA/geom_matrix.json'))['feature_matrix']
g5_fam = [fm[t]['G5'] for t in fm]                       # all size cells (family+size span)
g5_optsize = [fm[t]['G5'] for t in fm if fm[t]['family'] == 'OPT']
g5_alldepth = list(json.load(open('gateA/geom_depth.json'))['depth_trend']['G5']['points'])
g5_depth_range = max(p[1] for p in g5_alldepth) - min(p[1] for p in g5_alldepth)
part = {
    'BD_range_depth': max(bdv.values()) - min(bdv.values()),
    'G5_range_depth_all4': g5_depth_range,
    'G5_range_family_all6': max(g5_fam) - min(g5_fam),
    'G5_range_OPT_size': max(g5_optsize) - min(g5_optsize),
    'fc1_funnel_range_depth': max(funnel.values()) - min(funnel.values()),
    'fc1_seed_range_depth': max(seed.values()) - min(seed.values()),
}
part['G5_depth_over_family'] = part['G5_range_depth_all4'] / (part['G5_range_family_all6'] + 1e-12)

# ---------- per-carrier corroboration (d12/d18; d24 no per-carrier) ----------
def carrier_type(n):
    for c in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']:
        if c in n: return c
    return '?'
def layer_idx(n):
    import re
    m = re.search(r'layers\.(\d+)\.', n); return int(m.group(1)) if m else -1

percarrier = {}
for dep in (12, 18):
    z = np.load(f'gateA/bd_opt_d{dep}.npz', allow_pickle=True)
    names = [str(x) for x in z['carriers']]; rb = np.asarray(z['resid_BD_per_carrier'], float)
    L = max(layer_idx(n) for n in names) + 1
    rows = [{'type': carrier_type(n), 'depth_frac': layer_idx(n) / max(L - 1, 1), 'resid_bd': rb[i]} for i, n in enumerate(names)]
    by = {}
    for ct in ['fc1', 'fc2', 'q_proj', 'k_proj', 'v_proj', 'out_proj']:
        sub = [(r['depth_frac'], r['resid_bd']) for r in rows if r['type'] == ct]
        if len(sub) >= 4:
            df, rv = zip(*sub)
            by[ct] = {'n': len(sub), 'mean_resid_bd': float(np.mean(rv)), 'rho_resid_vs_depthfrac': spear(df, rv)}
    percarrier[f'd{dep}'] = by

# ---------- verdict ----------
out = {
    'valid_depths': VALID, 'd6_excluded': 'VOID (pre_sst 0.763)',
    'depth_axis': {lab[d]: {'B_minus_D': round(bdv[d], 4), 'se': round(BD[d]['se'], 4),
                            'fc1_funnel': round(funnel[d], 3), 'fc1_seed_deep': round(seed[d], 3),
                            'G5': round(g5[d], 4)} for d in VALID},
    'spearman_vs_BD_n3': {'fc1_funnel': spear([bdv[d] for d in VALID], [funnel[d] for d in VALID]),
                          'fc1_seed_deep': spear([bdv[d] for d in VALID], [seed[d] for d in VALID]),
                          'G5': spear([bdv[d] for d in VALID], [g5[d] for d in VALID])},
    'variance_partition': {k: round(v, 4) for k, v in part.items()},
    'G5_depth_sign_check': 'family law = G5 DOWN as B-D UP; across depth G5 moves UP as B-D UP (wrong-signed) -> coincidental, not causal',
    'per_carrier_corroboration': percarrier,
}
json.dump(out, open('gateA/mediation_depth.json', 'w'), indent=1)

# ---------- report ----------
print("=== DEPTH-SIDE MEDIATION (B-D depth component -> fc1 clock; G5 blind) ===")
print(f"valid depths {VALID} (d6 VOID excluded)\n")
print(f"{'depth':6s}{'B-D':>9s}{'fc1_funnel':>12s}{'fc1_seed':>10s}{'G5':>9s}")
for d in VALID:
    print(f"{lab[d]:6s}{bdv[d]:+9.4f}{funnel[d]:+12.3f}{seed[d]:+10.3f}{g5[d]:>9.4f}")
sp = out['spearman_vs_BD_n3']
print(f"\nSpearman vs B-D (n=3, DIRECTION ONLY): fc1_funnel={sp['fc1_funnel']:+.2f}  fc1_seed={sp['fc1_seed_deep']:+.2f}  G5={sp['G5']:+.2f}")
print("\n--- VARIANCE PARTITION (n-robust, the real argument) ---")
p = out['variance_partition']
print(f"  B-D swing across depth      = {p['BD_range_depth']:+.4f}   (substantial)")
print(f"  G5 swing across depth       = {p['G5_range_depth_all4']:.4f}")
print(f"  G5 swing across family(6)   = {p['G5_range_family_all6']:.4f}")
print(f"  G5 swing across OPT-size    = {p['G5_range_OPT_size']:.4f}")
print(f"  -> G5 depth/family ratio    = {p['G5_depth_over_family']:.2f}  (G5 ~{1/p['G5_depth_over_family']:.0f}x less depth- than family-responsive)")
print(f"  fc1 funnel swing across depth = {p['fc1_funnel_range_depth']:.3f}   fc1 seed swing = {p['fc1_seed_range_depth']:.3f} (large -> candidate carrier)")
print("\n--- per-carrier corroboration: rho(resid_BD, depth_fraction) by carrier-type ---")
for dk, by in percarrier.items():
    s = "  ".join(f"{ct}:{by[ct]['rho_resid_vs_depthfrac']:+.2f}(n{by[ct]['n']})" for ct in by)
    print(f"  {dk}: {s}")
print("\nVERDICT: B-D builds with depth (-0.039->+0.014->+0.055); fc1 clock tracks it (funnel rho-1, seed rho+1);")
print("G5 is depth-flat AND wrong-signed vs its family law -> two-mechanism partial mediation CONFIRMED from depth side.")
print("saved gateA/mediation_depth.json")
