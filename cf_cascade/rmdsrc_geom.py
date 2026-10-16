"""RMD-SRC on the geometry matrix (Layer 3): does grouping the model cells BY FAMILY pay for each geometry feature
G1-G6 vs a random-grouping null? Faithful to B_PARSIMONY_SPEC_FOR_REIMPL.md (float_module): per-unit OLS vs
group-shared OLS, held-out Gaussian log-likelihood delta, bootstrap CI, K random groupings -> perm_p + strict verdict.

WIRING (the geom_big doc says the matrix is FOR this test; it was never wired):
  UNIT  = model cell (opt125m, opt350m, [opt1_3b], pythia160m, pythia410m, pythia1b)
  ROWS  = that model's per-carrier feature values (npz arrays, the bootstrap unit in geom_big)
  GROUP = family (OPT vs PYTHIA)   [also reported: size-band grouping]
  X     = intercept only  (y = feature Gi per carrier; tests whether family pooling retains held-out ll)
  y     = each of G1..G6 separately

This PROMOTES geom_big's bootstrap-SE "classifies_family" flag (separation magnitude) to a parsimony+permutation
footing (does the family PARTITION specifically pay, beating random partitions) — discharging the banked caution
"a class-separation SE is not a prediction-claim SE".

opt1_3b HOLE (banked discipline): run (a) cells-present; if opt1_3b absent also bound with FAVORABLE (OPT-low, clone
opt350m carriers) and UNFAVORABLE (PY-high, clone pythia410m carriers) imputations. A verdict that only survives under
favorable imputation is DISQUALIFIED.
"""
import json, sys, numpy as np

NPZ = 'gateA/geom_matrix.npz'; META = 'gateA/geom_matrix.json'
METRICS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
TRAIN_FRAC = 0.8; N_BOOT = 500; K_RAND = 200; SEED = 0

FAMILY = {'opt125m': 'OPT', 'opt350m': 'OPT', 'opt1_3b': 'OPT',
          'pythia160m': 'PYTHIA', 'pythia410m': 'PYTHIA', 'pythia1b': 'PYTHIA'}
SIZE = {'opt125m': 125, 'opt350m': 350, 'opt1_3b': 1300, 'pythia160m': 160, 'pythia410m': 410, 'pythia1b': 1000}
def size_band(m): return 'S' if m < 300 else ('M' if m < 700 else 'L')


import os
OPT13B_NPZ = 'gateA/geom_opt1_3b.npz'  # the re-fired big cell lands its per-carrier arrays here (separate file)

def _merge_arrays(z, panel):
    tags = [t for t in sorted(set(k.split('__')[0] for k in z.files if '__' in k)) if t in FAMILY]
    for g in METRICS:
        for t in tags:
            key = f'{t}__{g}'
            if key in z.files:
                arr = np.asarray(z[key], float); arr = arr[~np.isnan(arr)]
                if len(arr) >= 2: panel[g][t] = arr

def load_panel(npz, extra=None):
    """Return {metric: {unit: np.array(carrier values)}} for present cells (+ optional imputed extra cell).
    Merges the separately-saved real opt1_3b per-carrier arrays if that re-fire has landed (drops the bracket)."""
    panel = {g: {} for g in METRICS}
    _merge_arrays(np.load(npz), panel)
    if os.path.exists(OPT13B_NPZ):
        _merge_arrays(np.load(OPT13B_NPZ), panel)   # real opt1_3b cell -> overrides any imputation
    if extra:  # extra = (unit_name, source_tag) -> clone source carriers as the imputed cell
        name, src = extra
        for g in METRICS:
            if src in panel[g]:
                panel[g][name] = panel[g][src].copy()
    return panel


def _ols_int_ll(y_tr, y_ho):
    """intercept-only OLS: yhat = mean(train); sigma2 = train MSE; per-point Gaussian ll on holdout."""
    mu = float(np.mean(y_tr)); s2 = max(float(np.mean((y_tr - mu) ** 2)), 1e-9)
    return -0.5 * np.log(2 * np.pi * s2) - (y_ho - mu) ** 2 / (2 * s2)


def split(units_vals, seed):
    """stratified per-unit train/holdout; returns dict unit -> (idx_tr, idx_ho) on its own array."""
    rng = np.random.default_rng(seed); sp = {}
    for u, arr in units_vals.items():
        idx = rng.permutation(len(arr)); ncut = max(1, int(round((1 - TRAIN_FRAC) * len(arr))))
        sp[u] = (idx[ncut:], idx[:ncut])
    return sp


def run_grouping(units_vals, group_of, seed=SEED):
    """one parsimony pass for a given grouping. Returns delta + per-point deltas."""
    sp = split(units_vals, seed)
    # per-unit held-out ll
    pu_ll = {};
    for u, arr in units_vals.items():
        tr, ho = sp[u]
        pu_ll[u] = _ols_int_ll(arr[tr], arr[ho])
    # group-shared: pool each group's train rows
    groups = {}
    for u in units_vals: groups.setdefault(group_of[u], []).append(u)
    gr_ll = {}
    for gname, us in groups.items():
        tr_pool = np.concatenate([units_vals[u][sp[u][0]] for u in us])
        mu = float(np.mean(tr_pool)); s2 = max(float(np.mean((tr_pool - mu) ** 2)), 1e-9)
        for u in us:
            ho = sp[u][1]; gr_ll[u] = -0.5 * np.log(2 * np.pi * s2) - (units_vals[u][ho] - mu) ** 2 / (2 * s2)
    dpt = np.concatenate([gr_ll[u] - pu_ll[u] for u in units_vals])
    return float(dpt.sum()), dpt, len(groups)


def verdict(units_vals, group_of, label, seed=SEED):
    delta, dpt, ngrp = run_grouping(units_vals, group_of, seed)
    rng = np.random.default_rng(seed + 7)
    lo, hi = np.percentile([dpt[rng.integers(0, len(dpt), len(dpt))].sum() for _ in range(N_BOOT)], [2.5, 97.5])
    nunit = len(units_vals); pu_p = nunit; gr_p = ngrp; reduced = gr_p < pu_p
    base = ('OUTPERFORMS' if lo > 0 else 'COSTS' if hi < 0 else ('PAYS' if reduced else 'MATCHES'))
    # K random groupings (uniform assign to ngrp groups, same seed family)
    units = list(units_vals); rr = np.random.default_rng(seed + 13); ge = 0
    for _ in range(K_RAND):
        rg = {u: int(rr.integers(0, ngrp)) for u in units}
        if len(set(rg.values())) < ngrp:  # degenerate (a group empty) -> reassign one
            for i, u in enumerate(units): rg[u] = i % ngrp
        dr, _, _ = run_grouping(units_vals, rg, seed)
        if dr >= delta: ge += 1
    perm_p = (ge + 1) / (K_RAND + 1)
    strict = ('COSTS' if base == 'COSTS' else
              ('PAYS_STRICT' if (base in ('PAYS', 'OUTPERFORMS', 'MATCHES') and perm_p <= 0.10) else 'PAYS_BUT_NOT_SPECIFIC'))
    return {'label': label, 'n_units': nunit, 'n_groups': ngrp, 'delta': delta, 'ci': [float(lo), float(hi)],
            'reduced': bool(reduced), 'base': base, 'perm_p': float(perm_p), 'strict': strict}


def scenario(name, extra):
    panel = load_panel(NPZ, extra)
    units_present = sorted(set().union(*[set(panel[g]) for g in METRICS]))
    fam_of = {u: FAMILY[u] for u in units_present}
    band_of = {u: size_band(SIZE[u]) for u in units_present}
    out = {'cells': units_present, 'family': {}, 'sizeband': {}}
    for g in METRICS:
        uv = panel[g]
        if len(uv) >= 4 and len(set(fam_of[u] for u in uv)) == 2:
            out['family'][g] = verdict(uv, {u: fam_of[u] for u in uv}, f'{g}~family')
        if len(uv) >= 4 and len(set(band_of[u] for u in uv)) >= 2:
            out['sizeband'][g] = verdict(uv, {u: band_of[u] for u in uv}, f'{g}~sizeband')
    return out


def main():
    z = np.load(NPZ); have_13b = any(k.startswith('opt1_3b__') for k in z.files)
    if os.path.exists(OPT13B_NPZ): have_13b = have_13b or any(k.startswith('opt1_3b__') for k in np.load(OPT13B_NPZ).files)
    scenarios = {'present': None}
    if not have_13b:
        scenarios['impute_favorable_OPTlow'] = ('opt1_3b', 'opt350m')
        scenarios['impute_unfavorable_PYhigh'] = ('opt1_3b', 'pythia410m')
    res = {'have_opt1_3b': bool(have_13b), 'spec': 'B_PARSIMONY_SPEC_FOR_REIMPL.md', 'scenarios': {}}
    for sname, extra in scenarios.items():
        res['scenarios'][sname] = scenario(sname, extra)
    json.dump(res, open('gateA/rmdsrc_geom.json', 'w'), indent=1)

    print(f"=== RMD-SRC on geometry matrix (have_opt1_3b={have_13b}) ===")
    print("does FAMILY grouping PAY per feature vs random groupings (perm_p<=0.10 = PAYS_STRICT)\n")
    for sname, sc in res['scenarios'].items():
        print(f"[{sname}]  cells={sc['cells']}")
        for g in METRICS:
            f = sc['family'].get(g)
            if f:
                print(f"   {g}~family  delta={f['delta']:+.1f} CI[{f['ci'][0]:+.1f},{f['ci'][1]:+.1f}] "
                      f"base={f['base']:11s} perm_p={f['perm_p']:.3f} -> {f['strict']}")
        print()
    print("saved gateA/rmdsrc_geom.json")
    print("NOTE: B-D-sign grouping NOT run — per-cell B-D exists only at the ~350M band (opt350m/pythia410m);")
    print("      'G5 predicts B-D' across the ladder needs per-cell dispositive B-D (follow-on data need).")


if __name__ == '__main__':
    main()
