"""SETTLE the scale de-confound (CPU). Reads geom_basis_<TAG>.npz for the ladder and answers: does the childhood
DEPOSIT drain out of the curvature basis at scale, or is the raw out-of-V% rise a fixed-K/growing-d artifact?

Decisive column = EXCESS under the SCALE-INVARIANT energy-90% basis: median_carrier(fe90 - K_e90/in_dim).
The energy basis removes the fixed-K confound (K grows with the spectrum), so if excess_e90 is stable/growing with
d_model -> NO real drain (deposit stays curvature-enriched; the inert-map mechanism is collision, not exodus). If it
shrinks toward 0 -> real drain. Also checks the empirical Rrand null ~ analytic K/in_dim (sanity).
"""
import json, numpy as np, os
DIMS = {'opt125m': ('OPT', 768), 'opt350m': ('OPT', 1024), 'opt1_3b': ('OPT', 2048),
        'pythia160m': ('PYTHIA', 768), 'pythia410m': ('PYTHIA', 1024), 'pythia1b': ('PYTHIA', 2048)}
ORDER = ['opt125m', 'opt350m', 'opt1_3b', 'pythia160m', 'pythia410m', 'pythia1b']

rows = {}
for tag in ORDER:
    p = f'gateA/geom_basis_{tag}.npz'
    if not os.path.exists(p):
        rows[tag] = None; continue
    z = np.load(p)
    g = lambda k: np.asarray(z[f'{tag}__{k}'], float)
    med = lambda k: float(np.median(g(k)))
    excess = lambda f, n: float(np.median(g(f) - g(n)))
    rows[tag] = {
        'fam': DIMS[tag][0], 'd_model': DIMS[tag][1], 'n_carriers': len(g('f128')),
        'f128': med('f128'), 'null128': med('null128'), 'excess128': excess('f128', 'null128'),
        'fe90': med('fe90'), 'null_e90': med('null_e90'), 'excess_e90': excess('fe90', 'null_e90'),
        'fe50': med('fe50'), 'null_e50': med('null_e50'), 'excess_e50': excess('fe50', 'null_e50'),
        'frand': med('frand'), 'Ke90': med('Ke90'), 'Ke50': med('Ke50'),
        'rand_vs_analytic_null': med('frand') - med('null128'),   # ~0 validates the analytic null
    }

json.dump({k: v for k, v in rows.items()}, open('gateA/settle_basis.json', 'w'), indent=1)

print("=== SETTLE: does the deposit drain out of the curvature basis at scale? ===")
print("decisive = EXCESS_e90 (scale-invariant energy-90% basis, exact analytic null). drain => shrinks to ~0 with scale.\n")
print(f"{'cell':12s}{'d_model':>8s}{'Ke90':>6s}{'fe90':>8s}{'null_e90':>9s}{'EXCESS_e90':>11s}{'excess128':>10s}{'rand-null':>10s}")
for tag in ORDER:
    r = rows[tag]
    if r is None:
        print(f"{tag:12s}{'-- not landed --':>40s}"); continue
    print(f"{tag:12s}{r['d_model']:>8d}{r['Ke90']:>6.0f}{r['fe90']:>8.3f}{r['null_e90']:>9.3f}{r['excess_e90']:>+11.3f}{r['excess128']:>+10.3f}{r['rand_vs_analytic_null']:>+10.3f}")

# verdict per family on excess_e90 vs scale
print()
for fam in ('OPT', 'PYTHIA'):
    cells = [t for t in ORDER if rows[t] and rows[t]['fam'] == fam]
    if len(cells) >= 2:
        pts = [(rows[t]['d_model'], round(rows[t]['excess_e90'], 3)) for t in cells]
        trend = rows[cells[-1]]['excess_e90'] - rows[cells[0]]['excess_e90']
        print(f"{fam}: excess_e90 vs d_model {pts}  net {trend:+.3f}  -> {'DRAIN' if rows[cells[-1]]['excess_e90'] < 0.02 else 'NO DRAIN (stays enriched)'}")
print("\nrand-null col ~0 validates analytic K/in_dim. saved gateA/settle_basis.json")
