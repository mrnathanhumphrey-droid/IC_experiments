"""SPIRAL-OPT-1: does Pythia's depth funnel replicate on OPT-350m? CPU-only on existing OPT artifacts.
Faithful OPT swap of spiral1.py. Per-(layer,carrier) never pooled. Pre-reg PREREG_SPIRAL_OPT_2026_06_21 (eadc8832).
Three non-identical measures, pre-registered direction (Pythia funnel replicates = coarse->fine):
  (a) cross-task-B resolution = mean pairwise sim among task-Bs' seed profiles  -> predict DOWN with depth
  (b) effective dim of task-stack = eff rank of (task x dir) seed matrix        -> predict UP with depth
  (c) per-layer cross-pair partial (universality strength, contest profiles)    -> predict DOWN with depth
TRIPWIRE 3: funnel POSITIVE per measure iff rho-magnitude AND layer-shuffle z BOTH clear. Null = shuffle depth order.
RECONCILED vs Pythia (TRIPWIRE 4): OPT has 6 carriers (q/k/v/out_proj,fc1,fc2) not 3; 3 crosspair task-Bs (no 'code');
breadth index parsed from names (carrier order k,v,q,out,fc1,fc2); (c) computed from npz contest arrays (OPT json report
errored) -- definition verified to reproduce Pythia cached partial_per_layer to 8.9e-16.
"""
import numpy as np, json
from scipy.stats import rankdata

rng = np.random.default_rng(0)
NLAY = 24
CARRIERS = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
ATTN = ['q_proj', 'k_proj', 'v_proj', 'out_proj']; MLP = ['fc1', 'fc2']

def prank(x): return rankdata(x)
def pcorr(a, b): return np.corrcoef(prank(a), prank(b))[0, 1]
def partial_spear(x, y, z):
    rx, ry, rz = prank(x), prank(y), prank(z)
    rxy = np.corrcoef(rx, ry)[0, 1]; rxz = np.corrcoef(rx, rz)[0, 1]; ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2) + 1e-30)
def eff_rank(M):
    s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False); ev = s**2
    return np.nan if ev.sum() <= 0 else (ev.sum()**2) / (ev**2).sum()
def trend_null(measure, niter=5000):
    d = np.arange(len(measure)); real = np.corrcoef(prank(measure), prank(d))[0, 1]
    null = np.array([np.corrcoef(prank(measure), prank(rng.permutation(d)))[0, 1] for _ in range(niter)])
    z = (real - null.mean()) / (null.std() + 1e-30)
    p = (np.sum(np.abs(null) >= abs(real)) + 1) / (niter + 1)
    return float(real), float(z), float(p), float(null.std())

# ---------- GUARD (reproduce-known-quantity, TRIPWIRE 3): my (c) == Pythia cached partial_per_layer ----------
crP = np.load('gateA/crosspair_residual.npz', allow_pickle=True)
jP = json.load(open('gateA/crosspair_residual.json'))
_carr = 'attention.query_key_value'
_mine = np.array([partial_spear(crP[f'contest__agnews__{_carr}'][d], crP[f'contest__yelp__{_carr}'][d],
                                crP[f'lam__{_carr}'][d]) for d in range(24)])
_ref = np.array(jP['report']['contest'][_carr]['pairs']['agnews-yelp']['partial_per_layer'])
GUARD = float(np.max(np.abs(_mine - _ref)))
assert GUARD < 1e-9, f"GUARD FAILED: (c) does not reproduce Pythia ({GUARD})"

# ---------- substrate 1: breadth E (9 real task-Bs, 128 curvature-ordered dirs) ----------
bz = np.load('gateA/breadth_pass_opt.npz', allow_pickle=True)
REAL = [str(x) for x in bz['real_keys']]               # 9 tasks
names = [str(x) for x in bz['names']]
BIDX = {c: [i for i, nm in enumerate(names) if nm.endswith(c + '.weight')] for c in CARRIERS}
for c in CARRIERS: assert len(BIDX[c]) == NLAY, f"breadth idx for {c}: {len(BIDX[c])}"
def E(task, d, carr): return bz[f'E__{task}__{BIDX[carr][d]}']     # (128,)
NDIR_B = 128; dir_idx = np.arange(NDIR_B).astype(float)            # curvature-rank proxy

# ---------- substrate 2: crosspair (3 task-Bs, 64 dirs, explicit lam) ----------
cr = np.load('gateA/crosspair_residual_opt.npz', allow_pickle=True)
CP_TASKS = ['agnews', 'yelp', 'subj']                              # OPT has 3 (no 'code')
def CPseed(task, carr): return cr[f'seed__{task}__{carr}']         # (24,64)
def CPcontest(task, carr): return cr[f'contest__{task}__{carr}']   # (24,64)
def CPlam(carr): return cr[f'lam__{carr}']                         # (24,64)

out = {'guard_c_reproduces_pythia_partial_per_layer': GUARD, 'prereg': 'eadc8832',
       'carriers': CARRIERS, 'crosspair_tasks': CP_TASKS, 'substrates': {}}

def run_substrate(get_layer_taskmat, lamfn, ntask):
    res = {}
    for carr in CARRIERS:
        a_raw = np.zeros(NLAY); a_res = np.zeros(NLAY); b_raw = np.zeros(NLAY); b_res = np.zeros(NLAY)
        for d in range(NLAY):
            M = get_layer_taskmat(d, carr); ctrl = lamfn(d, carr)
            raw_p = []; res_p = []
            for i in range(ntask):
                for j in range(i + 1, ntask):
                    raw_p.append(pcorr(M[i], M[j])); res_p.append(partial_spear(M[i], M[j], ctrl))
            a_raw[d] = np.mean(raw_p); a_res[d] = np.mean(res_p)
            Mr = np.vstack([prank(M[i]) for i in range(ntask)]); b_raw[d] = eff_rank(Mr)
            Mres = np.vstack([prank(M[i]) - np.polyval(np.polyfit(prank(ctrl), prank(M[i]), 1), prank(ctrl)) for i in range(ntask)])
            b_res[d] = eff_rank(Mres)
        cres = {}
        for label, series in [('a_meanpairsim_raw', a_raw), ('a_meanpairsim_curvres', a_res),
                              ('b_effrank_raw', b_raw), ('b_effrank_curvres', b_res)]:
            real, z, p, ns = trend_null(series)
            cres[label] = {'series': series.tolist(), 'rho_vs_depth': real, 'z_vs_shuffle': z, 'p': p, 'null_std': ns}
        res[carr] = cres
    return res

out['substrates']['breadth9'] = run_substrate(
    lambda d, carr: np.vstack([E(t, d, carr) for t in REAL]), lambda d, carr: dir_idx, len(REAL))
out['substrates']['crosspair3'] = run_substrate(
    lambda d, carr: np.vstack([CPseed(t, carr)[d] for t in CP_TASKS]), lambda d, carr: CPlam(carr)[d], len(CP_TASKS))

# ---------- measure (c): per-layer cross-pair partial (universality, contest profiles) ----------
c_res = {}
for carr in CARRIERS:
    pairs = [(i, j) for i in range(len(CP_TASKS)) for j in range(i + 1, len(CP_TASKS))]
    profs = []
    for (i, j) in pairs:
        ci = CPcontest(CP_TASKS[i], carr); cj = CPcontest(CP_TASKS[j], carr); lam = CPlam(carr)
        profs.append([partial_spear(ci[d], cj[d], lam[d]) for d in range(NLAY)])
    prof = np.mean(profs, axis=0)
    real, z, p, ns = trend_null(prof)
    c_res[carr] = {'series': prof.tolist(), 'rho_vs_depth': real, 'z_vs_shuffle': z, 'p': p, 'null_std': ns}
out['c_crosspair_partial_universality'] = c_res

json.dump(out, open('gateA/spiral_opt1.json', 'w'), indent=1)

# ---------- report (TRIPWIRE 6 lede-first) ----------
def Pbench():
    try:
        P = json.load(open('gateA/spiral1.json'))
        return P
    except Exception: return None
P = Pbench()
def line(carr, lab, d):
    print(f"   {carr:9s} {lab:24s} rho={d['rho_vs_depth']:+.3f}  z_shuf={d['z_vs_shuffle']:+6.2f}  p={d['p']:.4f}")
print(f"\n=== SPIRAL-OPT-1 (guard (c) reproduces Pythia {GUARD:.1e}) ===")
print("PRE-REG: (a) DOWN, (b) UP, (c) DOWN with depth = Pythia funnel replicates\n")
for sub in ['breadth9', 'crosspair3']:
    print(f"[{sub}]")
    for carr in CARRIERS:
        for lab in ['a_meanpairsim_raw', 'a_meanpairsim_curvres', 'b_effrank_raw', 'b_effrank_curvres']:
            line(carr, lab, out['substrates'][sub][carr][lab])
    print()
print("[c] per-layer cross-pair partial (universality)")
for carr in CARRIERS: line(carr, 'c_universality', out['c_crosspair_partial_universality'][carr])
print("\nsaved gateA/spiral_opt1.json")
