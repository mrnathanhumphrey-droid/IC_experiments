"""SPIRAL-OPT-3-v2 (PARITY-CRITICAL): does the task-A forgetting-seed concentrate where OPT's MLP contraction is
sharpest? Per (fc1,fc2) SEPARATELY. Pre-reg PREREG_SPIRAL_OPT_V2 (59ebf808). seed/lam from perlayer_seed_opt;
contraction series from spiral_opt1 (a_meanpairsim_curvres). TRIPWIRE 3: POSITIVE iff (1) ρ(seed_conc, contraction
sharpness) clears AND (2) depth-shuffle null AND (3) MANDATORY swap fragile→topcurv survives (the lam/generic-curvature
version must NOT reproduce the trend — else it is curvature-family routing, not deposit-specific).
FORK: ρ>0 + swap survives → depth-structure LOAD-BEARING on OPT (close overturned at parity). ρ≈0 → inert (close holds).
"""
import numpy as np, json
from scipy.stats import rankdata
rng = np.random.default_rng(0)
NLAY = 24
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]; MLP = ["fc1", "fc2"]
p = np.load("gateA/perlayer_seed_opt.npz", allow_pickle=True)
S1 = json.load(open("gateA/spiral_opt1.json"))

def spear(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def shuffle_null(x, y, niter=5000):
    real = spear(x, y); idx = np.arange(len(x))
    null = np.array([spear(x, y[rng.permutation(idx)]) for _ in range(niter)])
    z = (real - null.mean()) / (null.std() + 1e-30)
    pval = (np.sum(np.abs(null) >= abs(real)) + 1) / (niter + 1)
    return real, float(z), float(pval)

def sharpness(series):  # |layer-local slope| (central diff), per locked pre-reg
    s = np.array(series); d = np.zeros_like(s)
    d[1:-1] = np.abs(s[2:] - s[:-2]) / 2; d[0] = abs(s[1] - s[0]); d[-1] = abs(s[-1] - s[-2]); return d

out = {"prereg": "59ebf808", "carriers": {}}
for ct in CARRIERS:
    seed = p[f"seed__{ct}"]; lam = p[f"lam__{ct}"]                  # (24,128)
    seed_conc = seed.sum(1)                                          # per-layer total forgetting-seed energy
    lam_conc = lam.sum(1)                                            # per-layer generic curvature scale (SWAP signal)
    frag_frac = np.array([np.sort(seed[L])[-16:].sum() / (seed[L].sum() + 1e-12) for L in range(NLAY)])  # fragile-16 occupancy
    topc_frac = np.array([seed[L][:16].sum() / (seed[L].sum() + 1e-12) for L in range(NLAY)])            # topcurv-16 occupancy
    a_cp = S1["substrates"]["crosspair3"][ct]["a_meanpairsim_curvres"]["series"]   # contraction series (clean read)
    sharp = sharpness(a_cp)
    # primary: seed concentrates where contraction sharpest
    r_sharp, z_sharp, p_sharp = shuffle_null(seed_conc, sharp)
    # interpretation rows
    r_depth, z_depth, _ = shuffle_null(seed_conc, np.arange(NLAY).astype(float))   # seed deep?
    r_val, z_val, _ = shuffle_null(seed_conc, np.array(a_cp))                       # seed where resolution low (expect <0)
    # SWAP control: lam (generic curvature) must NOT reproduce the sharpness trend
    r_lam, z_lam, _ = shuffle_null(lam_conc, sharp)
    r_fragfrac, z_fragfrac, _ = shuffle_null(frag_frac, sharp)
    r_topcfrac, z_topcfrac, _ = shuffle_null(topc_frac, sharp)
    swap_survives = bool(abs(z_sharp) > 2 and abs(z_lam) < 2)        # seed trends, generic curvature doesn't
    positive = bool(z_sharp > 2 and swap_survives)
    out["carriers"][ct] = {"rho_seed_sharp": r_sharp, "z_seed_sharp": z_sharp, "p": p_sharp,
                           "rho_seed_depth": r_depth, "z_seed_depth": z_depth,
                           "rho_seed_aval": r_val, "z_seed_aval": z_val,
                           "rho_lam_sharp": r_lam, "z_lam_sharp": z_lam,
                           "rho_fragfrac_sharp": r_fragfrac, "z_fragfrac_sharp": z_fragfrac,
                           "rho_topcfrac_sharp": r_topcfrac, "z_topcfrac_sharp": z_topcfrac,
                           "swap_survives": swap_survives, "positive": positive}

json.dump(out, open("gateA/spiral_opt3.json", "w"), indent=1)
print("=== SPIRAL-OPT-3-v2 seed-concentration vs MLP contraction (per carrier, never pooled) ===")
print("POSITIVE iff: ρ(seed,sharpness) z>2 AND swap survives (generic-curvature lam z<2)\n")
for ct in CARRIERS:
    r = out["carriers"][ct]; tag = "**MLP**" if ct in MLP else "attn"
    print(f"  {ct:9s} {tag:6s} seed~sharp ρ={r['rho_seed_sharp']:+.3f} z={r['z_seed_sharp']:+.2f} | "
          f"swap lam~sharp z={r['z_lam_sharp']:+.2f} | seed~depth ρ={r['rho_seed_depth']:+.3f} | "
          f"POSITIVE={r['positive']} (swap_survives={r['swap_survives']})")
print("\nMLP read (the parity-critical carriers; SportsThought gate):")
for ct in MLP:
    r = out["carriers"][ct]
    v = "LOAD-BEARING (seed concentrates with contraction, deposit-specific)" if r["positive"] else \
        ("seed trends but swap FAILS (generic curvature routing)" if r["z_seed_sharp"] > 2 else "INERT (no seed-concentration trend)")
    print(f"  {ct}: {v}")
print("\nsaved gateA/spiral_opt3.json")
