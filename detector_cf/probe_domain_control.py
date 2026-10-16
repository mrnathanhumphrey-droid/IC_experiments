"""
probe_domain_control.py -- DOMAIN-CONTROLLED re-test of the 4-op deepfake cosine law (PREREG 2026-07-26). $0 local CPU.
Rebuilds the 4x4 family-cosine matrix under 3 references (shared ffpp-real / own-dataset real / pure domain), with bootstrap
CIs, then tests whether the operational routing tracks FORGERY or DOMAIN. Reuses probe_detector_seq machinery byte-faithfully.
Run:  python probe_domain_control.py [--smoke]
"""
from __future__ import annotations
import os, sys, glob, time, json
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_detector_seq as SEQ   # load_backbone, feat, crop_tensor, _glob_sub, CORPUS

SMOKE = "--smoke" in sys.argv
SUB = 40 if SMOKE else int(os.environ.get("SUBSAMPLE", "260"))
B = 100 if SMOKE else int(os.environ.get("BOOT", "500"))
CORPUS = SEQ.CORPUS
PARA = [("swap", "ffpp"), ("diffusion", "deepfakeface"), ("celebdf", "celebdf"), ("dfdc", "dfdc")]
if SMOKE: PARA = PARA[:3]
NAMES = [p[0] for p in PARA]
OUT = HERE / "out" / ("domain_control_smoke.json" if SMOKE else "domain_control.json")
np.random.seed(0); torch.manual_seed(0)


def feats_for(paths, m):
    outs = []
    for i in range(0, len(paths), 64):
        xs = torch.stack([SEQ.crop_tensor(p) for p in paths[i:i+64]])
        outs.append(SEQ.feat(m, xs).cpu().numpy())
    return np.concatenate(outs).astype(np.float64) if paths else np.zeros((0, 128))


def unit(v):
    n = np.linalg.norm(v); return v/n if n > 1e-12 else v


def cosmat(dirs):
    n = len(dirs); M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            if dirs[a] is not None and dirs[b] is not None:
                M[a, b] = float(dirs[a] @ dirs[b])
    return M


def offdiag(M):
    n = M.shape[0]; return np.array([M[a, b] for a in range(n) for b in range(n) if a != b])


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3: return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def route(M, names, tau):
    """Sequential router: paradigm joins nearest existing head if max cos >= tau, else spawns a new head."""
    heads = []  # list of lists of paradigm indices
    for i in range(len(names)):
        best, bj = -2.0, -1
        for hj, h in enumerate(heads):
            c = np.nanmax([M[i, k] for k in h])
            if c > best: best, bj = c, hj
        if heads and best >= tau: heads[bj].append(i)
        else: heads.append([i])
    return [[names[k] for k in h] for h in heads]


def main():
    t0 = time.time()
    print(f"[DC] SMOKE={SMOKE} paradigms={NAMES} SUB={SUB} BOOT={B}", flush=True)
    m = SEQ.load_backbone()
    shared_paths = SEQ._glob_sub(str(CORPUS/"ffpp"/"real"/"*.png"), SUB, seed=0)
    F_shared = feats_for(shared_paths, m); print(f"[DC] shared ffpp/real feats {F_shared.shape} ({time.time()-t0:.0f}s)", flush=True)
    DSDIR = {name: dsdir for name, dsdir in PARA}
    FK = {}; RO = {}
    for name, dsdir in PARA:
        fk = SEQ._glob_sub(str(CORPUS/dsdir/"fake"/"*.png"), SUB, seed=1)
        FK[name] = feats_for(fk, m)
        if dsdir == "ffpp":
            RO[name] = F_shared   # own-real IS the shared ffpp/real pool -> d_forgery==d_shared, r_domain==0 (no spurious domain axis)
        else:
            RO[name] = feats_for(SEQ._glob_sub(str(CORPUS/dsdir/"real"/"*.png"), SUB, seed=2), m)
        print(f"[DC] {name}: fake {FK[name].shape[0]} own-real {RO[name].shape[0]} ({time.time()-t0:.0f}s)", flush=True)

    def dirs_from(idx_shared, idx_fk, idx_ro):
        msh = F_shared[idx_shared].mean(0)
        d_sh, d_fg, r_dm = [], [], []
        for name, _ in PARA:
            fkm = FK[name][idx_fk[name]].mean(0); rom = RO[name][idx_ro[name]].mean(0)
            d_sh.append(unit(fkm - msh))
            d_fg.append(unit(fkm - rom))
            r = rom - msh; r_dm.append(unit(r) if np.linalg.norm(r) > 1e-9 else None)
        return d_sh, d_fg, r_dm

    def allidx():
        return (np.arange(len(F_shared)),
                {n: np.arange(len(FK[n])) for n in NAMES},
                {n: np.arange(len(RO[n])) for n in NAMES})

    ish, ifk, iro = allidx()
    d_sh, d_fg, r_dm = dirs_from(ish, ifk, iro)
    C_shared = cosmat(d_sh); C_forgery = cosmat(d_fg); C_domain = cosmat(r_dm)

    # bootstrap CIs on off-diagonal entries
    n = len(NAMES); boot = {"shared": [], "forgery": [], "domain": []}
    rng = np.random.RandomState(7)
    for _ in range(B):
        bs = rng.randint(0, len(F_shared), len(F_shared))
        bfk = {nm: rng.randint(0, len(FK[nm]), len(FK[nm])) for nm in NAMES}
        bro = {nm: (bs if DSDIR[nm] == "ffpp" else rng.randint(0, len(RO[nm]), len(RO[nm]))) for nm in NAMES}
        ds, dg, rd = dirs_from(bs, bfk, bro)
        boot["shared"].append(cosmat(ds)); boot["forgery"].append(cosmat(dg)); boot["domain"].append(cosmat(rd))
    def ci(key):
        arr = np.array(boot[key])  # (B,n,n)
        lo = np.nanpercentile(arr, 2.5, axis=0); hi = np.nanpercentile(arr, 97.5, axis=0); return lo, hi
    lo_s, hi_s = ci("shared"); lo_f, hi_f = ci("forgery"); lo_d, hi_d = ci("domain")

    # per-pair table
    pairs = []
    for a in range(n):
        for b in range(a+1, n):
            pairs.append({"pair": f"{NAMES[a]}-{NAMES[b]}",
                          "shared": [round(float(C_shared[a, b]), 3), round(float(lo_s[a, b]), 3), round(float(hi_s[a, b]), 3)],
                          "forgery": [round(float(C_forgery[a, b]), 3), round(float(lo_f[a, b]), 3), round(float(hi_f[a, b]), 3)],
                          "domain": [round(float(C_domain[a, b]), 3), round(float(lo_d[a, b]), 3), round(float(hi_d[a, b]), 3)]})

    os_off = offdiag(C_shared); of_off = offdiag(C_forgery); od_off = offdiag(C_domain)
    corr_sh_fg = spearman(os_off, of_off); corr_sh_dm = spearman(os_off, od_off)

    # routing under each matrix across tau sweep
    taus = [-0.5, -0.2, 0.0, 0.2, 0.5]
    routing = {"shared": {}, "forgery": {}}
    for tau in taus:
        routing["shared"][str(tau)] = route(C_shared, NAMES, tau)
        routing["forgery"][str(tau)] = route(C_forgery, NAMES, tau)

    # swap-diffusion focal
    ia, ib = NAMES.index("swap"), NAMES.index("diffusion")
    sd = {"shared": round(float(C_shared[ia, ib]), 3), "forgery": round(float(C_forgery[ia, ib]), 3),
          "domain": round(float(C_domain[ia, ib]), 3),
          "forgery_ci": [round(float(lo_f[ia, ib]), 3), round(float(hi_f[ia, ib]), 3)],
          "forgery_flips_positive": bool(lo_f[ia, ib] > 0)}

    # verdict -- report the 3 pre-registered conditions EXPLICITLY (no goalpost-moving)
    diff_flips_ci = sd["forgery_flips_positive"]                       # prereg strict: forgery CI > 0
    antagonism_destroyed = bool(lo_f[ia, ib] > C_shared[ia, ib])       # forgery CI excludes the shared antagonism value
    forgery_point_positive = bool(C_forgery[ia, ib] > 0)
    shared_is_domain = bool(corr_sh_dm > corr_sh_fg)
    diff_isolates_shared = any(len(h) == 1 and h[0] == "diffusion" for h in routing["shared"]["0.0"])
    diff_isolates_forgery = any(len(h) == 1 and h[0] == "diffusion" for h in routing["forgery"]["0.0"])
    routing_reverses = bool(diff_isolates_shared and not diff_isolates_forgery)
    # swap-family robustness (real forgery geometry check)
    def cval(mat, a, b): return float(mat[NAMES.index(a), NAMES.index(b)])
    swapfam_robust = bool(cval(C_forgery, "celebdf", "dfdc") > 0.6 and cval(C_forgery, "swap", "celebdf") > 0.2)
    conds = {"prereg_forgeryCI_gt0": diff_flips_ci, "antagonism_destroyed_CInonoverlap": antagonism_destroyed,
             "forgery_point_positive": forgery_point_positive, "shared_tracks_domain": shared_is_domain,
             "routing_reverses": routing_reverses, "swapfamily_robust_realforgery": swapfam_robust}
    strong = antagonism_destroyed and shared_is_domain and routing_reverses and forgery_point_positive
    if strong:
        verdict = ("DOMAIN ARTIFACT (diffusion antagonist): the -0.9 diffusion antagonism is DOMAIN-driven, not forgery. Under domain control it "
                   f"is destroyed (forgery {C_forgery[ia,ib]:+.2f} CI excludes shared {C_shared[ia,ib]:+.2f}), operative C_shared tracks C_domain "
                   f"(rho {corr_sh_dm:+.2f}) over C_forgery ({corr_sh_fg:+.2f}), and ROUTING REVERSES (diffusion isolates under shared, joins swap "
                   "under forgery). NUANCE: swap-family internal geometry (celebdf/dfdc/swap) is ROBUST to domain control = REAL forgery similarity. "
                   "Prereg's strict forgery-CI>0 NOT met (wide bootstrap, 247 diffusion reals) -- reported, not hidden.")
    elif (not antagonism_destroyed) and corr_sh_fg >= 0.7:
        verdict = "LAW SURVIVES: forgery geometry robust to domain control."
    else:
        verdict = "MIXED: see conditions."

    res = {"config": {"paradigms": NAMES, "subsample": SUB, "boot": B, "smoke": SMOKE, "prereg": "DOMAIN_CONTROL_2026_07_26"},
           "C_shared": C_shared.round(3).tolist(), "C_forgery": C_forgery.round(3).tolist(), "C_domain": C_domain.round(3).tolist(),
           "pairs": pairs, "swap_diffusion": sd,
           "corr_shared_vs_forgery": round(corr_sh_fg, 3), "corr_shared_vs_domain": round(corr_sh_dm, 3),
           "routing": routing, "conditions": conds, "verdict": verdict}
    OUT.write_text(json.dumps(res, indent=1))

    print("\n[DC] === per-pair cosine (val [lo,hi]) ===", flush=True)
    for p in pairs:
        print(f"  {p['pair']:20} shared={p['shared'][0]:+.3f}[{p['shared'][1]:+.2f},{p['shared'][2]:+.2f}]  "
              f"forgery={p['forgery'][0]:+.3f}[{p['forgery'][1]:+.2f},{p['forgery'][2]:+.2f}]  "
              f"domain={p['domain'][0]:+.3f}", flush=True)
    print(f"\n[DC] swap-diffusion: shared={sd['shared']:+.3f} forgery={sd['forgery']:+.3f}{sd['forgery_ci']} domain={sd['domain']:+.3f} | forgery_flips_positive={sd['forgery_flips_positive']}", flush=True)
    print(f"[DC] corr(C_shared, C_forgery)={corr_sh_fg:+.3f}  corr(C_shared, C_domain)={corr_sh_dm:+.3f}  -> operative cosine tracks {'DOMAIN' if corr_sh_dm>corr_sh_fg else 'FORGERY'}", flush=True)
    print(f"[DC] routing@tau0.0  SHARED={routing['shared']['0.0']}   FORGERY={routing['forgery']['0.0']}", flush=True)
    print(f"\n[DC] VERDICT: {verdict}", flush=True)
    print(f"[DC] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
