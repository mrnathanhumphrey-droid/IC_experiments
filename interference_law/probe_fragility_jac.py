"""
probe_fragility_jac.py — NEW-1a' (PREREG lock 6f5c2aa5): fragility = SENSITIVITY? The real instrument = Jacobian ‖d margin_i/d theta‖
(intrinsic, base-measured, non-circular). + first-order model firstorder(i,p) = -(g_i . dtheta_p) of the directed matrix. $0 local.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_gradient_alignment as G

DEV = G.DEV
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4)
VICTIMS = list(range(40)); PERTURBERS = [50, 55, 60, 65, 70, 75, 80, 85]; NEG = list(range(90, 100))
NJ = R._i("NJ", 128); SEED = R._i("SEED", 0)
OUT = HERE / "out" / "fragility_jac.json"; LOCK = "6f5c2aa5"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))

def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)

def partial_spearman(x, y, z):
    rx, ry, rz = rank(x), rank(y), rank(z); Z = np.c_[np.ones(len(rz)), rz]
    def res(a): c, *_ = np.linalg.lstsq(Z, a, rcond=None); return a - Z @ c
    ex, ey = res(rx), res(ry); return float(np.corrcoef(ex, ey)[0, 1])


def main():
    t0 = time.time()
    print(f"[JAC] lock={LOCK} FT={FT_STEPS}/{FT_LR} victims={len(VICTIMS)} perturbers={len(PERTURBERS)} NJ={NJ} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in NEG])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg_u8)

    FA0, heads, wax, auc0, dprime = {}, {}, {}, {}, {}
    for i in VICTIMS:
        FA0[i] = R.features(base, banks[i])
        yi = [1]*len(FA0[i]) + [0]*len(Fneg0)
        heads[i] = G.train_head(np.concatenate([FA0[i], Fneg0]), yi, 1)
        w = heads[i].weight.detach().cpu().numpy()[0].astype(np.float64); wax[i] = (w/np.linalg.norm(w)).astype(np.float32)
        auc0[i] = head_auc(heads[i], FA0[i], Fneg0)
        sa = FA0[i] @ wax[i]; sn = Fneg0 @ wax[i]
        dprime[i] = float((sa.mean()-sn.mean())/np.sqrt((sa.var()+sn.var())/2+1e-12))
    print(f"[JAC] 40 heads + d' ready ({time.time()-t0:.0f}s)", flush=True)

    # Jacobian g_i = d margin_i / d theta  (autograd through backbone, head fixed)
    Xn_g = R.prep(Xneg_u8[:NJ]).to(DEV)
    sens_jac = {}; gvecs = {}
    for i in VICTIMS:
        base.zero_grad(set_to_none=True)
        fi = base(R.prep(banks[i][:NJ]).to(DEV)); fn = base(Xn_g)
        m = heads[i](fi).mean() - heads[i](fn).mean()
        m.backward()
        g = torch.cat([p.grad.detach().flatten() for p in base.parameters()]).float().cpu()
        sens_jac[i] = float(g.norm()); gvecs[i] = g
    base.zero_grad(set_to_none=True)
    sj = np.array([sens_jac[i] for i in VICTIMS])
    print(f"[JAC] Jacobians done: ||g|| range [{sj.min():.3g}, {sj.max():.3g}] max/min={sj.max()/max(sj.min(),1e-9):.1f} ({time.time()-t0:.0f}s)", flush=True)

    # perturbers: fragility + dtheta_p + first-order predictions
    theta0 = G.theta_vec(base); fmat = {i: [] for i in VICTIMS}; fo = {i: [] for i in VICTIMS}
    for p in PERTURBERS:
        Xb = torch.cat([banks[p], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[p])), torch.zeros(len(Xneg_u8))]).long()
        mm = G.build_full(); mm.load_state_dict(base.state_dict()); G.full_ft(mm, Xb, yb, FT_STEPS, FT_LR)
        dtheta_p = (G.theta_vec(mm) - theta0).float()                       # CPU vector
        Fneg1 = R.features(mm, Xneg_u8)
        for i in VICTIMS:
            fmat[i].append(auc0[i] - head_auc(heads[i], R.features(mm, banks[i]), Fneg1))
            fo[i].append(-float(gvecs[i] @ dtheta_p))                        # first-order predicted margin DROP
        print(f"  perturb {p}: |dth|={float(dtheta_p.norm()):.2f}  mean frag so far={np.mean([np.mean(fmat[i]) for i in VICTIMS]):+.3f} ({time.time()-t0:.0f}s)", flush=True)
    fragile = {i: float(np.mean(fmat[i])) for i in VICTIMS}
    fvals = np.array([fragile[i] for i in VICTIMS]); med = float(np.median(fvals))
    print(f"\n[JAC] median fragility={med:+.3f} (band [0.03,0.45]: {'OK' if 0.03<=med<=0.45 else 'OUT'})", flush=True)

    dpv = np.array([dprime[i] for i in VICTIMS])
    rho_sj = float(R.spearman(sj, fvals)); rho_dp = float(R.spearman(dpv, fvals))
    rho_sj_dp = float(R.spearman(sj, dpv)); prho = partial_spearman(sj, fvals, dpv)
    # first-order model over all 320 (i,p) pairs
    fo_flat = np.array([fo[i][k] for i in VICTIMS for k in range(len(PERTURBERS))])
    ft_flat = np.array([fmat[i][k] for i in VICTIMS for k in range(len(PERTURBERS))])
    rho_fo = float(R.spearman(fo_flat, ft_flat)); pear_fo = float(np.corrcoef(fo_flat, ft_flat)[0, 1])

    print(f"[JAC] 1a'-P1 sensitivity: rho(sens_jac, fragility)={rho_sj:+.3f}  [expect >=+0.50]", flush=True)
    print(f"[JAC] 1a'-P2 instrument fixed: |rho|={abs(rho_sj):.3f} vs 1a dead sens 0.13; range max/min={sj.max()/max(sj.min(),1e-9):.1f}", flush=True)
    print(f"[JAC] 1a'-P3 first-order model: rho(-(g.dtheta), forget) over {len(fo_flat)} pairs={rho_fo:+.3f} (pear {pear_fo:+.3f})  [expect >=+0.40]", flush=True)
    print(f"[JAC] 1a'-P4 sens vs margin: rho(sens_jac,d')={rho_sj_dp:+.3f}  partial rho(sens_jac,frag | d')={prho:+.3f}  (d' alone {rho_dp:+.3f})", flush=True)

    p1 = bool(rho_sj >= 0.50); p2 = bool((abs(rho_sj) - 0.13 >= 0.20) and (sj.max()/max(sj.min(), 1e-9) >= 3))
    p3 = bool(rho_fo >= 0.40); p4_distinct = bool(prho >= 0.30)
    if p1 and p4_distinct: verdict = "FRAGILITY = margin-Jacobian SENSITIVITY (distinct from raw margin) — mechanistic + base-measurable."
    elif p1 and not p4_distinct: verdict = "FRAGILITY = SENSITIVITY = MARGIN (one mechanism, two views: separated => insensitive)."
    else: verdict = "FRAGILITY is NOT first-order Jacobian sensitivity -> it is SEPARATION/structure (1a's margin) and/or higher-order."
    print(f"\n[JAC] P1={p1} P2={p2} P3={p3} P4-distinct={p4_distinct} | VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"config": {"ft": [FT_STEPS, FT_LR], "victims": VICTIMS, "perturbers": PERTURBERS, "nj": NJ,
                               "seed": SEED, "prereg_lock": LOCK}, "median_fragility": med,
                               "P1_rho_sens_frag": round(rho_sj, 3), "P2_range_maxmin": round(float(sj.max()/max(sj.min(), 1e-9)), 2),
                               "P3_firstorder_rho": round(rho_fo, 3), "P3_firstorder_pear": round(pear_fo, 3),
                               "P4_rho_sens_dprime": round(rho_sj_dp, 3), "P4_partial_sens_frag_given_dp": round(prho, 3),
                               "rho_dprime_frag": round(rho_dp, 3),
                               "checks": {"P1": p1, "P2": p2, "P3": p3, "P4_distinct": p4_distinct}, "verdict": verdict,
                               "sens_jac": {str(i): round(sens_jac[i], 4) for i in VICTIMS},
                               "fragile": {str(i): round(fragile[i], 4) for i in VICTIMS},
                               "dprime": {str(i): round(dprime[i], 4) for i in VICTIMS}}, indent=1))
    print(f"[JAC] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
