"""
probe_unification_wa.py — Thread #7b (PREREG lock 8cffe1c8): re-test the unification projecting onto the DEPLOYED head axis w_A
(logistic weight), not the centroid-diff d_A that #7 mis-locked. Metric of record forget_old is IDENTICAL to #7 (non-saturation
already confirmed) → run full directly. Falsifiable content = the on/off DECOMPOSITION (P2b) + the cosAB→on-axis-sign LOOP (P4b),
NOT the partly-definitional raw on-axis correlation (P1b).
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R      # features/ddir/prep/auc/load_banks/spearman/probe_auc/NEG_CLASSES/B_POOL/_i/_f
import probe_gradient_alignment as G        # build_full/full_ft/theta_vec/train_head/DEV

DEV = G.DEV
FT_STEPS = R._i("FT_STEPS", 500); FT_LR = R._f("FT_LR", 3e-4)
N_B = R._i("N_BTASKS", 15)
SUB_A = [0, 6, 12, 18, 24, 30, 36, 42]
SEED = R._i("SEED", 0)
OUT = HERE / "out" / f"unification_wa{'' if SEED == 0 else f'_s{SEED}'}.json"
LOCK = "8cffe1c8"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_axis(head):
    w = head.weight.detach().cpu().numpy()[0].astype(np.float64); n = np.linalg.norm(w)
    return (w / n).astype(np.float32) if n > 0 else w


def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))


def margin(Fa, Fn, w):
    return float((Fa.mean(0) - Fn.mean(0)) @ w)


def dprime(Fa, Fn, w):
    sa = Fa @ w; sn = Fn @ w; sd = np.sqrt((sa.var() + sn.var()) / 2 + 1e-12)
    return float((sa.mean() - sn.mean()) / sd)


def main():
    t0 = time.time()
    print(f"[UNIF-WA] lock={LOCK} FT={FT_STEPS}/{FT_LR} N_B={N_B} A={len(SUB_A)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg_u8)
    print(f"[UNIF-WA] banks+base ready ({time.time()-t0:.0f}s)", flush=True)

    FA0, dA, wax, oldAUC0, m0, dp0 = {}, {}, {}, {}, {}, {}
    for a in SUB_A:
        FA0[a] = R.features(base, banks[a]); dA[a] = R.ddir(FA0[a], Fneg0)
        ya = [1]*len(FA0[a]) + [0]*len(Fneg0)
        head = G.train_head(np.concatenate([FA0[a], Fneg0]), ya, 1)      # deployed head, frozen from here
        wax[a] = head_axis(head); oldAUC0[a] = head_auc(head, FA0[a], Fneg0)
        m0[a] = margin(FA0[a], Fneg0, wax[a]); dp0[a] = dprime(FA0[a], Fneg0, wax[a])
        FA0[a + 10000] = head                                            # stash frozen head for re-eval
    print(f"[UNIF-WA] A-side: deployed base AUC median={np.median(list(oldAUC0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    rngB = np.random.RandomState(1); Btasks = [list(rngB.choice(R.B_POOL, 5, replace=False)) for _ in range(N_B)]
    theta0 = G.theta_vec(base); points = []
    for bi, bcl in enumerate(Btasks):
        FB0 = np.concatenate([R.features(base, banks[c]) for c in bcl]); dB = R.ddir(FB0, Fneg0)
        Xb = torch.cat([banks[c] for c in bcl]); yb = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bcl)])
        m = G.build_full(); m.load_state_dict(base.state_dict()); G.full_ft(m, Xb, yb, FT_STEPS, FT_LR)
        Fneg1 = R.features(m, Xneg_u8); dtheta = float((G.theta_vec(m) - theta0).norm())
        for a in SUB_A:
            FA1 = R.features(m, banks[a]); w = wax[a]
            forget_old = oldAUC0[a] - head_auc(FA0[a + 10000], FA1, Fneg1)
            onaxis_w = m0[a] - margin(FA1, Fneg1, w)                     # signed margin collapse along deployed axis
            ddprime = dp0[a] - dprime(FA1, Fneg1, w)
            dF = FA1.mean(0) - FA0[a].mean(0)
            on_absA = abs(float(dF @ w)); offaxis_w = float(np.linalg.norm(dF - (dF @ w) * w))
            points.append({"A": a, "B": bi, "forget_old": round(float(forget_old), 4),
                           "onaxis_w": round(onaxis_w, 4), "ddprime": round(ddprime, 4),
                           "on_absA": round(on_absA, 4), "offaxis_w": round(offaxis_w, 4),
                           "dtheta": round(dtheta, 3), "cosAB": round(float(dA[a] @ dB), 4)})
        fo = [p["forget_old"] for p in points if p["B"] == bi]
        print(f"  B{bi}: median forget_old={np.median(fo):+.3f} [{min(fo):+.3f},{max(fo):+.3f}] ({time.time()-t0:.0f}s)", flush=True)

    fo = np.array([p["forget_old"] for p in points]); med = float(np.median(fo))
    print(f"\n[UNIF-WA] n={len(points)} median forget_old={med:+.3f} (band [0.05,0.45]: {'OK' if 0.05<=med<=0.45 else 'OUT'})", flush=True)

    def rho(key, tgt="forget_old", pts=points):
        x = np.array([p[key] for p in pts]); y = np.array([p[tgt] for p in pts])
        return float(R.spearman(x, y)), float(np.corrcoef(x, y)[0, 1])

    res = {}
    for key in ("onaxis_w", "ddprime", "on_absA", "offaxis_w", "dtheta", "cosAB"):
        s, pe = rho(key); res[key] = {"spearman": round(s, 3), "pearson": round(pe, 3)}
        print(f"  rho({key:9}, forget_old) = {s:+.3f}  (pearson {pe:+.3f})", flush=True)
    s_ca_on = float(R.spearman(np.array([p["cosAB"] for p in points]), np.array([p["onaxis_w"] for p in points])))
    res["cosAB_vs_onaxis_w"] = round(s_ca_on, 3)
    print(f"  rho(cosAB, onaxis_w) = {s_ca_on:+.3f}  [P4b loop closure: expect <= -0.30]", flush=True)

    # within-B demeaned (P3b): does on-axis predict WHICH A forgets at fixed B / fixed ||dtheta||?
    Bidx = np.array([p["B"] for p in points]); yfo = fo.astype(float)
    wb = {}
    for key in ("onaxis_w", "on_absA", "offaxis_w"):
        x = np.array([p[key] for p in points]).astype(float).copy(); yd = yfo.copy()
        for b in range(N_B):
            mm = Bidx == b
            if mm.sum() > 1: x[mm] -= x[mm].mean(); yd[mm] -= yfo[mm].mean()
        wb[key] = round(float(R.spearman(x, yd)), 3)
        print(f"  within-B rho({key:9}, forget_old) = {wb[key]:+.3f}", flush=True)
    res["within_B"] = wb

    on = res["onaxis_w"]["spearman"]; off = res["offaxis_w"]["spearman"]; on_abs = res["on_absA"]["spearman"]
    p1b = on >= 0.50
    p2b = (on_abs - off >= 0.30) and (off < 0.30)
    p3b = wb["onaxis_w"] >= 0.40
    p4b = s_ca_on <= -0.30
    if p1b and p2b and p4b:
        verdict = "UNIFICATION RESCUED: forget = signed on-axis (w_A) drift; #7 null was a projection artifact; dichotomy dissolves."
    elif p1b and not p2b:
        verdict = "P1b holds but OFF-AXIS also predicts -> on-axis result TAUTOLOGICAL; unification UNPROVEN, dichotomy stands."
    elif not p1b:
        verdict = "P1b FAILS even along w_A -> dichotomy IRREDUCIBLE: plastic forgetting magnitude-governed regardless of projection."
    else:
        verdict = "PARTIAL: P1b/P2b hold but P4b loop broken -> on-axis is real but NOT the frozen -cos mechanism (analogy, not bridge)."
    print(f"\n[UNIF-WA] P1b={p1b} P2b={p2b} P3b={p3b} P4b={p4b} | VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"config": {"ft_steps": FT_STEPS, "ft_lr": FT_LR, "n_b": N_B, "sub_a": SUB_A, "seed": SEED,
                                          "prereg_lock": LOCK}, "median_forget_old": med, "predictors": res,
                               "checks": {"P1b": p1b, "P2b": p2b, "P3b": p3b, "P4b": p4b}, "verdict": verdict,
                               "points": points}, indent=1))
    print(f"[UNIF-WA] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
