"""
probe_unification.py — Thread #7: is the frozen/plastic DICHOTOMY one law in two projections? (PREREG lock 18693b8d.)
Plastic backbone (FULL fine-tune), but forgetting measured on the DEPLOYED (frozen) A-head — the metric where DIRECTION of feature
drift decides survival. Test whether on-axis margin collapse (Δθ projected into A's functional tangent space, obtained as A's own
centroid drift along d_A) RE-EMERGES as the governing variable that #6b's raw θ-space cosine missed.

MODE=gate : non-saturation check on the OLD-head metric (3 B-tasks; require median forget_old in [0.05,0.45]).
MODE=full : 8A x 15B curve; P1..P4 correlations. $0 local.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R      # features/ddir/prep/auc/load_banks/spearman/probe_auc/NEG_CLASSES/B_POOL/_i/_f
import probe_gradient_alignment as G        # build_full/full_ft/theta_vec/train_head/DEV

DEV = G.DEV; MODE = os.environ.get("MODE", "gate")
FT_STEPS = R._i("FT_STEPS", 500); FT_LR = R._f("FT_LR", 3e-4)
N_B = R._i("N_BTASKS", 3 if MODE == "gate" else 15)
SUB_A = [0, 6, 12, 18, 24, 30, 36, 42]
SEED = R._i("SEED", 0)
OUT = HERE / "out" / f"unification_{MODE}{'' if SEED == 0 else f'_s{SEED}'}.json"
LOCK = "18693b8d"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_auc(head, Fa, Fn):
    """AUC of a FIXED logistic head (ncls=1) applied to A(1) vs neg(0) features. Head weights never updated."""
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    sc = np.concatenate([sa, sn]); y = np.concatenate([np.ones(len(sa)), np.zeros(len(sn))])
    return R.auc(sc, y)


def main():
    t0 = time.time()
    print(f"[UNIF] lock={LOCK} MODE={MODE} FT={FT_STEPS}/{FT_LR} N_B={N_B} A={len(SUB_A)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])
    base = G.build_full(); base.eval()
    Fneg0 = R.features(base, Xneg_u8)
    print(f"[UNIF] banks+base ready ({time.time()-t0:.0f}s)", flush=True)

    # A-side @ base: features, discriminative dir, FROZEN deployed head, base centroid margin
    FA0, dA, wA, oldAUC0, g0proj = {}, {}, {}, {}, {}
    for a in SUB_A:
        FA0[a] = R.features(base, banks[a]); dA[a] = R.ddir(FA0[a], Fneg0)
        ya = [1]*len(FA0[a]) + [0]*len(Fneg0)
        wA[a] = G.train_head(np.concatenate([FA0[a], Fneg0]), ya, 1)   # frozen from here on
        oldAUC0[a] = head_auc(wA[a], FA0[a], Fneg0)
        g0 = FA0[a].mean(0) - Fneg0.mean(0); g0proj[a] = float(g0 @ dA[a])
    print(f"[UNIF] A-side: old-head base AUC median={np.median(list(oldAUC0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    rngB = np.random.RandomState(1); Btasks = [list(rngB.choice(R.B_POOL, 5, replace=False)) for _ in range(N_B)]
    theta0 = G.theta_vec(base)
    points = []
    for bi, bcl in enumerate(Btasks):
        FB0 = np.concatenate([R.features(base, banks[c]) for c in bcl]); dB = R.ddir(FB0, Fneg0)
        Xb = torch.cat([banks[c] for c in bcl]); yb = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bcl)])
        m = G.build_full(); m.load_state_dict(base.state_dict()); G.full_ft(m, Xb, yb, FT_STEPS, FT_LR)
        Fneg1 = R.features(m, Xneg_u8); dtheta = float((G.theta_vec(m) - theta0).norm())
        for a in SUB_A:
            FA1 = R.features(m, banks[a])
            forget_old = oldAUC0[a] - head_auc(wA[a], FA1, Fneg1)
            forget_fresh = R.probe_auc(FA0[a], Fneg0) - R.probe_auc(FA1, Fneg1)
            g1 = FA1.mean(0) - Fneg1.mean(0)
            onaxis = g0proj[a] - float(g1 @ dA[a])                    # margin COLLAPSE along d_A (positive = shrank)
            dF = FA1.mean(0) - FA0[a].mean(0)
            total = float(np.linalg.norm(dF))
            offaxis = float(np.linalg.norm(dF - (dF @ dA[a]) * dA[a]))
            points.append({"A": a, "B": bi, "forget_old": round(float(forget_old), 4),
                           "forget_fresh": round(float(forget_fresh), 4), "onaxis": round(onaxis, 4),
                           "total": round(total, 4), "offaxis": round(offaxis, 4),
                           "dtheta": round(dtheta, 3), "cosAB": round(float(dA[a] @ dB), 4)})
        fo = [p["forget_old"] for p in points if p["B"] == bi]
        print(f"  B{bi} {bcl}: median forget_old={np.median(fo):+.3f} [{min(fo):+.3f},{max(fo):+.3f}] ({time.time()-t0:.0f}s)", flush=True)

    fo = np.array([p["forget_old"] for p in points]); med = float(np.median(fo))
    inband = 0.05 <= med <= 0.45
    print(f"\n[UNIF] n={len(points)} median forget_old={med:+.3f} (partial band [0.05,0.45]: {'OK' if inband else 'OUT'})", flush=True)

    def rho(key, tgt="forget_old"):
        x = np.array([p[key] for p in points]); y = np.array([p[tgt] for p in points])
        return R.spearman(x, y), float(np.corrcoef(x, y)[0, 1])

    res = {}
    for key in ("onaxis", "offaxis", "total", "dtheta", "cosAB"):
        s, pe = rho(key); res[key] = {"spearman": round(s, 3), "pearson": round(pe, 3)}
        print(f"  rho({key:8}, forget_old) = {s:+.3f}  (pearson {pe:+.3f})", flush=True)
    s_ca_on = R.spearman(np.array([p["cosAB"] for p in points]), np.array([p["onaxis"] for p in points]))
    res["cosAB_vs_onaxis"] = round(float(s_ca_on), 3)
    print(f"  rho(cosAB, onaxis) = {s_ca_on:+.3f}  [P4 loop closure: expect < 0]", flush=True)

    # SECONDARY diagnostic (not the locked P1 metric, which is pooled): within-B demeaned rho isolates the directional question
    # for FIXED B-disruptiveness (fixed ||dtheta||) — does on-axis drift decide WHICH A forgets? (same control as #6b's rho_withinB)
    Bidx = np.array([p["B"] for p in points]); yfo = np.array([p["forget_old"] for p in points])
    wb = {}
    for key in ("onaxis", "offaxis", "total", "cosAB"):
        x = np.array([p[key] for p in points]).astype(float).copy(); yd = yfo.astype(float).copy()
        for b in range(N_B):
            m = Bidx == b
            if m.sum() > 1: x[m] -= x[m].mean(); yd[m] -= yfo[m].mean()
        wb[key] = round(float(R.spearman(x, yd)), 3)
        print(f"  within-B rho({key:8}, forget_old) = {wb[key]:+.3f}", flush=True)
    res["within_B"] = wb

    on = res["onaxis"]["spearman"]; off = abs(res["offaxis"]["spearman"]); dth = res["dtheta"]["spearman"]
    p1 = on >= 0.50; p2 = off < on and off < 0.30; p3 = (dth > 0) and (dth < on); p4 = (res["cosAB"]["spearman"] < 0) and (s_ca_on < 0)
    if p1 and p2:
        verdict = "DICHOTOMY RETIRED: one law — forget = on-axis margin collapse; frozen is the J_A=identity special case."
    elif p1 and not p2:
        verdict = "P1 holds but OFF-AXIS also strong -> metric partly tautological; unification UNPROVEN (report, try margin-normalized)."
    else:
        verdict = "P1 FAILS -> the dichotomy is REAL physics: frozen and plastic forgetting are genuinely different mechanisms."
    print(f"\n[UNIF] P1={p1} P2={p2} P3={p3} P4={p4} | VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"config": {"mode": MODE, "ft_steps": FT_STEPS, "ft_lr": FT_LR, "n_b": N_B, "sub_a": SUB_A,
                                          "seed": SEED, "prereg_lock": LOCK}, "median_forget_old": med, "inband": inband,
                               "predictors": res, "checks": {"P1": p1, "P2": p2, "P3": p3, "P4": p4},
                               "verdict": verdict, "points": points}, indent=1))
    print(f"[UNIF] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
