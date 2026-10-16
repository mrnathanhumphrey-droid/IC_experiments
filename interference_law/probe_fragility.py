"""
probe_fragility.py — NEW-1a (PREREG lock a36cf7eb): what IS fragility? Regress fragile_i (mean forgetting of capability i over a fixed
perturber set) on BASE-MEASURABLE structural predictors. Lead hypothesis: fragility = ISOTROPIC SENSITIVITY (sens_i, random-perturbation
margin sensitivity). $0 local.
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
R_RAND = R._i("R_RAND", 6); FRAC = R._f("FRAC", 0.01); SEED = R._i("SEED", 0)
OUT = HERE / "out" / "fragility.json"; LOCK = "a36cf7eb"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))

def margin(head, Fa, Fn):
    with torch.no_grad():
        return float(head(torch.from_numpy(Fa).to(DEV)).mean() - head(torch.from_numpy(Fn).to(DEV)).mean())

def spearman(x, y): return R.spearman(np.asarray(x, float), np.asarray(y, float))


def main():
    t0 = time.time()
    print(f"[FRAG] lock={LOCK} FT={FT_STEPS}/{FT_LR} victims={len(VICTIMS)} perturbers={len(PERTURBERS)} R={R_RAND} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in NEG])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg_u8)

    # base victim structure
    FA0, dvec, heads, wax, auc0 = {}, {}, {}, {}, {}
    for i in VICTIMS:
        FA0[i] = R.features(base, banks[i]); dvec[i] = R.ddir(FA0[i], Fneg0)
        yi = [1]*len(FA0[i]) + [0]*len(Fneg0)
        heads[i] = G.train_head(np.concatenate([FA0[i], Fneg0]), yi, 1)
        w = heads[i].weight.detach().cpu().numpy()[0].astype(np.float64); wax[i] = (w/np.linalg.norm(w)).astype(np.float32)
        auc0[i] = head_auc(heads[i], FA0[i], Fneg0)
    print(f"[FRAG] 40 victim heads ready, base AUC median={np.median(list(auc0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # structural predictors
    dprime, spread, munorm, wnorm = {}, {}, {}, {}
    for i in VICTIMS:
        sa = FA0[i] @ wax[i]; sn = Fneg0 @ wax[i]
        dprime[i] = float((sa.mean() - sn.mean()) / np.sqrt((sa.var() + sn.var())/2 + 1e-12))
        spread[i] = float(FA0[i].std(0).mean())
        munorm[i] = float(np.linalg.norm(FA0[i].mean(0)))
        wnorm[i] = float(np.linalg.norm(heads[i].weight.detach().cpu().numpy()))
    iso = {i: float(np.mean([dvec[i] @ dvec[k] for k in VICTIMS if k != i])) for i in VICTIMS}

    # sens_i: random-perturbation margin sensitivity
    theta0n = float(G.theta_vec(base).norm()); eps = FRAC * theta0n
    m0 = {i: margin(heads[i], FA0[i], Fneg0) for i in VICTIMS}
    dmarg = {i: 0.0 for i in VICTIMS}
    for r in range(R_RAND):
        m = G.build_full(); m.load_state_dict(base.state_dict())
        gs = [torch.randn_like(p) for p in m.parameters()]
        gn = float(torch.sqrt(sum((g*g).sum() for g in gs)))
        with torch.no_grad():
            for p, g in zip(m.parameters(), gs): p.add_(g * (eps/gn))
        m.eval(); Fneg_r = R.features(m, Xneg_u8)
        for i in VICTIMS: dmarg[i] += abs(margin(heads[i], R.features(m, banks[i]), Fneg_r) - m0[i])
    sens = {i: dmarg[i]/(R_RAND*eps) for i in VICTIMS}
    mean_rel = float(np.mean([dmarg[i]/(R_RAND*abs(m0[i])+1e-9) for i in VICTIMS]))
    print(f"[FRAG] sens done: ||theta0||={theta0n:.1f} eps={eps:.2f} mean|dmargin|/margin={mean_rel:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # fragility = mean forgetting over perturbers
    theta0 = G.theta_vec(base); fmat = {i: [] for i in VICTIMS}
    for p in PERTURBERS:
        Xb = torch.cat([banks[p], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[p])), torch.zeros(len(Xneg_u8))]).long()
        m = G.build_full(); m.load_state_dict(base.state_dict()); G.full_ft(m, Xb, yb, FT_STEPS, FT_LR)
        Fneg1 = R.features(m, Xneg_u8)
        for i in VICTIMS: fmat[i].append(auc0[i] - head_auc(heads[i], R.features(m, banks[i]), Fneg1))
        print(f"  perturb {p}: mean fragility so far={np.mean([np.mean(fmat[i]) for i in VICTIMS]):+.3f} ({time.time()-t0:.0f}s)", flush=True)
    fragile = {i: float(np.mean(fmat[i])) for i in VICTIMS}
    fvals = np.array([fragile[i] for i in VICTIMS]); med = float(np.median(fvals))
    print(f"\n[FRAG] median fragility={med:+.3f} (partial band [0.03,0.45]: {'OK' if 0.03<=med<=0.45 else 'OUT'})", flush=True)

    # univariate + multivariate
    preds = {"dprime": dprime, "spread": spread, "iso": iso, "munorm": munorm, "wnorm": wnorm, "sens": sens}
    uni = {}
    for name, d in preds.items():
        x = np.array([d[i] for i in VICTIMS]); uni[name] = round(spearman(x, fvals), 3)
    for name in sorted(uni, key=lambda k: -abs(uni[k])):
        print(f"  rho({name:7}, fragility) = {uni[name]:+.3f}", flush=True)

    X = np.stack([np.array([preds[n][i] for i in VICTIMS]) for n in preds], 1).astype(float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9); y = fvals
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(y)), Xz], y, rcond=None)
    r2_in = float(1 - ((y - np.c_[np.ones(len(y)), Xz]@beta)**2).sum() / (((y-y.mean())**2).sum()+1e-12))
    loo = []
    for k in range(len(y)):
        tr = [j for j in range(len(y)) if j != k]
        b, *_ = np.linalg.lstsq(np.c_[np.ones(len(tr)), Xz[tr]], y[tr], rcond=None)
        loo.append(np.r_[1, Xz[k]] @ b)
    loo = np.array(loo); loo_r2 = float(1 - ((y-loo)**2).sum() / (((y-y.mean())**2).sum()+1e-12))
    coef = {n: round(float(beta[1+j]), 4) for j, n in enumerate(preds)}
    print(f"\n[FRAG] multivariate: in-sample R²={r2_in:.3f}  LOO-CV R²={loo_r2:.3f}", flush=True)
    print(f"[FRAG] standardized coefs: " + ", ".join(f"{n}:{coef[n]:+.3f}" for n in sorted(coef, key=lambda k: -abs(coef[k]))), flush=True)

    top = max(uni, key=lambda k: abs(uni[k]))
    p1 = loo_r2 >= 0.40
    p2 = (uni["sens"] >= 0.50) and (top == "sens")
    p3 = uni["dprime"] <= -0.30
    p4 = uni["iso"] >= 0.30
    if p1 and p2: verdict = "FRAGILITY = ISOTROPIC SENSITIVITY (predictable + sens-driven): mechanistic + cheap estimator."
    elif p1 and not p2: verdict = f"FRAGILITY STRUCTURAL (predictable, top={top} not sens): separation/crowding theory, not generic sensitivity."
    else: verdict = "FRAGILITY NOT structurally predictable (LOO-R²<0.40) -> lives in victim×perturber interaction, not the capability."
    print(f"\n[FRAG] P1={p1} P2={p2} P3={p3} P4={p4} | top={top} | VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"config": {"ft": [FT_STEPS, FT_LR], "victims": VICTIMS, "perturbers": PERTURBERS, "R_rand": R_RAND,
                               "frac": FRAC, "eps": eps, "mean_rel_dmargin": mean_rel, "seed": SEED, "prereg_lock": LOCK},
                               "median_fragility": med, "univariate_rho": uni, "multivariate": {"r2_in": round(r2_in, 3),
                               "loo_r2": round(loo_r2, 3), "coefs": coef}, "top_predictor": top,
                               "checks": {"P1_predictable": p1, "P2_sensitivity": p2, "P3_margin": p3, "P4_crowding": p4},
                               "verdict": verdict, "fragile": {str(i): round(fragile[i], 4) for i in VICTIMS},
                               "predictors": {n: {str(i): round(float(preds[n][i]), 4) for i in VICTIMS} for n in preds}}, indent=1))
    print(f"[FRAG] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
