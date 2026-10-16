"""
probe_dynamics.py — Direction #2 (PREREG lock 9508ee75): the DYNAMICS of interference. Checkpoint the deployed-head forgetting curve
forget_A(t), A-discriminability d'_A(t), A-feature-centroid trajectory p_A(t), and B-competence acc_B(t) THROUGHOUT a B fine-tune.
Tests: D2-P1 ballistic-vs-curved (consecutive drift-increment cosine), D2-P2 early-warning (t=50 predicts t=500), D2-P3 static cosine
governs collision RATE (cosAB front-loaded), D2-P4 monotone-vs-rebound. $0 local.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R      # features/ddir/prep/auc/load_banks/spearman/NEG_CLASSES/B_POOL/_i/_f
import probe_gradient_alignment as G        # build_full/theta_vec/train_head/DEV

DEV = G.DEV
FT_LR = R._f("FT_LR", 3e-4)
N_B = R._i("N_BTASKS", 15)
CKPT = [0, 25, 50, 100, 200, 350, 500]
SUB_A = [0, 6, 12, 18, 24, 30, 36, 42]
SEED = R._i("SEED", 0)
OUT = HERE / "out" / f"dynamics{'' if SEED == 0 else f'_s{SEED}'}.json"
LOCK = "9508ee75"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_axis(head):
    w = head.weight.detach().cpu().numpy()[0].astype(np.float64); n = np.linalg.norm(w)
    return (w / n).astype(np.float32) if n > 0 else w

def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))

def dprime(Fa, Fn, w):
    sa = Fa @ w; sn = Fn @ w; sd = np.sqrt((sa.var() + sn.var()) / 2 + 1e-12)
    return float((sa.mean() - sn.mean()) / sd)

def cos(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def main():
    t0 = time.time()
    print(f"[DYN] lock={LOCK} LR={FT_LR} N_B={N_B} CKPT={CKPT} A={len(SUB_A)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg_u8)
    print(f"[DYN] banks+base ready ({time.time()-t0:.0f}s)", flush=True)

    heads, wax, oldA0, dA = {}, {}, {}, {}
    for a in SUB_A:
        FA0 = R.features(base, banks[a]); dA[a] = R.ddir(FA0, Fneg0)
        ya = [1]*len(FA0) + [0]*len(Fneg0)
        heads[a] = G.train_head(np.concatenate([FA0, Fneg0]), ya, 1)
        wax[a] = head_axis(heads[a]); oldA0[a] = head_auc(heads[a], FA0, Fneg0)
    print(f"[DYN] A-side deployed heads ready ({time.time()-t0:.0f}s)", flush=True)

    rngB = np.random.RandomState(1); Btasks = [list(rngB.choice(R.B_POOL, 5, replace=False)) for _ in range(N_B)]
    theta0 = G.theta_vec(base)
    # trajectories: traj[(a,b)] = dict of lists over CKPT
    traj = {}; accB_all = []; cosAB = {}
    for bi, bcl in enumerate(Btasks):
        FB0 = np.concatenate([R.features(base, banks[c]) for c in bcl]); dB = R.ddir(FB0, Fneg0)
        for a in SUB_A: cosAB[(a, bi)] = float(dA[a] @ dB)
        X = torch.cat([banks[c] for c in bcl]); y = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bcl)])
        m = G.build_full(); m.load_state_dict(base.state_dict())
        bhead = nn.Linear(512, len(bcl)).to(DEV)
        opt = torch.optim.AdamW(list(m.parameters()) + list(bhead.parameters()), lr=FT_LR, weight_decay=1e-4)
        for a in SUB_A: traj[(a, bi)] = {"forget": [], "dprime": [], "cent": []}
        accB = []; prev = 0
        for ck in CKPT:
            m.train()
            for _ in range(ck - prev):
                idx = torch.randint(0, len(y), (128,))
                loss = Fun.cross_entropy(bhead(m(R.prep(X[idx]).to(DEV))), y[idx].to(DEV))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            prev = ck
            m.eval()
            Fneg_t = R.features(m, Xneg_u8)
            for a in SUB_A:
                FA_t = R.features(m, banks[a])
                traj[(a, bi)]["forget"].append(oldA0[a] - head_auc(heads[a], FA_t, Fneg_t))
                traj[(a, bi)]["dprime"].append(dprime(FA_t, Fneg_t, wax[a]))
                traj[(a, bi)]["cent"].append(FA_t.mean(0).astype(np.float64))
            with torch.no_grad():
                acc = float((bhead(m(R.prep(X[:512]).to(DEV))).argmax(1).cpu() == y[:512]).float().mean())
            accB.append(round(acc, 3))
        accB_all.append(accB)
        dth = float((G.theta_vec(m) - theta0).norm())
        fo_final = np.median([traj[(a, bi)]["forget"][-1] for a in SUB_A])
        print(f"  B{bi}: accB {accB[0]:.2f}->{accB[-1]:.2f}  med forget_old(500)={fo_final:+.3f}  |dth|={dth:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # ── analysis ──────────────────────────────────────────────────────────────
    pairs = [(a, b) for b in range(N_B) for a in SUB_A]
    i50 = CKPT.index(50); i500 = CKPT.index(500)

    # D2-P1: consecutive drift-increment cosine (curvature). increments v_i = cent[i]-cent[i-1], i=1..6
    early_c, late_c = [], []
    for k in pairs:
        c = traj[k]["cent"]; v = [c[i] - c[i-1] for i in range(1, len(c))]     # v[0]=0->25, v[1]=25->50, v[2]=50->100, v[3]=100->200,...
        ec = [cos(v[i], v[i-1]) for i in (1, 2)]                                # c(v0,v1)@t=50, c(v1,v2)@t=100  → early (t<=100)
        lc = [cos(v[i], v[i-1]) for i in (4, 5)]                                # c(v3,v4)@t=350, c(v4,v5)@t=500 → late
        early_c.append(np.mean(ec)); late_c.append(np.mean(lc))
    mean_early_c = float(np.mean(early_c)); mean_late_c = float(np.mean(late_c))

    # D2-P2: early-warning
    f50 = np.array([traj[k]["forget"][i50] for k in pairs]); f500 = np.array([traj[k]["forget"][i500] for k in pairs])
    rho_ew = float(R.spearman(f50, f500))

    # D2-P3: static cosine governs the RATE (front-loaded?)
    ca = np.array([cosAB[k] for k in pairs])
    rho_ca_50 = float(R.spearman(ca, f50)); rho_ca_500 = float(R.spearman(ca, f500))

    # D2-P4: rebound fraction (dprime recovers >0.1 after its min)
    reb = 0
    for k in pairs:
        d = np.array(traj[k]["dprime"]); mi = int(d.argmin())
        if mi < len(d) - 1 and (d[mi+1:].max() - d[mi]) > 0.1: reb += 1
    reb_frac = reb / len(pairs)

    # forgetting curve (median over pairs at each checkpoint) + accB curve
    curve = [round(float(np.median([traj[k]["forget"][i] for k in pairs])), 4) for i in range(len(CKPT))]
    accB_curve = [round(float(np.median([a[i] for a in accB_all])), 3) for i in range(len(CKPT))]

    print(f"\n[DYN] n_pairs={len(pairs)}  forget curve (t={CKPT}): {curve}", flush=True)
    print(f"[DYN] accB curve: {accB_curve}", flush=True)
    print(f"[DYN] D2-P1 consecutive drift-cos: early(t<=100)={mean_early_c:+.3f}  late={mean_late_c:+.3f}  "
          f"[ballistic if early>=+0.60, curved if <0.40]", flush=True)
    print(f"[DYN] D2-P2 early-warning rho(forget@50, forget@500)={rho_ew:+.3f}  [expect >=+0.60]", flush=True)
    print(f"[DYN] D2-P3 rho(cosAB, forget@50)={rho_ca_50:+.3f}  rho(cosAB, forget@500)={rho_ca_500:+.3f}  "
          f"[front-loaded if 50 no weaker; Delta={rho_ca_50 - rho_ca_500:+.3f}]", flush=True)
    print(f"[DYN] D2-P4 rebound fraction={reb_frac:.2f}  [monotone if <0.15]", flush=True)

    p1 = mean_early_c >= 0.60; p1_curved = mean_early_c < 0.40
    p2 = rho_ew >= 0.60
    p3 = (rho_ca_50 <= 0) and (rho_ca_500 <= 0) and (rho_ca_50 <= rho_ca_500 + 1e-9)
    p3_strong = (rho_ca_50 - rho_ca_500) <= -0.10
    p4_monotone = reb_frac < 0.15
    verdict = []
    verdict.append("D2-P1=BALLISTIC" if p1 else ("D2-P1=CURVED(rotates)" if p1_curved else "D2-P1=MIXED"))
    verdict.append("D2-P2=early-warning-YES" if p2 else "D2-P2=early-warning-NO")
    verdict.append("D2-P3=cosine-front-loaded" + ("(STRONG)" if p3_strong else "") if p3 else "D2-P3=cosine-not-front-loaded")
    verdict.append("D2-P4=monotone" if p4_monotone else f"D2-P4=rebounds({reb_frac:.0%})")
    print(f"\n[DYN] VERDICT: {' | '.join(verdict)}", flush=True)

    OUT.write_text(json.dumps({"config": {"lr": FT_LR, "n_b": N_B, "ckpt": CKPT, "sub_a": SUB_A, "seed": SEED, "prereg_lock": LOCK},
                               "forget_curve": curve, "accB_curve": accB_curve,
                               "D2P1_drift_cos": {"early": round(mean_early_c, 3), "late": round(mean_late_c, 3)},
                               "D2P2_early_warning_rho": round(rho_ew, 3),
                               "D2P3_cosAB_rho": {"t50": round(rho_ca_50, 3), "t500": round(rho_ca_500, 3),
                                                  "delta": round(rho_ca_50 - rho_ca_500, 3)},
                               "D2P4_rebound_frac": round(reb_frac, 3),
                               "checks": {"P1_ballistic": p1, "P1_curved": p1_curved, "P2": p2, "P3": p3, "P3_strong": p3_strong,
                                          "P4_monotone": p4_monotone}, "verdict": " | ".join(verdict),
                               "traj": {f"{a}_{b}": {"forget": [round(float(x), 4) for x in traj[(a, b)]["forget"]],
                                                     "dprime": [round(float(x), 4) for x in traj[(a, b)]["dprime"]]}
                                        for (a, b) in pairs}}, indent=1))
    print(f"[DYN] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
