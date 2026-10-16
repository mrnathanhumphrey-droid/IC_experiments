"""
probe_dynamics_2b.py — Direction #2b (PREREG lock 8b4671be): resolve INSIDE the one-shot collision #2 couldn't see. Fine checkpoints
inside [0,25] + a SLOW-LR condition + ‖Δθ‖(t). Tests: 2b-P1 ballistic-vs-curved DURING the burst, 2b-P2 safe-travel (damage/‖Δθ‖ drops
after acquisition), 2b-P3 collision acquisition-locked, 2b-P4 one-shot intrinsic-not-LR-artifact. $0 local.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_gradient_alignment as G

DEV = G.DEV
N_B = R._i("N_BTASKS", 8)
SUB_A = [0, 6, 12, 18, 24, 30, 36, 42]
COND = {"fast": {"lr": 3e-4, "ckpt": [0, 1, 2, 3, 5, 8, 12, 18, 25, 40, 70, 120]},
        "slow": {"lr": 6e-5, "ckpt": [0, 5, 10, 20, 40, 70, 120, 200, 350, 500]}}
SEED = R._i("SEED", 0)
OUT = HERE / "out" / f"dynamics_2b{'' if SEED == 0 else f'_s{SEED}'}.json"
LOCK = "8b4671be"
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

def t90(curve, ckpt):
    c = np.asarray(curve, float); init, fin = c[0], c[-1]
    if abs(fin - init) < 1e-9: return ckpt[-1], len(ckpt) - 1
    tgt = init + 0.9 * (fin - init)
    for i, v in enumerate(c):
        if (v >= tgt) if fin >= init else (v <= tgt): return ckpt[i], i
    return ckpt[-1], len(ckpt) - 1


def run_condition(name, lr, ckpt, banks, base, Fneg0, heads, wax, oldA0, dA, Btasks, theta0):
    t0 = time.time(); Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])
    traj = {}; accB = {}; dth = {}; cosAB = {}
    for bi, bcl in enumerate(Btasks):
        FB0 = np.concatenate([R.features(base, banks[c]) for c in bcl]); dB = R.ddir(FB0, Fneg0)
        for a in SUB_A: cosAB[(a, bi)] = float(dA[a] @ dB)
        X = torch.cat([banks[c] for c in bcl]); y = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bcl)])
        m = G.build_full(); m.load_state_dict(base.state_dict())
        bhead = nn.Linear(512, len(bcl)).to(DEV)
        opt = torch.optim.AdamW(list(m.parameters()) + list(bhead.parameters()), lr=lr, weight_decay=1e-4)
        for a in SUB_A: traj[(a, bi)] = {"forget": [], "cent": []}
        accB[bi] = []; dth[bi] = []; prev = 0
        for ck in ckpt:
            m.train()
            for _ in range(ck - prev):
                idx = torch.randint(0, len(y), (128,))
                loss = Fun.cross_entropy(bhead(m(R.prep(X[idx]).to(DEV))), y[idx].to(DEV))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            prev = ck; m.eval()
            Fneg_t = R.features(m, Xneg_u8)
            for a in SUB_A:
                FA_t = R.features(m, banks[a])
                traj[(a, bi)]["forget"].append(oldA0[a] - head_auc(heads[a], FA_t, Fneg_t))
                traj[(a, bi)]["cent"].append(FA_t.mean(0).astype(np.float64))
            with torch.no_grad():
                accB[bi].append(float((bhead(m(R.prep(X[:512]).to(DEV))).argmax(1).cpu() == y[:512]).float().mean()))
            dth[bi].append(float((G.theta_vec(m) - theta0).norm()))
        print(f"    [{name}] B{bi}: accB {accB[bi][0]:.2f}->{accB[bi][-1]:.2f} "
              f"forget {traj[(0,bi)]['forget'][0]:+.2f}->{np.median([traj[(a,bi)]['forget'][-1] for a in SUB_A]):+.3f} "
              f"|dth| {dth[bi][-1]:.1f} ({time.time()-t0:.0f}s)", flush=True)

    # ── analysis ─────────────────────────────────────────────────────────────
    pairs = [(a, b) for b in range(N_B) for a in SUB_A]
    # per-b median-over-A forget curve
    fb = {b: [float(np.median([traj[(a, b)]["forget"][i] for a in SUB_A])) for i in range(len(ckpt))] for b in range(N_B)}

    # 2b-P1: consecutive centroid drift-cosine over acquisition window (t <= t90_accB)
    p1_vals = []
    for (a, b) in pairs:
        _, i90 = t90(accB[b], ckpt); i90 = max(i90, 3)                        # need >=3 ckpts for >=2 cosines
        c = traj[(a, b)]["cent"][:i90 + 1]; v = [c[i] - c[i-1] for i in range(1, len(c))]
        cc = [cos(v[i], v[i-1]) for i in range(1, len(v))]
        if cc: p1_vals.append(np.mean(cc))
    p1 = float(np.mean(p1_vals))

    # 2b-P2: damage-per-unit-travel, acquisition vs post
    ratios = []
    for b in range(N_B):
        _, i90 = t90(accB[b], ckpt); i90 = max(1, min(i90, len(ckpt) - 2))
        f = fb[b]; d = dth[b]
        acq = (f[i90] - f[0]) / max(d[i90] - d[0], 1e-6)
        post = (f[-1] - f[i90]) / max(d[-1] - d[i90], 1e-6)
        if post > 1e-9: ratios.append(acq / post)
        else: ratios.append(np.inf if acq > 1e-6 else 1.0)
    finite = [r for r in ratios if np.isfinite(r)]
    p2_ratio = float(np.median(finite)) if finite else float("inf")
    p2_infcount = sum(1 for r in ratios if not np.isfinite(r))

    # 2b-P3: t90_forget vs t90_accB (slots)
    slot_gaps = []; t90f_steps = []; t90a_steps = []
    for b in range(N_B):
        sf, iff = t90(fb[b], ckpt); sa, ia = t90(accB[b], ckpt)
        slot_gaps.append(abs(iff - ia)); t90f_steps.append(sf); t90a_steps.append(sa)
    p3_slotgap = float(np.median(slot_gaps))
    med_t90f = float(np.median(t90f_steps)); med_t90a = float(np.median(t90a_steps))

    final_forget = float(np.median([fb[b][-1] for b in range(N_B)]))
    return {"ckpt": ckpt, "lr": lr, "forget_curve": [round(float(np.median([fb[b][i] for b in range(N_B)])), 4) for i in range(len(ckpt))],
            "accB_curve": [round(float(np.median([accB[b][i] for b in range(N_B)])), 3) for i in range(len(ckpt))],
            "dth_curve": [round(float(np.median([dth[b][i] for b in range(N_B)])), 3) for i in range(len(ckpt))],
            "P1_drift_cos_acqwin": round(p1, 3), "P2_damage_ratio": (round(p2_ratio, 2) if np.isfinite(p2_ratio) else "inf"),
            "P2_inf_count": p2_infcount, "P3_t90_slotgap": round(p3_slotgap, 2),
            "med_t90_forget": med_t90f, "med_t90_accB": med_t90a, "final_forget": round(final_forget, 4)}


def main():
    t0 = time.time()
    print(f"[DYN2B] lock={LOCK} N_B={N_B} A={len(SUB_A)} conds={list(COND)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg_u8)
    heads, wax, oldA0, dA = {}, {}, {}, {}
    for a in SUB_A:
        FA0 = R.features(base, banks[a]); dA[a] = R.ddir(FA0, Fneg0)
        ya = [1]*len(FA0) + [0]*len(Fneg0)
        heads[a] = G.train_head(np.concatenate([FA0, Fneg0]), ya, 1); wax[a] = head_axis(heads[a]); oldA0[a] = head_auc(heads[a], FA0, Fneg0)
    theta0 = G.theta_vec(base)
    rngB = np.random.RandomState(1); Btasks = [list(rngB.choice(R.B_POOL, 5, replace=False)) for _ in range(N_B)]
    print(f"[DYN2B] setup ready ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for name, cfg in COND.items():
        print(f"[DYN2B] === condition {name} (lr={cfg['lr']}) ===", flush=True)
        results[name] = run_condition(name, cfg["lr"], cfg["ckpt"], banks, base, Fneg0, heads, wax, oldA0, dA, Btasks, theta0)

    F = results["fast"]; S = results["slow"]
    print(f"\n[DYN2B] FAST forget {F['forget_curve']}\n         accB   {F['accB_curve']}\n         |dth|  {F['dth_curve']}", flush=True)
    print(f"[DYN2B] SLOW forget {S['forget_curve']}\n         accB   {S['accB_curve']}\n         |dth|  {S['dth_curve']}", flush=True)

    p1 = F["P1_drift_cos_acqwin"]; p1_ball = p1 >= 0.50; p1_curved = p1 < 0.30
    p2 = F["P2_damage_ratio"]; p2_ok = (p2 == "inf") or (isinstance(p2, (int, float)) and p2 >= 3.0)
    p3_ok = (F["P3_t90_slotgap"] <= 2) and (S["med_t90_accB"] > F["med_t90_accB"])
    slow_meas = S["final_forget"] >= 0.03
    p4_ok = slow_meas and (S["med_t90_forget"] >= 2 * max(F["med_t90_forget"], 1e-9)) and (abs(S["final_forget"] - F["final_forget"]) <= 0.05 or S["final_forget"] < F["final_forget"])

    print(f"\n[DYN2B] 2b-P1 ballistic: FAST acq-window drift-cos={p1:+.3f}  [ballistic>=+.50 {p1_ball} | curved<.30 {p1_curved}]", flush=True)
    print(f"[DYN2B] 2b-P2 safe-travel: damage/travel acq:post ratio={p2} (inf_count={F['P2_inf_count']}/{N_B})  [>=3x {p2_ok}]", flush=True)
    print(f"[DYN2B] 2b-P3 acq-locked: FAST t90_forget-vs-accB slotgap={F['P3_t90_slotgap']}; t90_accB FAST {F['med_t90_accB']} -> SLOW {S['med_t90_accB']}  [{p3_ok}]", flush=True)
    print(f"[DYN2B] 2b-P4 intrinsic: final_forget FAST {F['final_forget']} vs SLOW {S['final_forget']} (meas>={slow_meas}); t90_forget FAST {F['med_t90_forget']} -> SLOW {S['med_t90_forget']}  [{p4_ok}]", flush=True)

    verdict = (f"P1={'BALLISTIC' if p1_ball else ('CURVED' if p1_curved else 'MIXED')} | "
               f"P2={'SAFE-TRAVEL' if p2_ok else 'SATURATION'} | P3={'ACQ-LOCKED' if p3_ok else 'not-locked'} | "
               f"P4={'INTRINSIC' if p4_ok else 'LR-artifact/inconclusive'}")
    print(f"\n[DYN2B] VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"config": {"n_b": N_B, "sub_a": SUB_A, "cond": COND, "seed": SEED, "prereg_lock": LOCK},
                               "results": results, "checks": {"P1_ballistic": p1_ball, "P1_curved": p1_curved, "P2_safe": p2_ok,
                               "P3_acqlocked": p3_ok, "P4_intrinsic": p4_ok, "slow_measurable": slow_meas}, "verdict": verdict}, indent=1))
    print(f"[DYN2B] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
