"""
probe_disruptiveness.py — Thread #1: what makes a task DISRUPTIVE? After #6b dethroned cosine, hunt the B-INTRINSIC predictor of
representational forgetting. Candidates: feature drift ||Δφ|| (neutral set), drift rank, param update ||Δθ||, grad norm ||g_B||,
B difficulty. Outcome = disruptiveness(B) = mean re-probe forgetting over A-classes. $0 local GPU.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_gradient_alignment as G       # build_full/full_ft/features(via R)/grad_vec/theta_vec/train_head
import probe_representational_law as R     # load_banks/features/prep/probe_auc/ddir/spearman

DEV = G.DEV
FT_STEPS = R._i("FT_STEPS", 400); FT_LR = R._f("FT_LR", 1e-3)
N_B = R._i("N_BTASKS", 4 if os.environ.get("MODE") == "smoke" else 24)
A_CLS = list(range(0, 12)); NEUTRAL = list(range(25, 41)); B_POOL = list(range(50, 90)); NEG = list(range(90, 100))
OUT = HERE / "out" / ("disruptiveness_smoke.json" if os.environ.get("MODE") == "smoke" else "disruptiveness.json")
torch.manual_seed(0); np.random.seed(0)


def ft_keep(base, Xb, yb, steps, lr):
    m = G.build_full(); m.load_state_dict(base.state_dict())
    head = nn.Linear(512, int(yb.max()) + 1).to(DEV)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    m.train()
    for _ in range(steps):
        idx = torch.randint(0, len(yb), (128,)); loss = Fun.cross_entropy(head(m(R.prep(Xb[idx]).to(DEV))), yb[idx].to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((head(m(R.prep(Xb[:512]).to(DEV))).argmax(1).cpu() == yb[:512]).float().mean())
    return m, acc


def eff_rank(M):  # participation ratio of singular values (broad vs concentrated)
    s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
    s2 = s ** 2; return float((s2.sum() ** 2) / (np.square(s2).sum() + 1e-30))


def main():
    t0 = time.time()
    print(f"[DISRUPT] N_B={N_B} FT={FT_STEPS}/{FT_LR} A={len(A_CLS)} neutral={len(NEUTRAL)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in NEG])[:500]
    Xneu_u8 = torch.cat([banks[c] for c in NEUTRAL])[:400]
    base = G.build_full(); base.eval()
    Fneg0 = R.features(base, Xneg_u8); Fneu0 = R.features(base, Xneu_u8)
    accA0 = {a: R.probe_auc(R.features(base, banks[a]), Fneg0) for a in A_CLS}
    theta0 = G.theta_vec(base)
    print(f"[DISRUPT] base ready, accA0 median={np.median(list(accA0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(1); Btasks = [list(rng.choice(B_POOL, 5, replace=False)) for _ in range(N_B)]
    rows = []
    for bi, bcl in enumerate(Btasks):
        Xb = torch.cat([banks[c] for c in bcl]); yb = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bcl)])
        # grad norm at base (trained B head)
        FB = np.concatenate([R.features(base, banks[c]) for c in bcl]); hB = G.train_head(FB, yb.numpy(), len(bcl))
        gB = G.grad_vec(base, hB, Xb, yb.numpy(), len(bcl)); grad_norm = float(gB.norm())
        # full-FT
        m, b_acc = ft_keep(base, Xb, yb, FT_STEPS, FT_LR)
        dtheta_norm = float((G.theta_vec(m) - theta0).norm())
        # feature drift on the NEUTRAL set (disjoint from A/neg/B)
        Fneu1 = R.features(m, Xneu_u8); dphi = Fneu1 - Fneu0
        drift = float(np.mean(np.linalg.norm(dphi, axis=1) / (np.linalg.norm(Fneu0, axis=1) + 1e-8)))
        drift_rank = eff_rank(dphi)
        # disruptiveness = mean re-probe forgetting over A
        Fneg1 = R.features(m, Xneg_u8)
        forgets = [accA0[a] - R.probe_auc(R.features(m, banks[a]), Fneg1) for a in A_CLS]
        disrupt = float(np.mean(forgets))
        rows.append({"B": bi, "disrupt": round(disrupt, 4), "drift": round(drift, 4), "drift_rank": round(drift_rank, 2),
                     "dtheta": round(dtheta_norm, 3), "grad_norm": round(grad_norm, 4), "b_acc": round(b_acc, 3)})
        print(f"  B{bi}: disrupt={disrupt:+.3f} drift={drift:.3f} rank={drift_rank:.1f} |dth|={dtheta_norm:.2f} "
              f"|g|={grad_norm:.3f} bacc={b_acc:.2f} ({time.time()-t0:.0f}s)", flush=True)

    dis = np.array([r["disrupt"] for r in rows]); med = float(np.median(dis))
    print(f"\n[DISRUPT] n_B={len(rows)} median_disrupt={med:+.3f} (partial band [0.05,0.45]: {'OK' if 0.05<=med<=0.45 else 'OUT'})", flush=True)
    res = {}
    for key in ("drift", "drift_rank", "dtheta", "grad_norm", "b_acc"):
        x = np.array([r[key] for r in rows]); rho = R.spearman(x, dis); pear = float(np.corrcoef(x, dis)[0, 1])
        res[key] = {"spearman": round(rho, 3), "pearson": round(pear, 3)}
        print(f"  {key:11}: rho(pred,disrupt)={rho:+.3f}  pearson={pear:+.3f}", flush=True)
    winner = max(res, key=lambda k: abs(res[k]["spearman"]))
    print(f"\n[DISRUPT] STRONGEST predictor of disruptiveness = {winner} (|rho|={abs(res[winner]['spearman']):.3f})  "
          f"[cosine was ~0 in #6b]", flush=True)
    OUT.write_text(json.dumps({"config": {"n_b": N_B, "ft": [FT_STEPS, FT_LR], "prereg_lock": "bec29a92"},
                               "median_disrupt": med, "predictors": res, "winner": winner, "rows": rows}, indent=1))
    print(f"[DISRUPT] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
