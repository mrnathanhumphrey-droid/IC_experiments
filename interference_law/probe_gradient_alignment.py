"""
probe_gradient_alignment.py — Thread #6b: is representational forgetting EVER cosine-governed? FULL fine-tune (no LoRA) + the
function-space/NTK predictor cos(g_A, g_B) = t=0 loss-gradient alignment (structured -> real dynamic range, unlike accumulated
Delta-theta which #6 showed is near-orthogonal). vs FEATURE cos and a Delta-theta control. Correlate with full-FT re-probe forgetting.

MODE=gate : non-sat tuning of full-FT strength. MODE=full : the 8A x 15B curve. $0 local first.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R      # load_banks/features/prep/ddir/NEG_CLASSES/B_POOL/spearman/_i/_f

DEV = R.DEV; MODE = os.environ.get("MODE", "gate")
FT_STEPS = R._i("FT_STEPS", 500); FT_LR = R._f("FT_LR", 3e-4)
N_B = R._i("N_BTASKS", 3 if MODE == "gate" else 15)
GRAD_N = R._i("GRAD_N", 256)
SUB_A = [0, 6, 12, 18, 24, 30, 36, 42]
SEED = R._i("SEED", 0)
OUT = HERE / "out" / f"gradient_alignment_{MODE}{'' if SEED==0 else f'_s{SEED}'}.json"
torch.manual_seed(SEED); np.random.seed(SEED)


def build_full():
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity()
    return m.to(DEV)


def train_head(Xfeat, y, ncls, epochs=150):
    h = nn.Linear(512, ncls).to(DEV); opt = torch.optim.AdamW(h.parameters(), lr=0.05, weight_decay=1e-4)
    Xt = torch.from_numpy(Xfeat).to(DEV); yt = torch.tensor(y).to(DEV)
    for _ in range(epochs):
        opt.zero_grad()
        if ncls == 1:
            pw = torch.tensor([float((np.array(y) == 0).sum())/max(float(np.array(y).sum()),1)], device=DEV)
            loss = Fun.binary_cross_entropy_with_logits(h(Xt).squeeze(1), yt.float(), pos_weight=pw)
        else:
            loss = Fun.cross_entropy(h(Xt), yt)
        loss.backward(); opt.step()
    h.eval(); return h


def grad_vec(model, head, X, y, ncls, n=GRAD_N):
    """Mean gradient of the task loss wrt ALL backbone params, at base model, head fixed. Structured functional direction."""
    model.zero_grad(set_to_none=True)
    idx = np.random.RandomState(0).choice(len(y), min(n, len(y)), replace=False)
    feat = model(R.prep(X[idx]).to(DEV)); logit = head(feat)
    yt = torch.tensor([y[i] for i in idx]).to(DEV)
    loss = (Fun.binary_cross_entropy_with_logits(logit.squeeze(1), yt.float()) if ncls == 1
            else Fun.cross_entropy(logit, yt))
    loss.backward()
    g = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]).float().cpu()
    model.zero_grad(set_to_none=True); return g


def theta_vec(model):
    return torch.cat([p.detach().float().flatten() for p in model.parameters()]).cpu()


def full_ft(model, Xb, yb, steps, lr):
    head = nn.Linear(512, int(yb.max()) + 1).to(DEV)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(yb), (128,)); loss = Fun.cross_entropy(head(model(R.prep(Xb[idx]).to(DEV))), yb[idx].to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); return model


def main():
    t0 = time.time()
    print(f"[GRAD] MODE={MODE} FT={FT_STEPS}/{FT_LR} N_B={N_B} grad_n={GRAD_N} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])
    print(f"[GRAD] banks loaded ({time.time()-t0:.0f}s)", flush=True)

    base = build_full(); base.eval()
    Fneg0 = R.features(base, Xneg_u8)
    # A-side: base probe AUC + feature dir + gradient dir
    accA0, dA, gA = {}, {}, {}
    for a in SUB_A:
        FA = R.features(base, banks[a]); accA0[a] = R.probe_auc(FA, Fneg0); dA[a] = R.ddir(FA, Fneg0)
        Xa = torch.cat([banks[a], Xneg_u8]); ya = [1]*len(banks[a]) + [0]*len(Xneg_u8)
        hA = train_head(np.concatenate([FA, Fneg0]), ya, 1)
        gA[a] = grad_vec(base, hA, Xa, ya, 1)
    print(f"[GRAD] A-side done: base accA0 median={np.median(list(accA0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    rngB = np.random.RandomState(1); Btasks = [list(rngB.choice(R.B_POOL, 5, replace=False)) for _ in range(N_B)]
    theta0 = theta_vec(base)
    points = []
    for bi, bcl in enumerate(Btasks):
        # B feature dir + gradient dir at base
        FB = np.concatenate([R.features(base, banks[c]) for c in bcl]); dB = R.ddir(FB, Fneg0)
        Xb = torch.cat([banks[c] for c in bcl]); yb = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bcl)])
        hB = train_head(FB, yb.numpy(), len(bcl)); gB = grad_vec(base, hB, Xb, yb.numpy(), len(bcl))
        # full-FT on B, then re-probe A + accumulated dtheta
        m = build_full(); m.load_state_dict(base.state_dict()); full_ft(m, Xb, yb, FT_STEPS, FT_LR)
        Fneg1 = R.features(m, Xneg_u8); dth = theta_vec(m) - theta0
        for a in SUB_A:
            acc1 = R.probe_auc(R.features(m, banks[a]), Fneg1)
            points.append({"A": a, "B": bi, "forget": round(accA0[a] - acc1, 4),
                           "feature": round(float(dA[a] @ dB), 4),
                           "grad": round(float(Fun.cosine_similarity(gA[a], gB, dim=0)), 4),
                           "dtheta": round(float(Fun.cosine_similarity(gA[a], dth, dim=0)), 4)})
        fs = [p["forget"] for p in points if p["B"] == bi]
        print(f"  B{bi}: median_forget={np.median(fs):+.3f}  grad_cos range so far ({time.time()-t0:.0f}s)", flush=True)

    fg = np.array([p["forget"] for p in points]); med = float(np.median(fg))
    print(f"\n[GRAD] n={len(points)} median_forget={med:+.3f} (band [0.05,0.40]: {'OK' if 0.05<=med<=0.40 else 'OUT'})", flush=True)
    res = {}
    for key in ("grad", "feature", "dtheta"):
        x = np.array([p[key] for p in points]); rho = R.spearman(x, fg)
        # per-B demeaned (within-B) rho: controls for B-disruptiveness
        xd = x.copy(); fd = fg.copy()
        for bi in range(N_B):
            m_ = np.array([p["B"] for p in points]) == bi
            if m_.sum() > 1: xd[m_] -= x[m_].mean(); fd[m_] -= fg[m_].mean()
        rho_wb = R.spearman(xd, fd)
        res[key] = {"rho": round(rho, 3), "rho_withinB": round(rho_wb, 3), "range": [round(float(x.min()),3), round(float(x.max()),3)]}
        print(f"  {key:8}: rho={rho:+.3f}  within-B rho={rho_wb:+.3f}  range[{x.min():+.3f},{x.max():+.3f}]", flush=True)

    gr = res["grad"]["range"]; degenerate = abs(gr[1]-gr[0]) < 0.02
    verdict = ("DEGENERATE grad cosine (no range) -> escalate" if degenerate else
               "LAW REVIVES (function-space)" if abs(res["grad"]["rho_withinB"]) > 0.4 and res["grad"]["rho_withinB"] < 0 else
               "HARD BOUNDARY: representational forgetting NOT cosine-governed even with structured gradient cosine")
    print(f"\n[GRAD] grad-cos has range={not degenerate} | VERDICT: {verdict}", flush=True)
    OUT.write_text(json.dumps({"config": {"mode": MODE, "ft_steps": FT_STEPS, "ft_lr": FT_LR, "n_b": N_B, "sub_a": SUB_A,
                                          "prereg_lock": "5b68b154"}, "median_forget": med, "cosines": res,
                               "verdict": verdict, "points": points}, indent=1))
    print(f"[GRAD] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
