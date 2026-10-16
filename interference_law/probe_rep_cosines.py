"""
probe_rep_cosines.py — Thread #6 crux: does a WEIGHT-UPDATE (Delta-theta) or FISHER-subspace cosine predict representational
forgetting where the FROZEN-FEATURE cosine failed (rho=-0.12)? Subset 8 A-classes x 15 B-tasks. Forgetting pulled from the full run
(same B-tasks). Reuses probe_representational_law machinery.

Three cosines per (A,B):
  FEATURE  = cos(d_A, d_B) in ORIGINAL phi (t=0).
  DTHETA   = cos(Delta-theta_A, Delta-theta_B)  (realized LoRA update dirs; A-binary vs B-5way; task-intrinsic, non-circular vs forget).
  FISHER   = overlap(V_A, V_B) = ||V_A V_B^T||_F^2 / k  vs random-subspace control (per-example LoRA-grad top-k subspaces).
Correlate each with forget -> which recovers the law? $0 local.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R      # machinery (load_banks/build_model/features/prep/ddir/lora_params/finetune_B/spearman)

DEV = R.DEV
RANK = R._i("RANK", 32); B_STEPS = R._i("B_STEPS", 1500); B_LR = R._f("B_LR", 4e-3)
A_STEPS = R._i("A_STEPS", 1200); A_LR = R._f("A_LR", 4e-3)
KFISH = R._i("KFISH", 16); NFISH = R._i("NFISH", 48)
SUB_A = [0, 6, 12, 18, 24, 30, 36, 42]
N_BTASKS = 15
OUT = HERE / "out" / "rep_cosines.json"
torch.manual_seed(0); np.random.seed(0)


def lora_vec(model):
    return torch.cat([p.detach().float().flatten() for p in R.lora_params(model)]).cpu()


def finetune_bin(model, banks, aclass, steps, lr):
    head = nn.Linear(512, 1).to(DEV); params = R.lora_params(model) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    Xp = banks[aclass]; Xn = torch.cat([banks[c] for c in R.NEG_CLASSES])
    X = torch.cat([Xp, Xn]); y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))])
    pw = torch.tensor([float((y == 0).sum()) / max(float(y.sum()), 1)], device=DEV)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(y), (128,)); feat = model(R.prep(X[idx]).to(DEV))
        loss = Fun.binary_cross_entropy_with_logits(head(feat).squeeze(1), y[idx].to(DEV), pos_weight=pw)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); return model, head


def fisher_V(model, head, X, y, kind, k, n):
    lps = R.lora_params(model); rng = np.random.RandomState(0); idx = rng.choice(len(y), min(n, len(y)), replace=False)
    G = []
    for j in idx:
        model.zero_grad(set_to_none=True)
        feat = model(R.prep(X[j:j+1]).to(DEV)); logit = head(feat)
        if kind == "bin":
            loss = Fun.binary_cross_entropy_with_logits(logit.squeeze(), y[j].float().to(DEV))
        else:
            loss = Fun.cross_entropy(logit, y[j:j+1].to(DEV))
        loss.backward()
        G.append(torch.cat([p.grad.detach().flatten() for p in lps]))
    G = torch.stack(G)                                   # n x P (on DEV)
    _, _, Vh = torch.linalg.svd(G, full_matrices=False)  # min(n,P) x P
    return Vh[:k].contiguous().half().cpu()              # k x P


def overlap(VA, VB):
    M = (VA.float() @ VB.float().T)                      # k x k
    return float((M * M).sum() / VA.shape[0])


def main():
    t0 = time.time()
    print(f"[COS] rank={RANK} B={B_STEPS}/{B_LR} A={A_STEPS}/{A_LR} k={KFISH} nF={NFISH} dev={DEV}", flush=True)
    full = json.loads((HERE / "out" / "representational_law_full.json").read_text())
    forget = {(p["A"], p["B"]): p["forget"] for p in full["points"]}

    banks = R.load_banks(); print(f"[COS] banks loaded ({time.time()-t0:.0f}s)", flush=True)
    base = R.build_model(RANK)
    Fneg0 = R.features(base, torch.cat([banks[c] for c in R.NEG_CLASSES]))
    dA = {a: R.ddir(R.features(base, banks[a]), Fneg0) for a in SUB_A}

    # reproduce the SAME 15 B-tasks as the full run (rng seed 1), get dB per B
    rngB = np.random.RandomState(1); Btasks = [list(rngB.choice(R.B_POOL, 5, replace=False)) for _ in range(N_BTASKS)]
    dB = {}
    for bi, bcl in enumerate(Btasks):
        FB0 = np.concatenate([R.features(base, banks[c]) for c in bcl]); dB[bi] = R.ddir(FB0, Fneg0)

    # A-side: Delta-theta_A + Fisher_A (fine-tune each A-binary from a fresh LoRA)
    dthA, VA = {}, {}
    for a in SUB_A:
        m = R.build_model(RANK); init = lora_vec(m)
        m, h = finetune_bin(m, banks, a, A_STEPS, A_LR)
        dthA[a] = (lora_vec(m) - init)
        Xn = torch.cat([banks[c] for c in R.NEG_CLASSES]); X = torch.cat([banks[a], Xn])
        y = torch.cat([torch.ones(len(banks[a])), torch.zeros(len(Xn))])
        VA[a] = fisher_V(m, h, X, y, "bin", KFISH, NFISH)
        print(f"  A{a}: dtheta+Fisher done ({time.time()-t0:.0f}s)", flush=True)

    # B-side: Delta-theta_B + Fisher_B (fine-tune each B-5way from a fresh LoRA)
    dthB, VB = {}, {}
    for bi, bcl in enumerate(Btasks):
        m = R.build_model(RANK); init = lora_vec(m)
        # replicate finetune_B but keep the head for Fisher
        head = nn.Linear(512, len(bcl)).to(DEV); params = R.lora_params(m) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=B_LR, weight_decay=1e-4)
        Xb = torch.cat([banks[c] for c in bcl]); yb = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bcl)])
        m.train()
        for _ in range(B_STEPS):
            idx = torch.randint(0, len(yb), (128,)); loss = Fun.cross_entropy(head(m(R.prep(Xb[idx]).to(DEV))), yb[idx].to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval()
        dthB[bi] = (lora_vec(m) - init); VB[bi] = fisher_V(m, head, Xb, yb, "ce", KFISH, NFISH)
        print(f"  B{bi}: dtheta+Fisher done ({time.time()-t0:.0f}s)", flush=True)

    # random-subspace control for Fisher overlap
    P = VA[SUB_A[0]].shape[1]; rg = torch.Generator().manual_seed(123)
    Rrand = torch.linalg.qr(torch.randn(P, KFISH, generator=rg))[0].T.half()

    rows = []
    for a in SUB_A:
        for bi in range(N_BTASKS):
            if (a, bi) not in forget: continue
            rows.append({"A": a, "B": bi, "forget": forget[(a, bi)],
                         "feature": float(dA[a] @ dB[bi]),
                         "dtheta": float(Fun.cosine_similarity(dthA[a], dthB[bi], dim=0)),
                         "fisher": overlap(VA[a], VB[bi]),
                         "fisher_rand": overlap(Rrand, VB[bi])})
    fg = np.array([r["forget"] for r in rows])
    print(f"\n[COS] n_pairs={len(rows)}  (forget median={np.median(fg):+.3f})", flush=True)
    res = {}
    for key in ("feature", "dtheta", "fisher"):
        x = np.array([r[key] for r in rows]); rho = R.spearman(x, fg); pear = float(np.corrcoef(x, fg)[0, 1])
        res[key] = {"spearman": round(rho, 3), "pearson": round(pear, 3), "min": round(float(x.min()), 3), "max": round(float(x.max()), 3)}
        print(f"  {key:8}: rho(cos,forget)={rho:+.3f}  pearson={pear:+.3f}  range[{x.min():+.2f},{x.max():+.2f}]", flush=True)
    fr = np.array([r["fisher_rand"] for r in rows]); ff = np.array([r["fisher"] for r in rows])
    print(f"  fisher overlap: real mean={ff.mean():.3f} vs random-subspace {fr.mean():.3f} (aimed if real>>rand)", flush=True)
    winner = max(res, key=lambda k: abs(res[k]["spearman"]))
    print(f"\n[COS] STRONGEST predictor = {winner} (|rho|={abs(res[winner]['spearman']):.3f}); feature was -0.12 at full scale", flush=True)

    OUT.write_text(json.dumps({"config": {"rank": RANK, "kfish": KFISH, "sub_a": SUB_A, "n_btasks": N_BTASKS,
                                          "prereg_lock": "1bab16d2"}, "cosines": res,
                               "fisher_vs_random": {"real": float(ff.mean()), "random": float(fr.mean())},
                               "winner": winner, "rows": rows}, indent=1))
    print(f"[COS] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
