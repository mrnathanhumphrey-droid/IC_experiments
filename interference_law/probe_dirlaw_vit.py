"""
probe_dirlaw_vit.py — NEW-1a'''' ViT arm (PREREG lock 8b4cf4c6, DOCUMENTED speed deviation for vit_b_16 only: GPU-side resize + AMP
autocast + batch 64 + 150 steps, because fp32/batch128/CPU-resize thrashed to ~11s/step). The locked PREDICTION (does align_dc
replicate on a pure transformer) is unchanged. Same decomposition as 1a'''. $0 local. Merges with resnet18+mobilenet (already landed).
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
VICTIMS = list(range(24)); PERTURBERS = [50, 55, 60, 65, 70, 75]; NEG = list(range(90, 100))
FT_STEPS = R._i("FT_STEPS", 150); FT_LR = R._f("FT_LR", 1e-4); BS = R._i("BS", 64); NJ = R._i("NJ", 64); SEED = R._i("SEED", 0)
OUT = HERE / "out" / "dirlaw_vit.json"; LOCK = "8b4cf4c6"
MEANd = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV); STDd = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
torch.manual_seed(SEED); np.random.seed(SEED)


def prep_gpu(u8):                       # resize on GPU (the fix)
    x = u8.to(DEV, non_blocking=True).float() / 255.0
    x = Fun.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - MEANd) / STDd

def build_vit():
    import torchvision.models as tvm
    m = tvm.vit_b_16(weights="IMAGENET1K_V1"); m.heads = nn.Identity(); return m.to(DEV)

@torch.no_grad()
def feats(net, u8, bs=128):
    net.eval(); outs = []
    for i in range(0, len(u8), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(prep_gpu(u8[i:i+bs]))
        outs.append(o.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)

def theta(net): return torch.cat([p.detach().float().flatten() for p in net.parameters()]).cpu()

def train_head1(Fa, Fn, dim, ep=150):
    h = nn.Linear(dim, 1).to(DEV); opt = torch.optim.AdamW(h.parameters(), lr=0.05, weight_decay=1e-4)
    X = torch.from_numpy(np.concatenate([Fa, Fn])).to(DEV); y = torch.tensor([1.]*len(Fa)+[0.]*len(Fn)).to(DEV)
    pw = torch.tensor([len(Fn)/max(len(Fa), 1)], device=DEV)
    for _ in range(ep):
        opt.zero_grad(); Fun.binary_cross_entropy_with_logits(h(X).squeeze(1), y, pos_weight=pw).backward(); opt.step()
    h.eval(); return h

def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy(); sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))

def full_ft(m, dim, Xb, yb, steps, lr):
    head = nn.Linear(dim, int(yb.max())+1).to(DEV)
    opt = torch.optim.AdamW(list(m.parameters())+list(head.parameters()), lr=lr, weight_decay=1e-4); m.train()
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(steps):
        idx = torch.randint(0, len(yb), (BS,))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = Fun.cross_entropy(head(m(prep_gpu(Xb[idx]))), yb[idx].to(DEV))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    m.eval(); return m

def sp(x, y): return float(R.spearman(np.asarray(x, float).ravel(), np.asarray(y, float).ravel()))
def dcenter(X): return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()


def main():
    t0 = time.time()
    print(f"[VIT] lock={LOCK} FT={FT_STEPS}/{FT_LR} BS={BS} victims={len(VICTIMS)} perturbers={len(PERTURBERS)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in NEG])
    base = build_vit(); base.eval(); dim = feats(base, banks[VICTIMS[0]][:2]).shape[1]
    Fneg0 = feats(base, Xneg_u8); heads, auc0 = {}, {}
    for i in VICTIMS:
        FA0 = feats(base, banks[i]); heads[i] = train_head1(FA0, Fneg0, dim); auc0[i] = head_auc(heads[i], FA0, Fneg0)
    print(f"[VIT] dim={dim} heads ready, base AUC med={np.median(list(auc0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    theta0 = theta(base); nV, nP = len(VICTIMS), len(PERTURBERS)
    dthetas, dmag, FORGET = [], np.zeros(nP), np.zeros((nV, nP))
    for pj, p in enumerate(PERTURBERS):
        Xb = torch.cat([banks[p], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[p])), torch.zeros(len(Xneg_u8))]).long()
        mm = copy.deepcopy(base); full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR)
        dth = (theta(mm) - theta0).float(); dthetas.append(dth); dmag[pj] = float(dth.norm())
        Fneg1 = feats(mm, Xneg_u8)
        for ci, i in enumerate(VICTIMS): FORGET[ci, pj] = auc0[i] - head_auc(heads[i], feats(mm, banks[i]), Fneg1)
        del mm; torch.cuda.empty_cache()
        print(f"    perturb {p}: |dth|={dmag[pj]:.2f} medFgt_col={np.median(FORGET[:,pj]):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    gmag = np.zeros(nV); DOT = np.zeros((nV, nP)); COS = np.zeros((nV, nP))
    Xn_g = banks[VICTIMS[0]]  # placeholder; use neg
    for ci, i in enumerate(VICTIMS):
        base.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            m = heads[i](base(prep_gpu(banks[i][:NJ]))).mean() - heads[i](base(prep_gpu(Xneg_u8[:NJ]))).mean()
        m.backward()
        g = torch.cat([pp.grad.detach().flatten() for pp in base.parameters()]).float().cpu(); gmag[ci] = float(g.norm())
        for pj in range(nP): DOT[ci, pj] = float(g @ dthetas[pj]); COS[ci, pj] = DOT[ci, pj]/(gmag[ci]*dmag[pj]+1e-12)
    base.zero_grad(set_to_none=True)

    ALIGN = -COS; GMAGb = np.repeat(gmag[:, None], nP, 1); med = float(np.median(FORGET)); inband = bool(0.03 <= med <= 0.45)
    tbl = {"fo_full": sp(-DOT, FORGET), "align": sp(ALIGN, FORGET), "align_dcenter": sp(dcenter(ALIGN), dcenter(FORGET)),
           "gmag_broadcast": sp(GMAGb, FORGET)}
    p1 = bool(tbl["align"] >= 0.25); p2 = bool(tbl["align_dcenter"] >= 0.15)
    p3 = bool((tbl["align"] > tbl["gmag_broadcast"]) and (tbl["gmag_broadcast"] <= 0.10))
    print(f"\n[VIT] medFgt={med:+.3f} inband={inband} | align={tbl['align']:+.3f} align_dc={tbl['align_dcenter']:+.3f} "
          f"fo={tbl['fo_full']:+.3f} gmag={tbl['gmag_broadcast']:+.3f} cos[{COS.min():+.4f},{COS.max():+.4f}]", flush=True)
    print(f"[VIT] P1(align>=.25)={p1} P2(align_dc>=.15)={p2} P3(magpattern)={p3} | dim={dim}", flush=True)
    OUT.write_text(json.dumps({"config": {"deviation": "AMP+bs64+150steps+gpu-resize for tractability", "ft": [FT_STEPS, FT_LR],
                               "bs": BS, "victims": VICTIMS, "perturbers": PERTURBERS, "seed": SEED, "prereg_lock": LOCK},
                               "dim": dim, "median_forget": round(med, 4), "inband": inband,
                               "cos_range": [round(float(COS.min()), 4), round(float(COS.max()), 4)],
                               "decomp": {k: round(v, 3) for k, v in tbl.items()},
                               "checks": {"P1_align": p1, "P2_interaction": p2, "P3_magpattern": p3}}, indent=1))
    print(f"[VIT] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
