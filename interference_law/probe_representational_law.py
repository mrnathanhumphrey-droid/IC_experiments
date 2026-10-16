"""
probe_representational_law.py — Thread #6: does the cosine govern forgetting when the BACKBONE MOVES (LoRA fine-tune), not just a
frozen-feature readout? CIFAR-100 + ResNet18, conv-LoRA on layer3+layer4. Frozen-re-probe forgetting.

Efficient design: fine-tune LoRA on ONE B-task, then re-probe ALL held-in A classes -> ~50 (cos, forget) points per fine-tune.
MODE=gate : the make-or-break non-saturation feasibility check (a few B-tasks; is median forget in [0.05,0.40]?).
MODE=full : the calibrated curve (feature cosine over all B-tasks; Delta-theta + Fisher on a subset added after the gate passes).

$0 local first. Reuses #1's probe/auc machinery + a conv-LoRA wrapper.
"""
from __future__ import annotations
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
MODE = os.environ.get("MODE", "gate")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def _i(k, d): return int(os.environ.get(k, d))
def _f(k, d): return float(os.environ.get(k, d))
PER_CLASS = _i("PER_CLASS", 100)
RESIZE = _i("RESIZE", 128)
RANK = _i("RANK", 8)
B_STEPS = _i("B_STEPS", 200)
B_LR = _f("B_LR", 5e-4)
N_BTASKS = _i("N_BTASKS", 3 if MODE == "gate" else 15)
BCLS = 5                         # classes per B-task (multi-class interference)
A_CLASSES = list(range(50))      # held-in "A" capabilities (same as #1)
NEG_CLASSES = list(range(90, 100))
B_POOL = list(range(50, 90))     # reserved pool to sample B-tasks from
OUT = HERE / "out" / f"representational_law_{MODE}.json"
torch.manual_seed(0); np.random.seed(0)


# ── data: CIFAR-100 image banks (uint8), ImageNet-normalized on the fly ────────
def load_banks():
    from datasets import load_dataset
    import torchvision.transforms.functional as TF
    ds = load_dataset("uoft-cs/cifar100", split="train")
    fine = np.array(ds["fine_label"]); imgs = ds["img"]
    rng = np.random.RandomState(0)
    banks = {}
    need = set(A_CLASSES) | set(NEG_CLASSES) | set(B_POOL)
    for c in need:
        idx = np.where(fine == c)[0]; rng.shuffle(idx); idx = idx[:PER_CLASS]
        t = torch.stack([TF.pil_to_tensor(imgs[int(j)].convert("RGB")) for j in idx])  # uint8 [n,3,32,32]
        banks[c] = t
    return banks


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(u8):  # uint8 [n,3,32,32] -> normalized float [n,3,RESIZE,RESIZE]
    x = u8.float() / 255.0
    x = Fun.interpolate(x, size=(RESIZE, RESIZE), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


# ── conv-LoRA on ResNet18 layer3+layer4 ───────────────────────────────────────
class LoRAConv2d(nn.Module):
    def __init__(s, base: nn.Conv2d, r):
        super().__init__(); s.base = base
        for p in s.base.parameters(): p.requires_grad_(False)
        cin, cout = base.in_channels, base.out_channels
        s.A = nn.Conv2d(cin, r, base.kernel_size, base.stride, base.padding, bias=False)
        s.B = nn.Conv2d(r, cout, 1, bias=False)
        nn.init.normal_(s.A.weight, std=0.02); nn.init.zeros_(s.B.weight)   # init update = 0
    def forward(s, x): return s.base(x) + s.B(s.A(x))


def inject_lora(model, r):
    for lname in ("layer3", "layer4"):
        layer = getattr(model, lname)
        for blk in layer:
            for attr in ("conv1", "conv2"):
                conv = getattr(blk, attr)
                if isinstance(conv, nn.Conv2d): setattr(blk, attr, LoRAConv2d(conv, r))
    return model


def build_model(r):
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters(): p.requires_grad_(False)
    inject_lora(m, r)
    m.fc = nn.Identity()
    return m.to(DEV).eval()


def lora_params(model):
    return [p for n, p in model.named_parameters() if (".A." in n or ".B." in n) and p.requires_grad]


@torch.no_grad()
def features(model, u8bank, bs=256):
    outs = []
    for i in range(0, len(u8bank), bs):
        outs.append(model(prep(u8bank[i:i+bs]).to(DEV)).cpu())
    return torch.cat(outs).numpy().astype(np.float32)   # [n,512]


# ── linear probe (A-vs-neg AUC) ───────────────────────────────────────────────
def auc(sc, y):
    y = np.asarray(y); pos = sc[y == 1]; neg = sc[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(a)+1)
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def probe_auc(Xa, Xn, split=0.3):
    # deterministic split, train logistic head A(1) vs neg(0), return test AUC
    def sp(n, tag):
        te = np.array([(int(hashlib.md5(f"{tag}{i}".encode()).hexdigest(),16)%1000) < split*1000 for i in range(n)])
        return ~te, te
    atr, ate = sp(len(Xa), "a"); ntr, nte = sp(len(Xn), "n")
    Xtr = np.concatenate([Xa[atr], Xn[ntr]]); ytr = np.concatenate([np.ones(atr.sum()), np.zeros(ntr.sum())]).astype(np.float32)
    Xte = np.concatenate([Xa[ate], Xn[nte]]); yte = np.concatenate([np.ones(ate.sum()), np.zeros(nte.sum())]).astype(np.float32)
    h = nn.Linear(Xtr.shape[1], 1); opt = torch.optim.AdamW(h.parameters(), lr=0.05, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    pw = torch.tensor([float((ytr==0).sum())/max(float(ytr.sum()),1)]); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(150):
        opt.zero_grad(); lf(h(Xt).squeeze(1), yt).backward(); opt.step()
    with torch.no_grad(): sc = torch.sigmoid(h(torch.from_numpy(Xte)).squeeze(1)).numpy()
    return auc(sc, yte)


def ddir(Xp, Xn):
    d = Xp.mean(0) - Xn.mean(0); n = np.linalg.norm(d); return (d/n).astype(np.float32) if n > 0 else d


def finetune_B(model, Bbanks, bclasses, steps, lr):
    # 5-way classification on the B-task classes; only LoRA params train
    head = nn.Linear(512, len(bclasses)).to(DEV)
    params = lora_params(model) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    X = torch.cat([Bbanks[c] for c in bclasses]); y = torch.cat([torch.full((len(Bbanks[c]),), i) for i, c in enumerate(bclasses)])
    model.train()
    for st in range(steps):
        idx = torch.randint(0, len(y), (128,)); feat = model(prep(X[idx]).to(DEV)); logits = head(feat)
        loss = Fun.cross_entropy(logits, y[idx].to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); return model


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean(); return float((rx@ry)/(np.sqrt(rx@rx)*np.sqrt(ry@ry)+1e-30))


def main():
    t0 = time.time()
    print(f"[REP] MODE={MODE} dev={DEV} rank={RANK} B_steps={B_STEPS} B_lr={B_LR} N_B={N_BTASKS} resize={RESIZE}", flush=True)
    banks = load_banks(); print(f"[REP] loaded {len(banks)} class banks ({time.time()-t0:.0f}s)", flush=True)
    Xneg_u8 = torch.cat([banks[c] for c in NEG_CLASSES])

    # ORIGINAL-model features (once): per-A probe AUC baseline + directions
    base = build_model(RANK)
    Fneg0 = features(base, Xneg_u8)
    FA0 = {a: features(base, banks[a]) for a in A_CLASSES}
    accA0 = {a: probe_auc(FA0[a], Fneg0) for a in A_CLASSES}
    dA = {a: ddir(FA0[a], Fneg0) for a in A_CLASSES}
    print(f"[REP] base accA0 median={np.median(list(accA0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(1)
    points = []   # (cos, forget, accA0, B_idx)
    for bi in range(N_BTASKS):
        bclasses = list(rng.choice(B_POOL, BCLS, replace=False))
        model = build_model(RANK)                                  # fresh LoRA from the same frozen backbone
        finetune_B(model, banks, bclasses, B_STEPS, B_LR)
        Fneg1 = features(model, Xneg_u8)
        FB0 = np.concatenate([features(base, banks[c]) for c in bclasses])   # B direction in ORIGINAL phi (t=0)
        dB = ddir(FB0, Fneg0)
        for a in A_CLASSES:
            acc1 = probe_auc(features(model, banks[a]), Fneg1)
            points.append({"B": bi, "A": a, "cos": round(float(dA[a] @ dB), 4),
                           "forget": round(accA0[a] - acc1, 4), "accA0": round(accA0[a], 4)})
        fs = [p["forget"] for p in points if p["B"] == bi]
        print(f"  B{bi} {bclasses}: median_forget={np.median(fs):+.3f} [{min(fs):+.3f},{max(fs):+.3f}] ({time.time()-t0:.0f}s)", flush=True)

    cs = np.array([p["cos"] for p in points]); fg = np.array([p["forget"] for p in points])
    med = float(np.median(fg)); rho = spearman(cs, fg)
    inband = 0.05 <= med <= 0.40
    print(f"\n[REP] n={len(points)} median_forget={med:+.3f} (non-sat band [0.05,0.40]: {'OK' if inband else 'OUT'})", flush=True)
    print(f"[REP] Spearman rho(cos,forget)={rho:+.3f}  Pearson={float(np.corrcoef(cs,fg)[0,1]):+.3f}", flush=True)
    # binned curve
    if len(points) > 30:
        ed = np.quantile(cs, np.linspace(0, 1, 6))
        for i in range(5):
            m = (cs >= ed[i]) & (cs <= ed[i+1] if i == 4 else cs < ed[i+1])
            if m.sum(): print(f"    cos[{ed[i]:+.2f},{ed[i+1]:+.2f}] n={m.sum():3d} forget={fg[m].mean():+.3f}", flush=True)

    verdict = ("GATE PASS: partial-forgetting regime exists -> proceed to full (add dtheta/Fisher, rank sweep, depth, seeds)"
               if inband else
               "GATE FAIL: forgetting saturated/absent -> retune B_STEPS/B_LR/RANK before the full curve")
    print(f"\n[REP] {verdict}", flush=True)
    OUT.write_text(json.dumps({"config": {"mode": MODE, "rank": RANK, "B_steps": B_STEPS, "B_lr": B_LR, "n_btasks": N_BTASKS,
                                          "resize": RESIZE, "per_class": PER_CLASS, "prereg_lock": "1bab16d2"},
                               "median_forget": med, "inband": inband, "spearman": rho, "points": points}, indent=1))
    print(f"[REP] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
