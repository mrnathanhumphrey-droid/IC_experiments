"""
probe_dirlaw_arch.py — NEW-1a'''' (PREREG lock 8b4cf4c6): does the plastic DIRECTIONAL LAW forget∝−cos(g_i,Δθ_p) survive an
ARCHITECTURE switch? Same 1a''' decomposition on resnet18 / mobilenet_v3_small / vit_b_16 (CNN+BN / depthwise-SE-CNN / pure transformer).
CIFAR-100 held; architecture varied. $0 local.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R      # load_banks/auc/spearman/_i/_f/NEG_CLASSES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VICTIMS = list(range(24)); PERTURBERS = [50, 55, 60, 65, 70, 75]; NEG = list(range(90, 100))
ARCHS = {"resnet18": 5e-4, "mobilenet_v3_small": 5e-4, "vit_b_16": 1e-4}      # per-arch FT LR
FT_STEPS = R._i("FT_STEPS", 300); NJ = R._i("NJ", 128); SEED = R._i("SEED", 0)
OUT = HERE / "out" / "dirlaw_arch.json"; LOCK = "8b4cf4c6"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
torch.manual_seed(SEED); np.random.seed(SEED)


def prep(u8):
    x = u8.float() / 255.0; x = Fun.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD

def build(name):
    import torchvision.models as tvm
    if name == "resnet18":
        m = tvm.resnet18(weights="IMAGENET1K_V1"); net = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    elif name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1"); m.classifier = nn.Identity(); net = m
    elif name == "vit_b_16":
        m = tvm.vit_b_16(weights="IMAGENET1K_V1"); m.heads = nn.Identity(); net = m
    else: raise ValueError(name)
    return net.to(DEV)

@torch.no_grad()
def feats(net, u8, bs=128):
    net.eval(); outs = []
    for i in range(0, len(u8), bs): outs.append(net(prep(u8[i:i+bs]).to(DEV)).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)

def theta(net): return torch.cat([p.detach().float().flatten() for p in net.parameters()]).cpu()

def train_head1(Fa, Fn, dim, ep=150):
    h = nn.Linear(dim, 1).to(DEV); opt = torch.optim.AdamW(h.parameters(), lr=0.05, weight_decay=1e-4)
    X = torch.from_numpy(np.concatenate([Fa, Fn])).to(DEV)
    y = torch.tensor([1.]*len(Fa) + [0.]*len(Fn)).to(DEV)
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
    for _ in range(steps):
        idx = torch.randint(0, len(yb), (128,))
        loss = Fun.cross_entropy(head(m(prep(Xb[idx]).to(DEV))), yb[idx].to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval(); return m

def sp(x, y): return float(R.spearman(np.asarray(x, float).ravel(), np.asarray(y, float).ravel()))
def dcenter(X): return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()


def run_arch(name, lr, banks, Xneg_u8):
    t0 = time.time(); print(f"[ARCH] === {name} (lr={lr}) ===", flush=True)
    base = build(name); base.eval()
    dim = feats(base, banks[VICTIMS[0]][:2]).shape[1]
    Fneg0 = feats(base, Xneg_u8)
    heads, auc0 = {}, {}
    for i in VICTIMS:
        FA0 = feats(base, banks[i]); heads[i] = train_head1(FA0, Fneg0, dim); auc0[i] = head_auc(heads[i], FA0, Fneg0)
    print(f"  [{name}] dim={dim} heads ready, base AUC med={np.median(list(auc0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    theta0 = theta(base); nV, nP = len(VICTIMS), len(PERTURBERS)
    dthetas, dmag, FORGET = [], np.zeros(nP), np.zeros((nV, nP))
    for pj, p in enumerate(PERTURBERS):
        Xb = torch.cat([banks[p], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[p])), torch.zeros(len(Xneg_u8))]).long()
        mm = copy.deepcopy(base); full_ft(mm, dim, Xb, yb, FT_STEPS, lr)
        dth = (theta(mm) - theta0).float(); dthetas.append(dth); dmag[pj] = float(dth.norm())
        Fneg1 = feats(mm, Xneg_u8)
        for ci, i in enumerate(VICTIMS): FORGET[ci, pj] = auc0[i] - head_auc(heads[i], feats(mm, banks[i]), Fneg1)
        del mm; torch.cuda.empty_cache()
        print(f"    perturb {p}: |dth|={dmag[pj]:.2f} medFgt_col={np.median(FORGET[:,pj]):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    Xn_g = prep(Xneg_u8[:NJ]).to(DEV); gmag = np.zeros(nV); DOT = np.zeros((nV, nP)); COS = np.zeros((nV, nP))
    for ci, i in enumerate(VICTIMS):
        base.zero_grad(set_to_none=True)
        m = heads[i](base(prep(banks[i][:NJ]).to(DEV))).mean() - heads[i](base(Xn_g)).mean(); m.backward()
        g = torch.cat([pp.grad.detach().flatten() for pp in base.parameters()]).float().cpu(); gmag[ci] = float(g.norm())
        for pj in range(nP): DOT[ci, pj] = float(g @ dthetas[pj]); COS[ci, pj] = DOT[ci, pj]/(gmag[ci]*dmag[pj]+1e-12)
    base.zero_grad(set_to_none=True)

    ALIGN = -COS; GMAGb = np.repeat(gmag[:, None], nP, 1)
    med = float(np.median(FORGET)); inband = bool(0.03 <= med <= 0.45)
    tbl = {"fo_full": sp(-DOT, FORGET), "align": sp(ALIGN, FORGET), "align_dcenter": sp(dcenter(ALIGN), dcenter(FORGET)),
           "gmag_broadcast": sp(GMAGb, FORGET)}
    p1 = bool(tbl["align"] >= 0.25); p2 = bool(tbl["align_dcenter"] >= 0.15)
    p3 = bool((tbl["align"] > tbl["gmag_broadcast"]) and (tbl["gmag_broadcast"] <= 0.10))
    print(f"  [{name}] medFgt={med:+.3f} inband={inband} | align={tbl['align']:+.3f} align_dc={tbl['align_dcenter']:+.3f} "
          f"fo={tbl['fo_full']:+.3f} gmag={tbl['gmag_broadcast']:+.3f} cos[{COS.min():+.3f},{COS.max():+.3f}] | "
          f"P1={p1} P2={p2} P3={p3} ({time.time()-t0:.0f}s)", flush=True)
    del base, dthetas; torch.cuda.empty_cache()
    return {"dim": dim, "lr": lr, "median_forget": round(med, 4), "inband": inband, "dmag_range": [round(float(dmag.min()), 2), round(float(dmag.max()), 2)],
            "cos_range": [round(float(COS.min()), 4), round(float(COS.max()), 4)], "decomp": {k: round(v, 3) for k, v in tbl.items()},
            "checks": {"P1_align": p1, "P2_interaction": p2, "P3_magpattern": p3}}


def main():
    t0 = time.time()
    print(f"[ARCH] lock={LOCK} archs={list(ARCHS)} victims={len(VICTIMS)} perturbers={len(PERTURBERS)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in NEG])
    print(f"[ARCH] CIFAR banks loaded ({time.time()-t0:.0f}s)", flush=True)
    results = {}
    for name, lr in ARCHS.items():
        try:
            results[name] = run_arch(name, lr, banks, Xneg_u8)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:160]}", flush=True)
        OUT.write_text(json.dumps({"config": {"victims": VICTIMS, "perturbers": PERTURBERS, "ft_steps": FT_STEPS, "archs": ARCHS,
                                   "seed": SEED, "prereg_lock": LOCK}, "results": results}, indent=1))

    print("\n=== DIRECTIONAL LAW ACROSS ARCHITECTURES ===", flush=True)
    print(f"  {'arch':20}{'dim':>5}{'medFgt':>8}{'align':>8}{'align_dc':>10}{'gmag':>8}  P1P2P3", flush=True)
    for n, r in results.items():
        if "error" in r: print(f"  {n:20} ERROR {r['error'][:60]}", flush=True); continue
        d = r["decomp"]; c = r["checks"]; flag = "" if r["inband"] else " [OOB]"
        print(f"  {n:20}{r['dim']:>5}{r['median_forget']:>8.3f}{d['align']:>8.3f}{d['align_dcenter']:>10.3f}{d['gmag_broadcast']:>8.3f}  "
              f"{int(c['P1_align'])}{int(c['P2_interaction'])}{int(c['P3_magpattern'])}{flag}", flush=True)
    nonres = [n for n in ARCHS if n != "resnet18" and "error" not in results.get(n, {}) and results[n].get("inband")]
    holds = [n for n in nonres if results[n]["checks"]["P1_align"] and results[n]["checks"]["P2_interaction"]]
    vit_ok = "vit_b_16" in holds
    verdict = ("HARDENED: directional law is ARCHITECTURE-GENERAL (holds on transformer + CNN)" if vit_ok and len(holds) >= 1 else
               "CONV-SPECIFIC: holds on CNN not transformer" if holds and not vit_ok else
               "SUBSTRATE-BOUND: directional law does NOT replicate off ResNet18 -> retract generality")
    print(f"\n[ARCH] holds_on={holds} vit_ok={vit_ok} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; d["holds_on"] = holds; OUT.write_text(json.dumps(d, indent=1))
    print(f"[ARCH] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
