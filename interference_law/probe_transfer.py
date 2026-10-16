"""
probe_transfer.py ? NEW-3 (PREREG lock 52671eee): is the forgetting law also the TRANSFER law? Frozen resnet18, HARD victim task (1-vs-rest,
headroom), sequential head continual A->B, retention change d = AUC_A(after B) - AUC_A(base). d>0 = backward TRANSFER, d<0 = forgetting.
Test whether ONE signed cosine axis governs the full continuum. CIFAR + DTD. $0 local.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A

DEV = A.DEV
CAPS = list(range(20)); PER_CLASS = R._i("PER_CLASS", 100); SIZE = 128
NTR_POS = R._i("NTR_POS", 45); NTR_NEG = R._i("NTR_NEG", 135); STEPS0 = R._i("STEPS0", 120); STEPSB = R._i("STEPSB", 70)
NSEED = R._i("NSEED", 3); SEED = R._i("SEED", 0); D_PROJ = R._i("D_PROJ", 0)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
ROOT = Path(os.environ.get("TVDATA_ROOT", str(HERE / "_data" / "tvdata")))
OUT = HERE / "out" / "transfer.json"; LOCK = "52671eee"
torch.manual_seed(SEED); np.random.seed(SEED)


def get_banks(name):
    if name == "cifar":
        b = R.load_banks(); return {c: b[c] for c in CAPS}
    import torchvision.datasets as tvd
    import torchvision.transforms.functional as TF
    ROOT.mkdir(parents=True, exist_ok=True)
    srcs = [tvd.DTD(str(ROOT), split=s, partition=1, download=True) for s in ("train", "val", "test")]
    banks = {c: [] for c in CAPS}
    for ds in srcs:
        for i in range(len(ds)):
            img, lab = ds[i]; lab = int(lab)
            if lab in CAPS and len(banks[lab]) < PER_CLASS:
                banks[lab].append(TF.pil_to_tensor(TF.resize(img.convert("RGB"), [SIZE, SIZE])))
        if all(len(banks[c]) >= PER_CLASS for c in CAPS): break
    return {c: torch.stack(v) for c, v in banks.items() if len(v) >= 40}


def split_idx(n, tag, frac=0.4):
    import hashlib
    te = np.array([(int(hashlib.md5(f"{tag}{i}".encode()).hexdigest(), 16) % 100) < frac*100 for i in range(n)])
    return ~te, te


def train_head(dim, Xtr, ytr, steps, seed, init=None):
    torch.manual_seed(seed); h = nn.Linear(dim, 1).to(DEV)
    if init is not None: h.load_state_dict(init)
    opt = torch.optim.AdamW(h.parameters(), lr=0.03, weight_decay=1e-3)
    X = torch.from_numpy(Xtr).to(DEV); y = torch.tensor(ytr, dtype=torch.float32).to(DEV)
    pw = torch.tensor([float((ytr == 0).sum())/max(float((ytr == 1).sum()), 1)], device=DEV)
    for _ in range(steps):
        opt.zero_grad(); Fun.binary_cross_entropy_with_logits(h(X).squeeze(1), y, pos_weight=pw).backward(); opt.step()
    return h

def auc_head(h, Xte, yte):
    with torch.no_grad(): sc = h(torch.from_numpy(Xte).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(sc, yte)


def run(name):
    t0 = time.time(); print(f"[XFER] === {name} ===", flush=True)
    banks = get_banks(name); caps = [c for c in CAPS if c in banks]; n = len(caps)
    base = A.build("resnet18"); base.eval(); FT = {c: A.feats(base, banks[c]) for c in caps}; dim = FT[caps[0]].shape[1]
    del base; torch.cuda.empty_cache()
    if D_PROJ > 0:  # random projection caps separability -> headroom (prereg: tune victim task harder if saturated)
        P = (np.random.RandomState(0).randn(dim, D_PROJ) / np.sqrt(D_PROJ)).astype(np.float32)
        FT = {c: (FT[c] @ P).astype(np.float32) for c in caps}; dim = D_PROJ
    # per-cap task: pos=feat_c, neg=pooled others (fixed sample); train/test split; direction
    rng = np.random.RandomState(0); task = {}
    allf = {c: FT[c] for c in caps}
    for c in caps:
        pos = FT[c]; others = np.concatenate([FT[o] for o in caps if o != c])
        ptr, pte = split_idx(len(pos), f"{name}p{c}"); negsel = rng.choice(len(others), min(NTR_NEG*3, len(others)), replace=False)
        neg = others[negsel]; ntr, nte = split_idx(len(neg), f"{name}n{c}")
        # limit train size for headroom
        pos_tr = pos[ptr][:NTR_POS]; neg_tr = neg[ntr][:NTR_NEG]
        Xtr = np.concatenate([pos_tr, neg_tr]); ytr = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))]).astype(np.float32)
        Xte = np.concatenate([pos[pte], neg[nte]]); yte = np.concatenate([np.ones(pte.sum()), np.zeros(nte.sum())]).astype(np.float32)
        d = R.ddir(pos, neg)
        task[c] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "d": d}
    cosM = np.array([[float(task[i]["d"] @ task[j]["d"]) for j in caps] for i in caps])

    Delta = np.zeros((n, n)); base_auc = []
    for s in range(NSEED):
        for ci, i in enumerate(caps):
            hA = train_head(dim, task[i]["Xtr"], task[i]["ytr"], STEPS0, seed=1000*s + ci)
            a0 = auc_head(hA, task[i]["Xte"], task[i]["yte"])
            if s == 0: base_auc.append(a0)
            init = {k: v.clone() for k, v in hA.state_dict().items()}
            for cj, j in enumerate(caps):
                if i == j: continue
                hAB = train_head(dim, task[j]["Xtr"], task[j]["ytr"], STEPSB, seed=1000*s + ci, init=init)
                Delta[ci, cj] += (auc_head(hAB, task[i]["Xte"], task[i]["yte"]) - a0) / NSEED
        print(f"  [{name}] seed {s} done ({time.time()-t0:.0f}s)", flush=True)

    m = ~np.eye(n, dtype=bool); dv = Delta[m]; cv = cosM[m]
    med_auc = float(np.median(base_auc)); headroom = bool(0.70 <= med_auc <= 0.92)
    xfer_frac = float((dv >= 0.01).mean()); forget_frac = float((dv <= -0.01).mean())
    rho = float(R.spearman(cv, dv)); pear = float(np.corrcoef(cv, dv)[0, 1])
    # zero-crossing: binned d vs cos deciles
    ed = np.quantile(cv, np.linspace(0, 1, 9)); curve = []
    for b in range(8):
        mm = (cv >= ed[b]) & (cv <= ed[b+1] if b == 7 else cv < ed[b+1])
        if mm.sum(): curve.append((round(float((ed[b]+ed[b+1])/2), 3), round(float(dv[mm].mean()), 4)))
    c0 = None
    for k in range(1, len(curve)):
        if curve[k-1][1] < 0 <= curve[k][1]: c0 = round((curve[k-1][0]+curve[k][0])/2, 3); break

    p1 = bool(headroom and xfer_frac >= 0.15); p2 = bool(rho >= 0.40)
    print(f"  [{name}] base_auc_med={med_auc:.3f} headroom={headroom} | transfer_frac(d>=.01)={xfer_frac:.2f} forget_frac={forget_frac:.2f} "
          f"drange[{dv.min():+.3f},{dv.max():+.3f}]", flush=True)
    print(f"  [{name}] rho(cos,d)={rho:+.3f} pearson={pear:+.3f} | zero-crossing c0={c0} | d-vs-cos curve: {curve}", flush=True)
    print(f"  [{name}] NEW3-P1(transfer exists)={p1} NEW3-P2(one signed law)={p2} ({time.time()-t0:.0f}s)", flush=True)
    return {"n": n, "base_auc_median": round(med_auc, 4), "headroom": headroom, "transfer_frac": round(xfer_frac, 3),
            "forget_frac": round(forget_frac, 3), "delta_range": [round(float(dv.min()), 4), round(float(dv.max()), 4)],
            "rho_cos_delta": round(rho, 3), "pearson_cos_delta": round(pear, 3), "zero_crossing_c0": c0,
            "delta_vs_cos_curve": curve, "checks": {"P1_transfer_exists": p1, "P2_one_signed_law": p2}}


def main():
    t0 = time.time(); print(f"[XFER] lock={LOCK} datasets={DATASETS} caps={len(CAPS)} nseed={NSEED} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "ntr": [NTR_POS, NTR_NEG], "steps": [STEPS0, STEPSB], "nseed": NSEED,
                                   "seed": SEED, "prereg_lock": LOCK}, "results": results}, indent=1))
    print("\n=== FORGETTING LAW = TRANSFER LAW? (frozen, signed retention d vs cos) ===", flush=True)
    print(f"  {'dataset':8}{'baseAUC':>8}{'xferFrac':>9}{'forgetFrac':>11}{'rho(cos,d)':>11}{'c0':>7}   P1 P2", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        c = r["checks"]
        print(f"  {nme:8}{r['base_auc_median']:>8.3f}{r['transfer_frac']:>9.2f}{r['forget_frac']:>11.2f}{r['rho_cos_delta']:>11.3f}"
              f"{str(r['zero_crossing_c0']):>7}   {int(c['P1_transfer_exists'])}  {int(c['P2_one_signed_law'])}", flush=True)
    ok = [n for n in DATASETS if "error" not in results.get(n, {}) and results[n].get("headroom")]
    p1 = all(results[n]["checks"]["P1_transfer_exists"] for n in ok) if ok else False
    p2 = all(results[n]["checks"]["P2_one_signed_law"] for n in ok) if ok else False
    verdict = ("ONE SIGNED LAW: forgetting law IS the transfer law (transfer=+cos tail), dataset-general" if p1 and p2 else
               "transfer exists but NOT cosine-governed (separate axis)" if p1 else
               "NO transfer even with headroom (forgetting is one-sided)" if ok else "no headroom -> untestable, retune")
    print(f"\n[XFER] headroom_ok={ok} P1={p1} P2={p2} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[XFER] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
