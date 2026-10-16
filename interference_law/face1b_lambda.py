"""
face1b_lambda.py — POWERED adjudicator (PREREG lock in PREREG_FACE1B_DOMINANCE_POWERED_2026_07_17.md). Runs on Lambda GPU.
Reuses probe_cyclic.build_F/.hodge (the phi machinery) + probe_dirlaw_arch feature/FT path (shipped alongside — NOT re-coded).
For {CIFAR,DTD} x 16 caps x 3 seeds: builds directed F per seed, phi=hodge(F), seed-averages phi to de-noise, then runs the SAME
pre-committed predictability regression as the screen (trio {dprime,wnorm,spread}, d' primary, ridge lam=1.0, LOO, bootstrap, permutation).
"""
from __future__ import annotations
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_cyclic as CYC
import probe_dirlaw_arch as A
import probe_representational_law as R

# remote-safe DTD download dir (probe_cyclic.ROOT is a Windows path)
CYC.ROOT = Path("/home/ubuntu/ilaw/tvdata"); CYC.ROOT.mkdir(parents=True, exist_ok=True)

CAPS = list(range(16))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
TRIO = ["dprime", "wnorm", "spread"]; PRIMARY = "dprime"; RIDGE_LAM = 1.0
ALL_PREDS = ["dprime", "spread", "munorm", "wnorm", "iso"]
PREREG = HERE / "PREREG_FACE1B_DOMINANCE_POWERED_2026_07_17.md"
LOCK = hashlib.sha256(PREREG.read_bytes()).hexdigest()[:8] if PREREG.exists() else "nolock"
OUT = HERE / "out" / os.environ.get("OUTNAME", "face1b_dominance_apriori.json"); OUT.parent.mkdir(parents=True, exist_ok=True)


def load_cifar_banks():
    """torchvision CIFAR-100 loader — bypasses the HF datasets/PIL-Exif path. Same format as R.load_banks: {c: uint8 [n,3,32,32]}."""
    import torchvision
    ds = torchvision.datasets.CIFAR100(root="/home/ubuntu/ilaw/tvdata", train=True, download=True)
    data = ds.data; labels = np.array(ds.targets); rng = np.random.RandomState(0); PER = 100; banks = {}
    for c in range(100):
        idx = np.where(labels == c)[0]; rng.shuffle(idx); idx = idx[:PER]
        banks[c] = torch.from_numpy(data[idx]).permute(0, 3, 1, 2).contiguous()  # [PER,3,32,32] uint8
    return banks


def predictors(base, banks, caps, neg, sl):
    """Same defs as probe_fragility, via the F-building feature/head path (A.feats / A.train_head1)."""
    Xneg = torch.cat([banks[c][sl] for c in neg]); Fneg = A.feats(base, Xneg); dim = Fneg.shape[1]
    FA = {c: A.feats(base, banks[c][sl]) for c in caps}
    dvec = {c: (lambda d: d/(np.linalg.norm(d)+1e-12))(FA[c].mean(0) - Fneg.mean(0)) for c in caps}
    heads = {c: A.train_head1(FA[c], Fneg, dim) for c in caps}
    wax = {c: (lambda w: (w/np.linalg.norm(w)).astype(np.float32))(heads[c].weight.detach().cpu().numpy()[0].astype(np.float64)) for c in caps}
    dprime, spread, munorm, wnorm = {}, {}, {}, {}
    for c in caps:
        sa = FA[c] @ wax[c]; sn = Fneg @ wax[c]
        dprime[c] = float((sa.mean() - sn.mean()) / np.sqrt((sa.var() + sn.var())/2 + 1e-12))
        spread[c] = float(FA[c].std(0).mean()); munorm[c] = float(np.linalg.norm(FA[c].mean(0)))
        wnorm[c] = float(np.linalg.norm(heads[c].weight.detach().cpu().numpy()))
    iso = {c: float(np.mean([dvec[c] @ dvec[k] for k in caps if k != c])) for c in caps}
    return {"dprime": dprime, "spread": spread, "munorm": munorm, "wnorm": wnorm, "iso": iso}


# ---- stats (identical to probe_face1_dominance_apriori) ----
def sp(x, y): return float(R.spearman(np.asarray(x, float), np.asarray(y, float)))

def boot_ci(x, y, alpha=95.0, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); m = len(x); rng = np.random.RandomState(seed); rs = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12: continue
        rs.append(sp(x[idx], y[idx]))
    lo = (100 - alpha) / 2
    return [round(float(np.percentile(rs, lo)), 3), round(float(np.percentile(rs, 100 - lo)), 3)]

def perm_p(x, y, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); obs = abs(sp(x, y)); rng = np.random.RandomState(seed); c = 0
    for _ in range(n): c += abs(sp(x, rng.permutation(y))) >= obs
    return round((c + 1) / (n + 1), 4)

def ridge_loo(X, y, lam=RIDGE_LAM):
    X = np.asarray(X, float); y = np.asarray(y, float); n, p = X.shape; pred = np.zeros(n)
    for k in range(n):
        tr = [j for j in range(n) if j != k]; Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9; Xs = (Xtr - mu)/sd; ym = ytr.mean()
        b = np.linalg.solve(Xs.T @ Xs + lam*np.eye(p), Xs.T @ (ytr - ym))
        pred[k] = ym + ((X[k] - mu)/sd) @ b
    r2 = float(1 - ((y - pred)**2).sum() / (((y - y.mean())**2).sum() + 1e-12))
    return pred, r2

def perm_null_loo95(X, y, n=1000, seed=0):
    y = np.asarray(y, float); rng = np.random.RandomState(seed); null = [ridge_loo(X, rng.permutation(y))[1] for _ in range(n)]
    return float(np.percentile(null, 95))


def regress(target, preds, caps):
    y = np.array([target[c] for c in caps], float)
    uni = {}
    for pn in ALL_PREDS:
        x = np.array([preds[pn][c] for c in caps], float)
        ci = boot_ci(x, y, 95.0 if pn == PRIMARY else 99.2)
        uni[pn] = {"rho": round(sp(x, y), 3), "ci": ci, "perm_p": perm_p(x, y), "excl0": bool(ci[0] > 0 or ci[1] < 0)}
    X = np.stack([[preds[pn][c] for c in caps] for pn in TRIO], 1)
    loo_pred, loo_r2 = ridge_loo(X, y); null95 = perm_null_loo95(X, y)
    ord_rho = sp(loo_pred, y); rng = np.random.RandomState(0); c = 0
    for _ in range(1000): c += abs(sp(ridge_loo(X, rng.permutation(y))[0], y)) >= abs(ord_rho)
    return {"n": len(caps), "univariate": uni, "trio_loo_r2": round(loo_r2, 3), "perm_null_loo_r2_95": round(null95, 3),
            "ordinal_loo_rankcorr": round(ord_rho, 3), "ordinal_perm_p": round((c + 1)/1001, 4)}


def run_dataset(name):
    t0 = time.time(); print(f"[F1b] === {name} ===", flush=True)
    banks = load_cifar_banks() if name == "cifar" else CYC.get_banks(name)
    caps = [c for c in CAPS if c in banks]; neg = CYC.NEG_BY[name]
    full = slice(0, len(banks[caps[0]]))
    base = A.build("resnet18"); base.eval()
    preds = predictors(base, banks, caps, neg, full)
    print(f"  [{name}] predictors done, {len(caps)} caps ({time.time()-t0:.0f}s)", flush=True)
    phis, frags, disrs = [], [], []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        F = CYC.build_F(base, banks, caps, neg, full)
        h, _, gf, cf = CYC.hodge(F)
        frag = np.array([np.nanmean([F[i, j] for j in range(len(caps)) if j != i]) for i in range(len(caps))])
        disr = np.array([np.nanmean([F[i, j] for i in range(len(caps)) if i != j]) for j in range(len(caps))])
        phis.append(h); frags.append(frag); disrs.append(disr)
        print(f"  [{name}] seed {s}: F built, grad%={gf:.2f} curl%={cf:.2f} ({time.time()-t0:.0f}s)", flush=True)
    phis = np.array(phis); phi_mean = phis.mean(0)
    # cross-seed reliability: pairwise Spearman of phi between seeds
    rel = [sp(phis[a], phis[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))]
    phi_d = {c: float(phi_mean[i]) for i, c in enumerate(caps)}
    frag_d = {c: float(np.mean(frags, 0)[i]) for i, c in enumerate(caps)}
    disr_d = {c: float(np.mean(disrs, 0)[i]) for i, c in enumerate(caps)}
    return {"caps": [int(c) for c in caps], "cross_seed_phi_rho": [round(r, 3) for r in rel],
            "cross_seed_phi_rho_mean": round(float(np.mean(rel)), 3), "preds": preds,
            "phi_mean": phi_d, "frag_mean": frag_d, "disrupt_mean": disr_d,
            "phi_regression": regress(phi_d, preds, caps),
            "frag_regression": regress(frag_d, preds, caps),
            "disrupt_regression": regress(disr_d, preds, caps)}


def main():
    t0 = time.time(); print(f"[F1b] lock={LOCK} seeds={SEEDS} datasets={DATASETS} dev={A.DEV}", flush=True)
    res = {}
    for name in DATASETS:
        try:
            res[name] = run_dataset(name)
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
        OUT.write_text(json.dumps({"config": {"lock": LOCK, "seeds": SEEDS, "trio": TRIO, "ridge_lam": RIDGE_LAM}, "results": res}, indent=1))

    # POOLED (n<=32): stack (predictor, phi) pairs across datasets
    ok = [n for n in DATASETS if "error" not in res.get(n, {})]
    pooled = None
    if ok:
        pooled_preds = {pn: {} for pn in ALL_PREDS}; pooled_phi = {}; k = 0
        for name in ok:
            r = res[name]
            for c in r["caps"]:
                for pn in ALL_PREDS: pooled_preds[pn][k] = r["preds"][pn][c]
                pooled_phi[k] = r["phi_mean"][c]; k += 1
        pooled = regress(pooled_phi, pooled_preds, list(range(k)))

    # decisions
    def verdict_for(reg, rel_mean):
        card = bool(reg["trio_loo_r2"] >= max(0.30, reg["perm_null_loo_r2_95"]) and reg["univariate"][PRIMARY]["excl0"])
        ordn = bool(reg["ordinal_loo_rankcorr"] >= 0.50 and reg["ordinal_perm_p"] < 0.05)
        return {"cardinal": card, "ordinal": ordn}
    cif = res.get("cifar", {}).get("phi_regression"); cif_rel = res.get("cifar", {}).get("cross_seed_phi_rho_mean")
    dec = {"cifar": verdict_for(cif, cif_rel) if cif else None,
           "pooled": verdict_for(pooled, None) if pooled else None,
           "pooled_n": pooled["n"] if pooled else 0}
    ordinal_win = (dec["cifar"] and dec["cifar"]["ordinal"]) or (dec["pooled"] and dec["pooled"]["ordinal"])
    cardinal_win = (dec["cifar"] and dec["cifar"]["cardinal"]) or (dec["pooled"] and dec["pooled"]["cardinal"])
    rels = [res[n]["cross_seed_phi_rho_mean"] for n in ok]
    denoised_ok = all(r > 0.51 for r in rels) if rels else False
    if ordinal_win and cardinal_win: headline = "SCREEN OVERTURNED — pecking order a-priori computable (ORDINAL + CARDINAL)."
    elif ordinal_win: headline = "SCREEN OVERTURNED — the pecking order IS a-priori computable (ORDINALLY); n=12 null was attenuation."
    elif denoised_ok: headline = "CONFIRMED — order is genuinely EMERGENT from interaction (de-noised phi, still no a-priori signal)."
    else: headline = "INCONCLUSIVE — phi stays intrinsically noisy even seed-averaged (upstream measurement limit, not a clean null)."

    full = json.loads(OUT.read_text()); full["pooled_phi_regression"] = pooled
    full["decision"] = {**dec, "ordinal_win": ordinal_win, "cardinal_win": cardinal_win, "denoised_ok": denoised_ok,
                        "cross_seed_reliability": {n: res[n]["cross_seed_phi_rho_mean"] for n in ok}, "headline": headline}
    OUT.write_text(json.dumps(full, indent=1))
    print(f"\n[F1b] cross-seed phi reliability: " + ", ".join(f"{n}={res[n]['cross_seed_phi_rho_mean']}" for n in ok), flush=True)
    for n in ok:
        pr = res[n]["phi_regression"]
        print(f"  [{n}] phi: LOO-R2={pr['trio_loo_r2']:+.3f}(null95 {pr['perm_null_loo_r2_95']:.3f}) ordinal={pr['ordinal_loo_rankcorr']:+.3f}(p{pr['ordinal_perm_p']}) rho(d',phi)={pr['univariate'][PRIMARY]['rho']:+.3f}{pr['univariate'][PRIMARY]['ci']}", flush=True)
    if pooled: print(f"  [POOLED n={pooled['n']}] phi: LOO-R2={pooled['trio_loo_r2']:+.3f} ordinal={pooled['ordinal_loo_rankcorr']:+.3f}(p{pooled['ordinal_perm_p']})", flush=True)
    print(f"\n[F1b] HEADLINE: {headline}", flush=True)
    print(f"[F1b] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
