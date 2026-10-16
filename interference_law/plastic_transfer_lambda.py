"""
plastic_transfer_lambda.py -- Transfer-geometry P2 (PREREG P2_2026_07_26). CANONICAL PLASTIC regime: source j reshapes the
backbone via LoRA-FT; measure FORWARD transfer FT(i<-j) through the reshaped features + feature-reuse descriptors (CKA/subspace/
LogME/eff-rank) that are now SOURCE-DEPENDENT. Tests: Q1 does the cosine dissociation persist; Q2 does any feature-reuse descriptor
predict plastic FT (>= LogME). Reuses face1c_p4_lambda helpers. Per-source incremental JSONL (survives watchdog kill). ASCII-only.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import face1c_p4_lambda as P4   # load_banks, acts_raw, lora_ft, train_head, head_auc, sp, MODEL_MAP

OUT = HERE / "out" / os.environ.get("OUTNAME", "plastic_transfer.json"); OUT.parent.mkdir(parents=True, exist_ok=True)
ROWS = HERE / "out" / os.environ.get("ROWSNAME", "plastic_rows.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE", "0") == "1"

MODEL = os.environ.get("MODEL", "pythia1.4b")
DATASETS = os.environ.get("DATASETS", "clinc150" if SMOKE else "clinc150,banking77").split(",")
NCAP = 3 if SMOKE else int(os.environ.get("NCAP", "14"))
SEEDS = [0, 1] if SMOKE else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
STEPS = 40 if SMOKE else int(os.environ.get("STEPS", "180"))
PER = 24 if SMOKE else int(os.environ.get("PER", "80"))
PROJDIM = 32 if SMOKE else int(os.environ.get("PROJDIM", "64"))
NFEW_POS = int(os.environ.get("NFEW_POS", "8")); NFEW_NEG = int(os.environ.get("NFEW_NEG", "24")); FEW_EP = int(os.environ.get("FEW_EP", "60"))
KPC = int(os.environ.get("KPC", "5")); BOOT = int(os.environ.get("BOOT", "2000")); PERM = int(os.environ.get("PERM", "2000"))
CAPS = list(range(NCAP)); NEG = list(range(60, 70))
LAYER_ENV = os.environ.get("LAYER", "")

from transformers import AutoModelForCausalLM, AutoTokenizer

# patch P4 globals so its load_banks/acts use our sizes
P4.CAPS = CAPS; P4.NEG = NEG; P4.PER = PER; P4.SMOKE = SMOKE
P4.STEPS = STEPS; P4.NCAP = NCAP


def proj_of(hdim):
    return (np.random.RandomState(777).randn(hdim, PROJDIM) / np.sqrt(hdim)).astype(np.float32)


def ddir(Fpos, Fneg):
    d = Fpos.mean(0) - Fneg.mean(0); return (d / (np.linalg.norm(d) + 1e-12)).astype(np.float64)


def linear_cka(X, Y):
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    m = min(len(X), len(Y)); X = X[:m]; Y = Y[:m]   # CKA needs equal row counts (banking77 caps vary in size)
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc, 'fro')**2
    return float(hsic / (np.linalg.norm(Xc.T @ Xc, 'fro') * np.linalg.norm(Yc.T @ Yc, 'fro') + 1e-12))


def topk(X, k):
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def subovl(X, Y, k):
    Ux, Uy = topk(X, k), topk(Y, k)
    s = np.linalg.svd(Ux.T @ Uy, compute_uv=False)
    return float(np.mean(s**2))


def effrank(X):
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
    s = np.linalg.svd(Xc, compute_uv=False); s = s[s > 1e-9]
    if len(s) == 0: return 0.0
    p = s / s.sum(); return float(np.exp(-(p * np.log(p)).sum()))


def logme(F, y):
    F = np.asarray(F, np.float64); y = np.asarray(y, np.float64) - float(np.mean(y))
    N, D = F.shape; yy = float(y @ y)
    if yy < 1e-12: return 0.0
    U, s, _ = np.linalg.svd(F, full_matrices=False); k = len(s); s2 = s**2
    z = U.T @ y; z2 = z**2; res_out = max(yy - float(z2.sum()), 1e-12)
    alpha, beta = 1.0, 1.0
    for _ in range(100):
        t = alpha + beta*s2
        gamma = float((beta*s2/t).sum())
        m2 = float(((beta*s*z/t)**2).sum()); mfit = res_out + float(((alpha*z/t)**2).sum())
        a2 = gamma/(m2+1e-12); b2 = (N-gamma)/(mfit+1e-12)
        if abs(a2-alpha) < 1e-6 and abs(b2-beta) < 1e-6: alpha, beta = a2, b2; break
        alpha, beta = a2, b2
    t = alpha + beta*s2
    logdetA = float(np.log(t).sum()) + (D-k)*np.log(alpha)
    m2 = float(((beta*s*z/t)**2).sum()); mfit = res_out + float(((alpha*z/t)**2).sum())
    logev = 0.5*(D*np.log(alpha) + N*np.log(beta) - beta*mfit - alpha*m2 - logdetA - N*np.log(2*np.pi))
    return float(logev / N)


def split_tt(F, frac=0.4):
    n = len(F); c = max(int(n*(1-frac)), 1); return F[:c], F[c:]


def fewauc(Fpos, Fneg, seed):
    ptr, pte = split_tt(Fpos); ntr, nte = split_tt(Fneg)
    rng = np.random.RandomState(seed)
    pi = rng.choice(len(ptr), min(NFEW_POS, len(ptr)), replace=False)
    ni = rng.choice(len(ntr), min(NFEW_NEG, len(ntr)), replace=False)
    h = P4.train_head(ptr[pi], ntr[ni], Fpos.shape[1], ep=FEW_EP)
    return P4.head_auc(h, pte, nte)


def done_keys():
    keys = set()
    if ROWS.exists():
        for ln in ROWS.read_text().splitlines():
            try: r = json.loads(ln); keys.add((r["ds"], r["seed"], r["j"], r["i"]))
            except Exception: pass
    return keys


def run_cell(dsname, tok, base, layer, P, t0):
    banks = P4.load_banks(dsname, tok); caps = [c for c in CAPS if c in banks]; neg = [c for c in NEG if c in banks]
    Xneg = [t for c in neg for t in banks[c]]
    print(f"[PL] {dsname}: {len(caps)} caps {len(neg)} neg layer={layer} ({time.time()-t0:.0f}s)", flush=True)
    base_acts = {c: P4.acts_raw(base, tok, layer, banks[c]) @ P for c in caps}
    neg_base = P4.acts_raw(base, tok, layer, Xneg) @ P
    dbase = {c: ddir(base_acts[c], neg_base) for c in caps}
    cold = {c: fewauc(base_acts[c], neg_base, 12345 + c) for c in caps}
    done = done_keys()
    with open(ROWS, "a") as fh:
        for s in SEEDS:
            for j in caps:
                if all((dsname, s, j, i) in done for i in caps if i != j):
                    print(f"[PL] {dsname} seed{s} src{j}: cached, skip", flush=True); continue
                pm = P4.lora_ft(base, tok, banks[j], STEPS, s*100 + j)
                phi = {i: P4.acts_raw(pm, tok, layer, banks[i]) @ P for i in caps}
                phi_neg = P4.acts_raw(pm, tok, layer, Xneg) @ P
                base = pm.unload(); torch.cuda.empty_cache()
                for i in caps:
                    if i == j or (dsname, s, j, i) in done: continue
                    warm = fewauc(phi[i], phi_neg, 12345 + i)
                    Fi = np.concatenate([phi[i], phi_neg]); yi = np.concatenate([np.ones(len(phi[i])), np.zeros(len(phi_neg))])
                    row = {"ds": dsname, "seed": s, "j": j, "i": i,
                           "FT": round(float(warm - cold[i]), 5), "warm": round(float(warm), 5), "cold": round(float(cold[i]), 5),
                           "COS": round(float(dbase[i] @ dbase[j]), 5),
                           "CKA_drift": round(linear_cka(base_acts[i], phi[i]), 5),
                           "CKA_reuse": round(linear_cka(phi[i], phi[j]), 5),
                           "SUBOVL": round(subovl(phi[i], phi[j], KPC), 5),
                           "LOGME": round(logme(Fi, yi), 5),
                           "EFFRANK": round(effrank(phi[i]), 5)}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[PL] {dsname} seed{s} src{j} done ({time.time()-t0:.0f}s)", flush=True)
    return caps


def spearman(a, b): return P4.sp(a, b)


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; n = len(x)
    if n < 8: return {"rho": None, "ci": [None, None], "perm_p": None, "n": n}
    rho = spearman(x, y); rng = np.random.RandomState(3)
    rb = np.array([spearman(x[idx], y[idx]) for idx in (rng.randint(0, n, n) for _ in range(BOOT))])
    perm = np.array([spearman(x, rng.permutation(y)) for _ in range(PERM)])
    return {"rho": round(rho, 4), "ci": [round(float(np.percentile(rb, 2.5)), 4), round(float(np.percentile(rb, 97.5)), 4)],
            "perm_p": round(float((np.abs(perm) >= abs(rho)).mean()), 4), "n": n}


def aggregate():
    rows = [json.loads(ln) for ln in ROWS.read_text().splitlines() if ln.strip()]
    DESCS = ["COS", "CKA_drift", "CKA_reuse", "SUBOVL", "LOGME", "EFFRANK"]
    out = {}
    for ds in DATASETS:
        R = [r for r in rows if r["ds"] == ds]
        if not R: continue
        caps = sorted(set([r["i"] for r in R]) | set([r["j"] for r in R])); idx = {c: k for k, c in enumerate(caps)}; n = len(caps)
        seeds = sorted(set(r["seed"] for r in R))
        FTs = np.full((len(seeds), n, n), np.nan); Dm = {d: np.full((n, n), np.nan) for d in DESCS}
        for r in R:
            si = seeds.index(r["seed"]); a, b = idx[r["i"]], idx[r["j"]]
            FTs[si, a, b] = r["FT"]
            for d in DESCS: Dm[d][a, b] = r[d]
        m = ~np.eye(n, dtype=bool)
        FT = np.nanmean(FTs, axis=0)
        even = np.nanmean(FTs[0::2], axis=0); odd = np.nanmean(FTs[1::2], axis=0)
        r_sb = None
        if len(seeds) > 1:
            rr = spearman(even[m][np.isfinite(even[m]) & np.isfinite(odd[m])], odd[m][np.isfinite(even[m]) & np.isfinite(odd[m])])
            r_sb = round(2*rr/(1+rr), 4) if rr > -1 else None
        mu = np.nanmean(FT[m]); row = np.nanmean(FT, axis=1, keepdims=True)-mu; col = np.nanmean(FT, axis=0, keepdims=True)-mu
        RES = FT - mu - np.broadcast_to(row, (n, n)) - np.broadcast_to(col, (n, n))
        ftv = FT[m]; resid = RES[m]
        desc_stats = {}
        for d in DESCS:
            xv = Dm[d][m]
            desc_stats[d] = {"vs_FT": corr(xv, ftv), "vs_resid": corr(xv, resid)}
        out[ds] = {"n": n, "nseed": len(seeds), "FT_reliability": r_sb,
                   "FT_mean": round(float(np.nanmean(ftv)), 4), "FT_std": round(float(np.nanstd(ftv)), 4),
                   "var_frac": {"row": round(float(np.nanvar(np.broadcast_to(row, (n, n))[m])/(np.nanvar(ftv)+1e-12)), 3),
                                "col": round(float(np.nanvar(np.broadcast_to(col, (n, n))[m])/(np.nanvar(ftv)+1e-12)), 3),
                                "resid": round(float(np.nanvar(resid)/(np.nanvar(ftv)+1e-12)), 3)},
                   "descriptors": desc_stats}
    # verdict
    def hits(target):
        h = []
        for d in ["CKA_drift", "CKA_reuse", "SUBOVL", "LOGME", "EFFRANK"]:
            ok = all(ds in out for ds in DATASETS if any(r["ds"] == ds for r in rows))
            dss = [ds for ds in out]
            good = all(out[ds]["descriptors"][d][target]["rho"] is not None and abs(out[ds]["descriptors"][d][target]["rho"]) >= 0.30
                       and out[ds]["descriptors"][d][target]["perm_p"] is not None and out[ds]["descriptors"][d][target]["perm_p"] < 0.01 for ds in dss) and len(dss) >= 1
            if good and len({np.sign(out[ds]["descriptors"][d][target]["rho"]) for ds in dss}) == 1: h.append(d)
        return h
    cos_persists = all((out[ds]["descriptors"]["COS"]["vs_FT"]["ci"][0] or -1) > -0.30 and (out[ds]["descriptors"]["COS"]["vs_FT"]["ci"][1] or 1) < 0.30 for ds in out) if out else None
    hit_ft = hits("vs_FT"); hit_res = hits("vs_resid")
    verdict = {"cos_dissociation_persists": cos_persists, "feature_reuse_hits_vs_FT": hit_ft, "feature_reuse_hits_vs_resid": hit_res,
               "headline": ("SECOND-LAW candidate: " + ",".join(sorted(set(hit_ft) | set(hit_res))) if (hit_ft or hit_res)
                            else "NULL: plastic transfer also geometrically opaque (no feature-reuse descriptor clears bar)")}
    return {"config": {"model": MODEL, "datasets": DATASETS, "ncap": NCAP, "seeds": SEEDS, "steps": STEPS, "projdim": PROJDIM, "smoke": SMOKE},
            "results": out, "verdict": verdict}


def main():
    t0 = time.time()
    print(f"[PL] SMOKE={SMOKE} model={MODEL} ds={DATASETS} ncap={NCAP} seeds={SEEDS} steps={STEPS} proj={PROJDIM} dev={DEV}", flush=True)
    model_id = P4.MODEL_MAP.get(MODEL, MODEL)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(DEV); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    nlayers = int(base.config.num_hidden_layers); layer = int(LAYER_ENV) if LAYER_ENV else nlayers // 2
    hdim = int(base.config.hidden_size); P = proj_of(hdim)
    print(f"[PL] {model_id} nlayers={nlayers} layer={layer} hdim={hdim}->{PROJDIM} ({time.time()-t0:.0f}s)", flush=True)
    for ds in DATASETS:
        try:
            run_cell(ds, tok, base, layer, P, t0)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[PL] {ds} ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
        try:
            OUT.write_text(json.dumps(aggregate(), indent=1)); print(f"[PL] aggregated after {ds}; JSON updated ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[PL] aggregate warn: {e}", flush=True)
    agg = aggregate(); OUT.write_text(json.dumps(agg, indent=1))
    print("\n[PL] === PLASTIC TRANSFER ===", flush=True)
    for ds, r in agg["results"].items():
        print(f"  {ds}: FT_rel={r['FT_reliability']} var(row/col/resid)={r['var_frac']} FT={r['FT_mean']:+.3f}+/-{r['FT_std']:.3f}", flush=True)
        for d, st in r["descriptors"].items():
            print(f"    {d:10} vsFT rho={st['vs_FT']['rho']} CI{st['vs_FT']['ci']} p={st['vs_FT']['perm_p']} | vsRESID rho={st['vs_resid']['rho']} p={st['vs_resid']['perm_p']}", flush=True)
    print(f"\n[PL] VERDICT: {agg['verdict']}", flush=True)
    print(f"[PL] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
