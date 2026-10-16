"""
face1c_control_lambda.py — PROJECTION-CONFOUND CONTROL for Face-1c (does the cyclic curl-rho survive projection changes?).
The 48 LoRA-SFTs are projection-INDEPENDENT, so run them ONCE, cache raw 1536-d activations, then sweep a projection grid
(dim x seed) in a cheap Phase B. Cyclic finding is real only if curl-rho >= 0.30 PERSISTS across the grid; else it was a 12-d artifact.
Same machinery as face1c_llm_lambda (causal-LM LoRA directed-F, mid-layer, victim probes). SMOKE=1 = tiny end-to-end check.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / os.environ.get("OUTNAME", "face1c_control.json"); OUT.parent.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
SMOKE = os.environ.get("SMOKE", "0") == "1"
NCAP = 3 if SMOKE else int(os.environ.get("NCAP", "16"))
SEEDS = [0, 1] if SMOKE else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
STEPS = 60 if SMOKE else int(os.environ.get("STEPS", "200"))
PER = 20 if SMOKE else int(os.environ.get("PER", "80"))
LAYER = int(os.environ.get("LAYER", "14")); LORA_R = 16; LORA_ALPHA = 64; LR = float(os.environ.get("LR", "5e-4")); BS = 16; MAXLEN = 64
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj".split(",")
PROJDIMS = [8, 12] if SMOKE else [int(x) for x in os.environ.get("PROJDIMS", "8,12,24,48").split(",")]
PROJSEEDS = [0, 1] if SMOKE else [int(x) for x in os.environ.get("PROJSEEDS", "0,1,2").split(",")]
CAPS = list(range(NCAP)); NEG = list(range(60, 70)); TRIO = ["dprime", "wnorm", "spread"]; PRIMARY = "dprime"; ALL_PREDS = ["dprime", "spread", "munorm", "wnorm", "iso"]; RIDGE_LAM = 1.0

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
TOK = AutoTokenizer.from_pretrained(MODEL)
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token


def load_banks():
    ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
    texts = ds["text"]; labels = np.array(ds["label"]); rng = np.random.RandomState(0); banks = {}
    for c in set(CAPS) | set(NEG):
        idx = np.where(labels == c)[0]; rng.shuffle(idx); banks[c] = [texts[int(i)] for i in idx[:PER]]
    return {c: v for c, v in banks.items() if len(v) >= (10 if SMOKE else 40)}


@torch.no_grad()
def acts_raw(model, texts, bs=64):   # raw 1536-d mean-pooled last/mid hidden — NO projection
    model.eval(); outs = []
    for i in range(0, len(texts), bs):
        enc = TOK(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
        h = model(**enc, output_hidden_states=True).hidden_states[LAYER]; m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        outs.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def lora_ft(base, texts, steps, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(base, cfg); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR); m.train()
    rng = np.random.RandomState(seed)
    for _ in range(steps):
        idx = rng.choice(len(texts), min(BS, len(texts)), replace=False)
        enc = TOK([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        loss = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def train_head(Fa, Fn, dim, ep=150):
    h = nn.Linear(dim, 1).to(DEV); opt = torch.optim.AdamW(h.parameters(), lr=0.05, weight_decay=1e-4)
    X = torch.from_numpy(np.concatenate([Fa, Fn])).float().to(DEV); y = torch.tensor([1.]*len(Fa)+[0.]*len(Fn)).to(DEV)
    pw = torch.tensor([len(Fn)/max(len(Fa), 1)], device=DEV)
    for _ in range(ep):
        opt.zero_grad(); Fun.binary_cross_entropy_with_logits(h(X).squeeze(1), y, pos_weight=pw).backward(); opt.step()
    h.eval(); return h

def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0: return 0.5
    order = s.argsort(); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s)+1)
    return float((ranks[y == 1].sum() - npos*(npos+1)/2)/(npos*nneg))

def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).float().to(DEV)).squeeze(1).cpu().numpy(); sn = head(torch.from_numpy(Fn).float().to(DEV)).squeeze(1).cpu().numpy()
    return auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))

def hodge(F):
    n = F.shape[0]; m = ~np.eye(n, dtype=bool); K = (F-F.T)/2
    h = np.array([K[i, [j for j in range(n) if j != i]].mean() for i in range(n)])
    grad = (h[:, None]-h[None, :])*m; curl = (K-(h[:, None]-h[None, :]))*m; kf = float((K[m]**2).sum())+1e-12
    return h, curl, float((grad[m]**2).sum())/kf, float((curl[m]**2).sum())/kf

def sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return 0.0 if rx.std() < 1e-12 or ry.std() < 1e-12 else float(np.corrcoef(rx, ry)[0, 1])

def ridge_loo(X, y, lam=RIDGE_LAM):
    X = np.asarray(X, float); y = np.asarray(y, float); n, p = X.shape; pred = np.zeros(n)
    for k in range(n):
        tr = [j for j in range(n) if j != k]; Xtr, ytr = X[tr], y[tr]; mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
        b = np.linalg.solve(((Xtr-mu)/sd).T@((Xtr-mu)/sd)+lam*np.eye(p), ((Xtr-mu)/sd).T@(ytr-ytr.mean())); pred[k] = ytr.mean()+((X[k]-mu)/sd)@b
    return pred, float(1-((y-pred)**2).sum()/(((y-y.mean())**2).sum()+1e-12))


def predictors_proj(base_acts, neg_base, P, caps):
    Fneg = neg_base @ P; FA = {c: base_acts[c] @ P for c in caps}; dim = P.shape[1]
    dvec = {c: (lambda d: d/(np.linalg.norm(d)+1e-12))(FA[c].mean(0)-Fneg.mean(0)) for c in caps}
    heads = {c: train_head(FA[c], Fneg, dim) for c in caps}
    wax = {c: (lambda w: (w/np.linalg.norm(w)).astype(np.float32))(heads[c].weight.detach().cpu().numpy()[0].astype(np.float64)) for c in caps}
    dp, sp_, mn, wn = {}, {}, {}, {}
    for c in caps:
        sa = FA[c]@wax[c]; sn = Fneg@wax[c]; dp[c] = float((sa.mean()-sn.mean())/np.sqrt((sa.var()+sn.var())/2+1e-12))
        sp_[c] = float(FA[c].std(0).mean()); mn[c] = float(np.linalg.norm(FA[c].mean(0))); wn[c] = float(np.linalg.norm(heads[c].weight.detach().cpu().numpy()))
    iso = {c: float(np.mean([dvec[c]@dvec[k] for k in caps if k != c])) for c in caps}
    return {"dprime": dp, "spread": sp_, "munorm": mn, "wnorm": wn, "iso": iso}, FA, Fneg, heads


def main():
    t0 = time.time(); print(f"[CTRL] SMOKE={SMOKE} ncap={NCAP} seeds={SEEDS} steps={STEPS} layer={LAYER} grid={PROJDIMS}x{PROJSEEDS} dev={DEV}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    banks = load_banks(); caps = [c for c in CAPS if c in banks]; neg = [c for c in NEG if c in banks]; Xneg = [t for c in neg for t in banks[c]]
    print(f"[CTRL] model+{len(caps)} caps ({time.time()-t0:.0f}s); PHASE A: base acts + {len(caps)*len(SEEDS)} LoRA-SFTs...", flush=True)

    base_acts = {c: acts_raw(base, banks[c]) for c in caps}; neg_base = acts_raw(base, Xneg)
    cache = {s: {} for s in SEEDS}   # cache[seed][j] = {'caps':{i:raw_acts}, 'neg':raw_acts}
    for s in SEEDS:
        for j in caps:
            pm = lora_ft(base, banks[j], STEPS, s*100 + j)
            cache[s][j] = {"caps": {i: acts_raw(pm, banks[i]) for i in caps}, "neg": acts_raw(pm, Xneg)}
            base = pm.unload(); torch.cuda.empty_cache()
        print(f"[CTRL] seed {s} fine-tunes cached ({time.time()-t0:.0f}s)", flush=True)

    print(f"[CTRL] PHASE B: projection grid ({time.time()-t0:.0f}s)...", flush=True)
    grid = []
    for pdim in PROJDIMS:
        for pseed in PROJSEEDS:
            P = (np.random.RandomState(1000+pseed).randn(neg_base.shape[1], pdim)/np.sqrt(neg_base.shape[1])).astype(np.float32)
            preds, FA, Fneg, heads = predictors_proj(base_acts, neg_base, P, caps)
            auc0 = {c: head_auc(heads[c], FA[c], Fneg) for c in caps}; n = len(caps); hs = []; curls = []; meds = []
            for s in SEEDS:
                F = np.zeros((n, n))
                for cj, j in enumerate(caps):
                    Fneg1 = cache[s][j]["neg"] @ P
                    for ci, i in enumerate(caps):
                        F[ci, cj] = (auc0[i] - head_auc(heads[i], cache[s][j]["caps"][i] @ P, Fneg1)) if i != j else 0.0
                h, curl, gf, cf = hodge(F); hs.append(h); mm = ~np.eye(n, dtype=bool); curls.append(curl[mm]); meds.append(float(np.median(F[mm])))
            hs = np.array(hs); phi = {c: float(hs.mean(0)[i]) for i, c in enumerate(caps)}
            rel_h = float(np.mean([sp(hs[a], hs[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))])) if len(SEEDS) > 1 else float("nan")
            rel_c = float(np.mean([sp(curls[a], curls[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))])) if len(SEEDS) > 1 else float("nan")
            X = np.stack([[preds[pn][c] for c in caps] for pn in TRIO], 1); lp, r2 = ridge_loo(X, np.array([phi[c] for c in caps]))
            dpr = sp([preds["dprime"][c] for c in caps], [phi[c] for c in caps])
            row = {"projdim": pdim, "projseed": pseed, "auc0_med": round(float(np.median(list(auc0.values()))), 3),
                   "medFgt": round(float(np.mean(meds)), 3), "dominance_rho": round(rel_h, 3), "curl_rho": round(rel_c, 3),
                   "phi_loo_r2": round(r2, 3), "rho_dprime_phi": round(dpr, 3)}
            grid.append(row); print(f"  [CTRL] dim={pdim} pseed={pseed}: auc0={row['auc0_med']} medFgt={row['medFgt']} dom-rho={row['dominance_rho']:+.3f} curl-rho={row['curl_rho']:+.3f} rho(d',phi)={row['rho_dprime_phi']:+.3f}", flush=True)

    inband = [r for r in grid if 0.03 <= r["medFgt"] <= 0.55]
    curls_ib = [r["curl_rho"] for r in inband]
    frac_cyclic = float(np.mean([c >= 0.30 for c in curls_ib])) if curls_ib else float("nan")
    med_curl = float(np.median(curls_ib)) if curls_ib else float("nan")
    persists = bool(frac_cyclic >= 0.66)
    if not curls_ib: headline = "INCONCLUSIVE — no in-band grid points."
    elif persists: headline = f"CYCLES PERSIST across projections (curl-rho>=0.30 in {frac_cyclic*100:.0f}% of in-band grid, median {med_curl:+.3f}) — NOT a projection artifact."
    else: headline = f"CYCLES ARE A PROJECTION ARTIFACT (curl-rho>=0.30 in only {frac_cyclic*100:.0f}% of in-band grid, median {med_curl:+.3f}) — the LLM cyclic finding does NOT survive."
    res = {"config": {"model": MODEL, "ncap": len(caps), "seeds": SEEDS, "steps": STEPS, "layer": LAYER, "smoke": SMOKE},
           "grid": grid, "frac_cyclic_inband": round(frac_cyclic, 3) if curls_ib else None, "median_curl_inband": round(med_curl, 3) if curls_ib else None,
           "cycles_persist": persists, "headline": headline}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n[CTRL] in-band grid pts={len(inband)}/{len(grid)} | frac curl-rho>=0.30 = {frac_cyclic} | median curl-rho = {med_curl}", flush=True)
    print(f"[CTRL] HEADLINE: {headline}", flush=True)
    print(f"[CTRL] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
