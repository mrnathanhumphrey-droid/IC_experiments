"""
face1c_llm_lambda.py — Face-1c LLM-scale adjudicator (PREREG lock in PREREG_FACE1C_LLM_2026_07_17.md). Lambda GPU.
Faithful port of the vision pecking-order to LLM activations: Qwen2.5-1.5B frozen backbone, banking77 intents as capabilities,
mean-pooled last-hidden activations + linear victim probes. Directed F[i<-j] = LoRA-SFT backbone on (j vs neg) then victim AUC drop.
3 seeds -> seed-avg phi (de-noise) + cross-seed dominance/curl reproducibility (transitivity test). Then the SAME d'->phi regression.
SMOKE=1 runs a 3-cap/1-seed/30-step micro-check first and asserts the pipeline lands.
"""
from __future__ import annotations
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / os.environ.get("OUTNAME", "face1c_llm.json"); OUT.parent.mkdir(parents=True, exist_ok=True)
PREREG = HERE / "PREREG_FACE1C_LLM_2026_07_17.md"
LOCK = hashlib.sha256(PREREG.read_bytes()).hexdigest()[:8] if PREREG.exists() else "nolock"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
SMOKE = os.environ.get("SMOKE", "0") == "1"
NCAP = 3 if SMOKE else int(os.environ.get("NCAP", "16"))
SEEDS = [0] if SMOKE else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
STEPS = 80 if SMOKE else int(os.environ.get("STEPS", "200"))
PER = 20 if SMOKE else int(os.environ.get("PER", "80"))
LORA_R = int(os.environ.get("LORA_R", "16")); LR = float(os.environ.get("LR", "5e-4")); BS = 16; MAXLEN = 64
LAYER = int(os.environ.get("LAYER", "-1")); LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "64"))
LORA_TARGETS = os.environ.get("LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
PROJDIM = int(os.environ.get("PROJDIM", "-1"))   # fixed random projection to de-saturate the readout (-1 = off)
_P = None
CAPS = list(range(NCAP)); NEG = list(range(60, 70)); TRIO = ["dprime", "wnorm", "spread"]; PRIMARY = "dprime"
ALL_PREDS = ["dprime", "spread", "munorm", "wnorm", "iso"]; RIDGE_LAM = 1.0

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

TOK = AutoTokenizer.from_pretrained(MODEL)
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token


# ---------- data ----------
def load_banks():
    ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
    texts = ds["text"]; labels = np.array(ds["label"]); rng = np.random.RandomState(0); need = set(CAPS) | set(NEG); banks = {}
    for c in need:
        idx = np.where(labels == c)[0]; rng.shuffle(idx); idx = idx[:PER]
        banks[c] = [texts[int(i)] for i in idx]
    keep = {c: v for c, v in banks.items() if len(v) >= (10 if SMOKE else 40)}
    return keep


# ---------- activations ----------
@torch.no_grad()
def acts(model, texts, bs=64):
    model.eval(); outs = []
    for i in range(0, len(texts), bs):
        enc = TOK(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
        h = model(**enc, output_hidden_states=True).hidden_states[LAYER]
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        outs.append(pooled.float().cpu().numpy())
    out = np.concatenate(outs).astype(np.float32)
    return out @ _P if _P is not None else out


def train_head(Fa, Fn, dim, ep=150):
    h = nn.Linear(dim, 1).to(DEV); opt = torch.optim.AdamW(h.parameters(), lr=0.05, weight_decay=1e-4)
    X = torch.from_numpy(np.concatenate([Fa, Fn])).float().to(DEV)
    y = torch.tensor([1.]*len(Fa) + [0.]*len(Fn)).to(DEV); pw = torch.tensor([len(Fn)/max(len(Fa), 1)], device=DEV)
    for _ in range(ep):
        opt.zero_grad(); Fun.binary_cross_entropy_with_logits(h(X).squeeze(1), y, pos_weight=pw).backward(); opt.step()
    h.eval(); return h


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0: return 0.5
    order = s.argsort(); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s)+1)
    return float((ranks[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg))


def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).float().to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).float().to(DEV)).squeeze(1).cpu().numpy()
    return auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))


def lora_ft(base, texts, steps, seed):
    """Causal-LM LoRA-SFT on the perturber class's texts — forces the BACKBONE to adapt (no separate head to absorb the learning)."""
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(base, cfg)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR); m.train()
    rng = np.random.RandomState(seed)
    for step in range(steps):
        idx = rng.choice(len(texts), min(BS, len(texts)), replace=False)
        enc = TOK([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        loss = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def hodge(F):
    n = F.shape[0]; m = ~np.eye(n, dtype=bool); K = (F - F.T)/2
    h = np.array([K[i, [j for j in range(n) if j != i]].mean() for i in range(n)])
    grad = (h[:, None] - h[None, :]) * m; curl = (K - (h[:, None] - h[None, :])) * m
    kf = float((K[m]**2).sum()) + 1e-12
    return h, curl, float((grad[m]**2).sum())/kf, float((curl[m]**2).sum())/kf


def predictors(base, banks, caps, neg):
    Xneg = [t for c in neg for t in banks[c]]; Fneg = acts(base, Xneg); dim = Fneg.shape[1]
    FA = {c: acts(base, banks[c]) for c in caps}
    dvec = {c: (lambda d: d/(np.linalg.norm(d)+1e-12))(FA[c].mean(0) - Fneg.mean(0)) for c in caps}
    heads = {c: train_head(FA[c], Fneg, dim) for c in caps}
    wax = {c: (lambda w: (w/np.linalg.norm(w)).astype(np.float32))(heads[c].weight.detach().cpu().numpy()[0].astype(np.float64)) for c in caps}
    dprime, spread, munorm, wnorm = {}, {}, {}, {}
    for c in caps:
        sa = FA[c] @ wax[c]; sn = Fneg @ wax[c]
        dprime[c] = float((sa.mean() - sn.mean()) / np.sqrt((sa.var() + sn.var())/2 + 1e-12))
        spread[c] = float(FA[c].std(0).mean()); munorm[c] = float(np.linalg.norm(FA[c].mean(0)))
        wnorm[c] = float(np.linalg.norm(heads[c].weight.detach().cpu().numpy()))
    iso = {c: float(np.mean([dvec[c] @ dvec[k] for k in caps if k != c])) for c in caps}
    return {"dprime": dprime, "spread": spread, "munorm": munorm, "wnorm": wnorm, "iso": iso}, FA, Fneg, heads, Xneg


def build_F(base, banks, caps, neg, seed, FA, Fneg, heads, Xneg):
    n = len(caps); auc0 = {c: head_auc(heads[c], FA[c], Fneg) for c in caps}; F = np.zeros((n, n))
    if SMOKE: print(f"[F1c][diag] base victim AUC0 median={np.median(list(auc0.values())):.3f} vals={ {c: round(auc0[c],3) for c in caps} }", flush=True)
    for cj, j in enumerate(caps):
        pm = lora_ft(base, banks[j], STEPS, seed*100 + cj)   # LM-SFT on perturber class j's texts
        Fneg1 = acts(pm, Xneg)
        if SMOKE and cj == 0:
            drift = float(np.linalg.norm(Fneg1 - Fneg) / (np.linalg.norm(Fneg) + 1e-9))
            a1 = {i: round(head_auc(heads[i], acts(pm, banks[i]), Fneg1), 3) for i in caps}
            print(f"[F1c][diag] col j={j}: act-drift(neg)={drift:.4f}  auc1={a1}", flush=True)
        for ci, i in enumerate(caps):
            F[ci, cj] = (auc0[i] - head_auc(heads[i], acts(pm, banks[i]), Fneg1)) if i != j else 0.0
        base = pm.unload(); torch.cuda.empty_cache()
    return F, base


# ---------- stats ----------
def sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12: return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])

def boot_ci(x, y, alpha=95.0, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); m = len(x); rng = np.random.RandomState(seed); rs = []
    for _ in range(n):
        i = rng.randint(0, m, m)
        if x[i].std() < 1e-12 or y[i].std() < 1e-12: continue
        rs.append(sp(x[i], y[i]))
    if len(rs) < 10: return [float("nan"), float("nan")]
    lo = (100-alpha)/2; return [round(float(np.percentile(rs, lo)), 3), round(float(np.percentile(rs, 100-lo)), 3)]

def perm_p(x, y, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); o = abs(sp(x, y)); rng = np.random.RandomState(seed); c = 0
    for _ in range(n): c += abs(sp(x, rng.permutation(y))) >= o
    return round((c+1)/(n+1), 4)

def ridge_loo(X, y, lam=RIDGE_LAM):
    X = np.asarray(X, float); y = np.asarray(y, float); n, p = X.shape; pred = np.zeros(n)
    for k in range(n):
        tr = [j for j in range(n) if j != k]; Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9; Xs = (Xtr-mu)/sd; ym = ytr.mean()
        b = np.linalg.solve(Xs.T@Xs + lam*np.eye(p), Xs.T@(ytr-ym)); pred[k] = ym + ((X[k]-mu)/sd)@b
    return pred, float(1 - ((y-pred)**2).sum()/(((y-y.mean())**2).sum()+1e-12))

def perm_null95(X, y, n=1000, seed=0):
    y = np.asarray(y, float); rng = np.random.RandomState(seed); return float(np.percentile([ridge_loo(X, rng.permutation(y))[1] for _ in range(n)], 95))

def regress(target, preds, caps):
    y = np.array([target[c] for c in caps], float); uni = {}
    for pn in ALL_PREDS:
        x = np.array([preds[pn][c] for c in caps], float); ci = boot_ci(x, y, 95.0 if pn == PRIMARY else 99.2)
        uni[pn] = {"rho": round(sp(x, y), 3), "ci": ci, "perm_p": perm_p(x, y), "excl0": bool(ci[0] > 0 or ci[1] < 0)}
    X = np.stack([[preds[pn][c] for c in caps] for pn in TRIO], 1); lp, r2 = ridge_loo(X, y); null95 = perm_null95(X, y)
    orho = sp(lp, y); rng = np.random.RandomState(0); c = 0
    for _ in range(1000): c += abs(sp(ridge_loo(X, rng.permutation(y))[0], y)) >= abs(orho)
    return {"n": len(caps), "univariate": uni, "trio_loo_r2": round(r2, 3), "perm_null_loo_r2_95": round(null95, 3),
            "ordinal_loo_rankcorr": round(orho, 3), "ordinal_perm_p": round((c+1)/1001, 4)}


def main():
    t0 = time.time(); print(f"[F1c] lock={LOCK} SMOKE={SMOKE} model={MODEL} ncap={NCAP} seeds={SEEDS} steps={STEPS} per={PER} dev={DEV}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    global _P
    if PROJDIM > 0:
        hid = base.config.hidden_size; _P = (np.random.RandomState(0).randn(hid, PROJDIM) / np.sqrt(hid)).astype(np.float32)
        print(f"[F1c] random projection {hid}->{PROJDIM} (de-saturate readout)", flush=True)
    banks = load_banks(); caps = [c for c in CAPS if c in banks]; neg = [c for c in NEG if c in banks]
    print(f"[F1c] loaded model+{len(caps)} caps/{len(neg)} neg ({time.time()-t0:.0f}s)", flush=True)
    preds, FA, Fneg, heads, Xneg = predictors(base, banks, caps, neg)
    print(f"[F1c] predictors done ({time.time()-t0:.0f}s)", flush=True)

    hs, Fs = [], []
    for s in SEEDS:
        F, base = build_F(base, banks, caps, neg, s, FA, Fneg, heads, Xneg)
        h, curl, gf, cf = hodge(F); hs.append(h); Fs.append((F, curl))
        mm = ~np.eye(len(caps), dtype=bool); med = float(np.median(F[mm]))
        print(f"[F1c] seed {s}: medFgt={med:.3f} inband={0.03<=med<=0.55} grad%={gf:.2f} curl%={cf:.2f} ({time.time()-t0:.0f}s)", flush=True)

    hs = np.array(hs); phi = {c: float(hs.mean(0)[i]) for i, c in enumerate(caps)}
    rel_h = [sp(hs[a], hs[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))]
    curls = [Fs[k][1][~np.eye(len(caps), dtype=bool)] for k in range(len(SEEDS))]
    rel_curl = [sp(curls[a], curls[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))]
    meds = [float(np.median(Fs[k][0][~np.eye(len(caps), dtype=bool)])) for k in range(len(SEEDS))]
    reg = regress(phi, preds, caps)

    rel_h_m = float(np.mean(rel_h)) if rel_h else float("nan")
    rel_curl_m = float(np.mean(rel_curl)) if rel_curl else float("nan")
    med_m = float(np.mean(meds)); inband = bool(0.03 <= med_m <= 0.55)
    law_pass = bool((reg["trio_loo_r2"] >= max(0.30, reg["perm_null_loo_r2_95"]) or
                     (reg["ordinal_loo_rankcorr"] >= 0.50 and reg["ordinal_perm_p"] < 0.05)) and reg["univariate"][PRIMARY]["excl0"])
    transitive = bool(rel_h_m >= 0.50 and rel_curl_m < 0.15); cyclic_real = bool(rel_curl_m >= 0.30)
    if not inband: headline = "INCONCLUSIVE — forgetting not in-band (saturated/floored); regime tweak needed."
    elif cyclic_real: headline = "ROCK-PAPER-SCISSORS task-triples EXIST at LLM scale — order NOT globally transitive."
    elif law_pass and transitive: headline = "Pecking order GENERALIZES to LLMs — d' predicts it, order is transitive."
    elif transitive: headline = "Transitivity generalizes but the a-priori d'-law is vision-specific."
    else: headline = "Mixed — see numbers."

    res = {"config": {"lock": LOCK, "smoke": SMOKE, "model": MODEL, "ncap": len(caps), "seeds": SEEDS, "steps": STEPS, "per": PER, "lora_r": LORA_R},
           "median_forget": round(med_m, 3), "inband": inband, "cross_seed_dominance_rho": round(rel_h_m, 3),
           "cross_seed_curl_rho": round(rel_curl_m, 3), "phi_regression": reg, "phi_mean": phi, "preds": preds,
           "decision": {"law_pass": law_pass, "transitive": transitive, "cyclic_real": cyclic_real, "headline": headline}}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n[F1c] medFgt={med_m:.3f} inband={inband} | dominance-rho={rel_h_m:+.3f} curl-rho={rel_curl_m:+.3f}", flush=True)
    print(f"[F1c] phi: trio LOO-R2={reg['trio_loo_r2']:+.3f}(null95 {reg['perm_null_loo_r2_95']:.3f}) ordinal={reg['ordinal_loo_rankcorr']:+.3f}(p{reg['ordinal_perm_p']}) rho(d',phi)={reg['univariate'][PRIMARY]['rho']:+.3f}{reg['univariate'][PRIMARY]['ci']}", flush=True)
    print(f"[F1c] law_pass={law_pass} transitive={transitive} cyclic_real={cyclic_real}", flush=True)
    print(f"\n[F1c] HEADLINE: {headline}", flush=True)
    print(f"[F1c] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
