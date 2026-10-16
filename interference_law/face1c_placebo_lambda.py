"""
face1c_placebo_lambda.py — P1 PLACEBO / NULL CONTROL for the Face-1c LLM cyclic finding (does curl reproduce WITHOUT task-specific forgetting?).
Three conditions on ONE base / projection / caps / seeds:
  REAL     = LoRA-SFT on perturber class j's texts (task-specific)      <- the real experiment
  RANDTEXT = LoRA-SFT on a task-AGNOSTIC random text mix per column      <- "any real training" null
  NOISE    = isotropic Gaussian weight-noise, ||Δθ|| MATCHED per-column to REAL  <- "any movement of this magnitude" null
For each: build directed F, Hodge-decompose, cross-seed curl-ρ + dominance-ρ. DECISIVE: is real curl-ρ clearly ABOVE both placebos?
If a placebo reproduces curl-ρ >= 0.30 -> the cycles are a pipeline artifact (retract). If placebos ~0 while real ~0.37 -> genuine.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / os.environ.get("OUTNAME", "face1c_placebo.json"); OUT.parent.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
SMOKE = os.environ.get("SMOKE", "0") == "1"
NCAP = 3 if SMOKE else int(os.environ.get("NCAP", "16"))
SEEDS = [0, 1] if SMOKE else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
STEPS = 40 if SMOKE else int(os.environ.get("STEPS", "200"))
PER = 20 if SMOKE else int(os.environ.get("PER", "80"))
LAYER = int(os.environ.get("LAYER", "14")); LORA_R = 16; LORA_ALPHA = 64; LR = float(os.environ.get("LR", "5e-4")); BS = 16; MAXLEN = 64
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj".split(",")
PROJDIMS = [12] if SMOKE else [int(x) for x in os.environ.get("PROJDIMS", "12,24").split(",")]
PROJSEEDS = [0] if SMOKE else [int(x) for x in os.environ.get("PROJSEEDS", "0,1").split(",")]
CAPS = list(range(NCAP)); NEG = list(range(60, 70))
CONDS = ["real", "randtext", "noise"]

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
def acts_raw(model, texts, bs=64):
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


def lora_delta_norm(m):
    tot = 0.0
    for mod in m.modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            try:
                A = mod.lora_A["default"].weight; B = mod.lora_B["default"].weight; sc = mod.scaling["default"]
                tot += float((((B @ A) * sc).float()**2).sum())
            except Exception: pass
    return tot ** 0.5


def matched_noise(base, target_norm, seed):
    g = torch.Generator(device=DEV).manual_seed(int(seed))
    tgt = [(n, p) for n, p in base.named_parameters() if ("weight" in n) and any(t in n for t in LORA_TARGETS)]
    z = {n: torch.randn(p.shape, generator=g, device=DEV).to(p.dtype) for n, p in tgt}
    tot = sum(float((zz.float()**2).sum()) for zz in z.values()) ** 0.5
    scale = target_norm / (tot + 1e-9); added = {n: z[n]*scale for n in z}
    with torch.no_grad():
        d = dict(base.named_parameters())
        for n, zz in added.items(): d[n].add_(zz)
    return added

def restore(base, added):
    with torch.no_grad():
        d = dict(base.named_parameters())
        for n, zz in added.items(): d[n].sub_(zz)


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
    return float((ranks[y == 1].sum()-npos*(npos+1)/2)/(npos*nneg))

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


def build_F_from_cache(cache_cj, caps, base_acts, neg_base, P, auc0, heads):
    n = len(caps); F = np.zeros((n, n))
    for cj, j in enumerate(caps):
        Fneg1 = cache_cj[j]["neg"] @ P
        for ci, i in enumerate(caps):
            F[ci, cj] = (auc0[i] - head_auc(heads[i], cache_cj[j]["caps"][i] @ P, Fneg1)) if i != j else 0.0
    return F


def main():
    t0 = time.time(); print(f"[PBO] SMOKE={SMOKE} ncap={NCAP} seeds={SEEDS} steps={STEPS} conds={CONDS} grid={PROJDIMS}x{PROJSEEDS} dev={DEV}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    banks = load_banks(); caps = [c for c in CAPS if c in banks]; neg = [c for c in NEG if c in banks]
    Xneg = [t for c in neg for t in banks[c]]; ALLTXT = [t for c in caps+neg for t in banks[c]]
    base_acts = {c: acts_raw(base, banks[c]) for c in caps}; neg_base = acts_raw(base, Xneg)
    print(f"[PBO] model+{len(caps)} caps ({time.time()-t0:.0f}s); PHASE A: {len(caps)*len(SEEDS)} x (REAL+RANDTEXT SFT + NOISE)...", flush=True)

    def ext(model): return {"caps": {i: acts_raw(model, banks[i]) for i in caps}, "neg": acts_raw(model, Xneg)}
    cache = {cond: {s: {} for s in SEEDS} for cond in CONDS}
    for s in SEEDS:
        rng = np.random.RandomState(1234 + s)
        for cj, j in enumerate(caps):
            pm = lora_ft(base, banks[j], STEPS, s*100 + cj); dn = lora_delta_norm(pm)
            cache["real"][s][j] = ext(pm); base = pm.unload(); torch.cuda.empty_cache()
            added = matched_noise(base, dn, seed=s*1000 + cj + 77); cache["noise"][s][j] = ext(base); restore(base, added); torch.cuda.empty_cache()
            rtext = [ALLTXT[k] for k in rng.choice(len(ALLTXT), min(PER, len(ALLTXT)), replace=False)]
            pm2 = lora_ft(base, rtext, STEPS, s*100 + cj + 5000); cache["randtext"][s][j] = ext(pm2); base = pm2.unload(); torch.cuda.empty_cache()
        print(f"[PBO] seed {s} done, delta_norm~{dn:.2f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[PBO] PHASE B: projections {PROJDIMS}x{PROJSEEDS} x {len(CONDS)} conds ({time.time()-t0:.0f}s)...", flush=True)
    per_cond = {c: {"curl": [], "dom": [], "med": []} for c in CONDS}
    rows = []
    for pdim in PROJDIMS:
        for pseed in PROJSEEDS:
            P = (np.random.RandomState(1000+pseed).randn(neg_base.shape[1], pdim)/np.sqrt(neg_base.shape[1])).astype(np.float32)
            FA = {c: base_acts[c] @ P for c in caps}; Fneg = neg_base @ P
            heads = {c: train_head(FA[c], Fneg, pdim) for c in caps}; auc0 = {c: head_auc(heads[c], FA[c], Fneg) for c in caps}
            n = len(caps); mm = ~np.eye(n, dtype=bool)
            for cond in CONDS:
                hs, curls, meds = [], [], []
                for s in SEEDS:
                    F = build_F_from_cache(cache[cond][s], caps, base_acts, neg_base, P, auc0, heads)
                    h, curl, gf, cf = hodge(F); hs.append(h); curls.append(curl[mm]); meds.append(float(np.median(F[mm])))
                hs = np.array(hs)
                rel_h = float(np.mean([sp(hs[a], hs[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))]))
                rel_c = float(np.mean([sp(curls[a], curls[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))]))
                med = float(np.mean(meds))
                rows.append({"cond": cond, "projdim": pdim, "projseed": pseed, "auc0": round(float(np.median(list(auc0.values()))), 3),
                             "medFgt": round(med, 3), "dom_rho": round(rel_h, 3), "curl_rho": round(rel_c, 3)})
                if 0.03 <= med <= 0.55:
                    per_cond[cond]["curl"].append(rel_c); per_cond[cond]["dom"].append(rel_h); per_cond[cond]["med"].append(med)
                print(f"  [PBO] {cond:8} dim={pdim} ps={pseed}: auc0={rows[-1]['auc0']} medFgt={med:+.3f} dom-rho={rel_h:+.3f} curl-rho={rel_c:+.3f}", flush=True)

    summ = {c: {"median_curl_rho": round(float(np.median(per_cond[c]["curl"])), 3) if per_cond[c]["curl"] else None,
                "median_dom_rho": round(float(np.median(per_cond[c]["dom"])), 3) if per_cond[c]["dom"] else None,
                "median_medFgt": round(float(np.median(per_cond[c]["med"])), 3) if per_cond[c]["med"] else None,
                "n_inband": len(per_cond[c]["curl"])} for c in CONDS}
    real_curl = summ["real"]["median_curl_rho"]; plc = [summ["randtext"]["median_curl_rho"], summ["noise"]["median_curl_rho"]]
    plc_max = max([x for x in plc if x is not None], default=None)
    if real_curl is None or plc_max is None: headline = "INCONCLUSIVE — missing in-band points."
    elif any(x is not None and x >= 0.30 for x in plc): headline = f"ARTIFACT — a placebo reproduces curl-rho>=0.30 (randtext {summ['randtext']['median_curl_rho']}, noise {summ['noise']['median_curl_rho']}) vs real {real_curl}. Cyclic finding does NOT survive."
    elif real_curl >= 0.30 and (real_curl - plc_max) >= 0.15: headline = f"GENUINE — real curl-rho {real_curl} clears placebos (max {plc_max}) by >=0.15. Cycles are task-specific interference, NOT a pipeline/projection artifact."
    else: headline = f"AMBIGUOUS — real {real_curl} vs placebo-max {plc_max}, separation < 0.15."
    res = {"config": {"model": MODEL, "ncap": len(caps), "seeds": SEEDS, "steps": STEPS, "layer": LAYER, "smoke": SMOKE},
           "summary": summ, "rows": rows, "headline": headline}
    OUT.write_text(json.dumps(res, indent=1))
    print("\n[PBO] === curl-rho by condition (in-band median) ===", flush=True)
    for c in CONDS: print(f"  {c:8}: curl-rho={summ[c]['median_curl_rho']}  dom-rho={summ[c]['median_dom_rho']}  medFgt={summ[c]['median_medFgt']}  (n_inband={summ[c]['n_inband']})", flush=True)
    print(f"\n[PBO] HEADLINE: {headline}", flush=True)
    print(f"[PBO] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
