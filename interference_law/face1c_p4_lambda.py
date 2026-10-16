"""
face1c_p4_lambda.py — P4 CROSS-SUBSTRATE REPLICATION of the Face-1c LLM cyclic (rock-paper-scissors) forgetting finding.
Addresses the last open critique (#3): the whole result rests on n=1 model (Qwen2.5-1.5B) + n=1 dataset (banking77).
Does the reproducible cyclic curl-rho survive a DIFFERENT ARCHITECTURE and a DIFFERENT, more diverse TASK SUITE?

Design = a 2x2-minus-one factorial over (model, dataset), run as CELLS in ONE instance (one teardown), each cell = the
projection-sweep persistence test from face1c_control_lambda (cache 48 fine-tunes once, sweep a projection grid; cyclic
only if curl-rho>=0.30 PERSISTS across projections). arch-agnostic LoRA via target_modules='all-linear'; LAYER auto=nlayers//2.

CELLS (default): pythia1.4b:clinc150 (BOTH new = the money cell), pythia1.4b:banking77 (isolates architecture),
                 qwen:clinc150 (isolates dataset).
Pre-committed: REPLICATES if the BOTH-NEW cell persists (curl-rho>=0.30 in >=66% in-band); GENERAL if all three do.
If the both-new cell fails -> the cyclic finding is Qwen/banking77-specific (honest downgrade).
SMOKE=1 = 1 tiny cell end-to-end (loaders + new arch + fine-tune moves + Hodge) before any real spend.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / os.environ.get("OUTNAME", "face1c_p4.json"); OUT.parent.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE", "0") == "1"

MODEL_MAP = {
    "qwen": "Qwen/Qwen2.5-1.5B",
    "pythia1.4b": "EleutherAI/pythia-1.4b",
    "pythia410m": "EleutherAI/pythia-410m",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "gemma2-2b": "google/gemma-2-2b",
}
DEFAULT_CELLS = "pythia1.4b:clinc150" if SMOKE else "pythia1.4b:clinc150,pythia1.4b:banking77,qwen:clinc150"
CELLS = [c.strip() for c in os.environ.get("CELLS", DEFAULT_CELLS).split(",") if c.strip()]

NCAP = 3 if SMOKE else int(os.environ.get("NCAP", "16"))
SEEDS = [0, 1] if SMOKE else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
STEPS = 40 if SMOKE else int(os.environ.get("STEPS", "200"))
PER = 20 if SMOKE else int(os.environ.get("PER", "80"))
LORA_R = 16; LORA_ALPHA = 64; LR = float(os.environ.get("LR", "5e-4")); BS = 16; MAXLEN = 64
LORA_TARGETS = os.environ.get("LORA_TARGETS", "all-linear")  # arch-agnostic across Qwen / NeoX / Llama
PROJDIMS = [12] if SMOKE else [int(x) for x in os.environ.get("PROJDIMS", "8,12,24,48").split(",")]
PROJSEEDS = [0] if SMOKE else [int(x) for x in os.environ.get("PROJSEEDS", "0,1,2").split(",")]
LAYER_ENV = os.environ.get("LAYER", "")  # blank -> auto nlayers//2
CAPS = list(range(NCAP)); NEG = list(range(60, 70))
CYC_BAR = 0.30; PERSIST_FRAC = 0.66  # match the control's pre-committed bars

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def load_banks(dsname, tok):
    if dsname == "banking77":
        ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
        texts = ds["text"]; labels = np.array(ds["label"])
    elif dsname == "clinc150":
        ds = load_dataset("clinc_oos", "plus", split="train", trust_remote_code=True)
        lc = "intent" if "intent" in ds.column_names else "label"
        texts = ds["text"]; labels = np.array(ds[lc])
    else:
        raise ValueError(f"unknown dataset {dsname}")
    rng = np.random.RandomState(0); banks = {}
    for c in set(CAPS) | set(NEG):
        idx = np.where(labels == c)[0]; rng.shuffle(idx); banks[c] = [texts[int(i)] for i in idx[:PER]]
    return {c: v for c, v in banks.items() if len(v) >= (10 if SMOKE else 40)}


@torch.no_grad()
def acts_raw(model, tok, layer, texts, bs=64):
    model.eval(); outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
        h = model(**enc, output_hidden_states=True).hidden_states[layer]; m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        outs.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def lora_ft(base, tok, texts, steps, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(base, cfg); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR); m.train()
    rng = np.random.RandomState(seed)
    for _ in range(steps):
        idx = rng.choice(len(texts), min(BS, len(texts)), replace=False)
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
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


def dprime_of(FA, Fneg, heads, caps):
    dp = {}
    for c in caps:
        w = heads[c].weight.detach().cpu().numpy()[0].astype(np.float64); w = w/(np.linalg.norm(w)+1e-12)
        sa = FA[c]@w; sn = Fneg@w; dp[c] = float((sa.mean()-sn.mean())/np.sqrt((sa.var()+sn.var())/2+1e-12))
    return dp


def run_cell(cellstr, t0):
    mkey, dsname = cellstr.split(":")
    model_id = MODEL_MAP.get(mkey, mkey)
    print(f"\n[P4] === CELL {cellstr}  model={model_id} data={dsname} ({time.time()-t0:.0f}s) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(DEV); base.eval()
    for p in base.parameters(): p.requires_grad_(False)
    nlayers = int(base.config.num_hidden_layers); layer = int(LAYER_ENV) if LAYER_ENV else nlayers // 2
    banks = load_banks(dsname, tok); caps = [c for c in CAPS if c in banks]; neg = [c for c in NEG if c in banks]
    Xneg = [t for c in neg for t in banks[c]]
    print(f"[P4] {cellstr}: {len(caps)} caps, {len(neg)} neg, nlayers={nlayers} -> LAYER={layer} ({time.time()-t0:.0f}s)", flush=True)

    base_acts = {c: acts_raw(base, tok, layer, banks[c]) for c in caps}; neg_base = acts_raw(base, tok, layer, Xneg)
    cache = {s: {} for s in SEEDS}
    for s in SEEDS:
        for j in caps:
            pm = lora_ft(base, tok, banks[j], STEPS, s*100 + j)
            cache[s][j] = {"caps": {i: acts_raw(pm, tok, layer, banks[i]) for i in caps}, "neg": acts_raw(pm, tok, layer, Xneg)}
            base = pm.unload(); torch.cuda.empty_cache()
        print(f"[P4] {cellstr}: seed {s} fine-tunes cached ({time.time()-t0:.0f}s)", flush=True)

    grid = []; hdim = neg_base.shape[1]
    for pdim in PROJDIMS:
        for pseed in PROJSEEDS:
            P = (np.random.RandomState(1000+pseed).randn(hdim, pdim)/np.sqrt(hdim)).astype(np.float32)
            FA = {c: base_acts[c] @ P for c in caps}; Fneg = neg_base @ P
            heads = {c: train_head(FA[c], Fneg, pdim) for c in caps}; auc0 = {c: head_auc(heads[c], FA[c], Fneg) for c in caps}
            dp = dprime_of(FA, Fneg, heads, caps)
            n = len(caps); mm = ~np.eye(n, dtype=bool); hs = []; curls = []; meds = []
            for s in SEEDS:
                F = np.zeros((n, n))
                for cj, j in enumerate(caps):
                    Fneg1 = cache[s][j]["neg"] @ P
                    for ci, i in enumerate(caps):
                        F[ci, cj] = (auc0[i] - head_auc(heads[i], cache[s][j]["caps"][i] @ P, Fneg1)) if i != j else 0.0
                h, curl, gf, cf = hodge(F); hs.append(h); curls.append(curl[mm]); meds.append(float(np.median(F[mm])))
            hs = np.array(hs); phi = hs.mean(0)
            rel_h = float(np.mean([sp(hs[a], hs[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))])) if len(SEEDS) > 1 else float("nan")
            rel_c = float(np.mean([sp(curls[a], curls[b]) for a in range(len(SEEDS)) for b in range(a+1, len(SEEDS))])) if len(SEEDS) > 1 else float("nan")
            dpr = sp([dp[c] for c in caps], list(phi))
            row = {"projdim": pdim, "projseed": pseed, "auc0_med": round(float(np.median(list(auc0.values()))), 3),
                   "medFgt": round(float(np.mean(meds)), 3), "dominance_rho": round(rel_h, 3), "curl_rho": round(rel_c, 3),
                   "rho_dprime_phi": round(dpr, 3)}
            grid.append(row)
            print(f"  [P4] {cellstr} dim={pdim} ps={pseed}: auc0={row['auc0_med']} medFgt={row['medFgt']:+.3f} dom-rho={row['dominance_rho']:+.3f} curl-rho={row['curl_rho']:+.3f} rho(d',phi)={row['rho_dprime_phi']:+.3f}", flush=True)

    inband = [r for r in grid if 0.03 <= r["medFgt"] <= 0.55]
    curls_ib = [r["curl_rho"] for r in inband]
    frac = float(np.mean([c >= CYC_BAR for c in curls_ib])) if curls_ib else None
    med_curl = float(np.median(curls_ib)) if curls_ib else None
    med_dom = float(np.median([r["dominance_rho"] for r in inband])) if inband else None
    persists = bool(frac is not None and frac >= PERSIST_FRAC)
    del base; torch.cuda.empty_cache()
    return {"cell": cellstr, "model": model_id, "dataset": dsname, "layer": layer, "ncap": len(caps),
            "grid": grid, "n_inband": len(inband), "frac_cyclic_inband": round(frac, 3) if frac is not None else None,
            "median_curl_inband": round(med_curl, 3) if med_curl is not None else None,
            "median_dom_inband": round(med_dom, 3) if med_dom is not None else None, "persists": persists}


def combined_verdict(cells):
    def is_both_new(c): return (not c["model"].startswith("Qwen")) and (c["dataset"] != "banking77")
    both = [c for c in cells if is_both_new(c)]
    money = both[0] if both else None
    all_persist = all(c["persists"] for c in cells) and len(cells) > 0
    if money is None:
        return "INCOMPLETE — no both-new (new-arch x new-data) cell present."
    if not money["persists"]:
        return (f"DOWNGRADE — cyclic finding is Qwen/banking77-SPECIFIC. Both-new cell {money['cell']} "
                f"curl-rho median {money['median_curl_inband']} in {money['frac_cyclic_inband']} of in-band (bar {PERSIST_FRAC}). Does NOT generalize.")
    if all_persist:
        return (f"REPLICATES-GENERAL — cyclic forgetting reproduces across architecture AND dataset. "
                f"All {len(cells)} cells persist; both-new {money['cell']} curl-rho median {money['median_curl_inband']}.")
    fails = [c["cell"] for c in cells if not c["persists"]]
    return (f"REPLICATES (partial) — both-new cell {money['cell']} PERSISTS (curl-rho {money['median_curl_inband']}), "
            f"cyclic finding generalizes; but these cells did not clear the bar: {fails}.")


def main():
    t0 = time.time()
    print(f"[P4] SMOKE={SMOKE} cells={CELLS} ncap={NCAP} seeds={SEEDS} steps={STEPS} grid={PROJDIMS}x{PROJSEEDS} targets={LORA_TARGETS} dev={DEV}", flush=True)
    results = []
    for cellstr in CELLS:
        try:
            res = run_cell(cellstr, t0)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"cell": cellstr, "error": str(e)[:300], "persists": False, "model": cellstr.split(":")[0], "dataset": cellstr.split(":")[-1]}
        results.append(res)
        headline = combined_verdict([r for r in results if "grid" in r])
        OUT.write_text(json.dumps({"config": {"cells": CELLS, "seeds": SEEDS, "steps": STEPS, "targets": LORA_TARGETS, "smoke": SMOKE},
                                   "cells": results, "headline_sofar": headline}, indent=1))
        print(f"[P4] cell {cellstr} done, persists={res.get('persists')} curl={res.get('median_curl_inband')} ({time.time()-t0:.0f}s); JSON updated", flush=True)

    headline = combined_verdict([r for r in results if "grid" in r])
    final = json.loads(OUT.read_text()); final["headline"] = headline; OUT.write_text(json.dumps(final, indent=1))
    print("\n[P4] === PER-CELL ===", flush=True)
    for r in results:
        if "grid" in r:
            print(f"  {r['cell']:24}: curl-rho={r['median_curl_inband']} dom-rho={r['median_dom_inband']} frac={r['frac_cyclic_inband']} persists={r['persists']} (n_inband={r['n_inband']})", flush=True)
        else:
            print(f"  {r['cell']:24}: ERROR {r.get('error')}", flush=True)
    print(f"\n[P4] HEADLINE: {headline}", flush=True)
    print(f"[P4] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
