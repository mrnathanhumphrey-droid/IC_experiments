"""BASIS-INVARIANCE GATE (option a) — is the childhood deposit's out-of-V drain REAL geometry or a fixed-rank artifact?
V=top-128 is a SHRINKING fraction of a growing carrier dim (768->2048), so a fixed-rank basis captures less BY CONSTRUCTION.
This recomputes the deposit's mapfrac under SCALE-INVARIANT basis definitions + a chance-corrected ENRICHMENT, across the
size ladder. Decisive read: ENRICHMENT = mapfrac / (K/d). If enrichment(K=128) is FLAT across scale -> raw mapfrac drop was
mechanical K/d. If enrichment DROPS with scale -> the deposit genuinely leaves the curvature subspace (real orthogonal residual).
Also reports energy-rank K_E (dims holding X% Fisher mass) — does curvature stay low-rank or spread with scale?
B-arm only (childhood deposit), 3 seeds, no D/held-out/retention = light. CELL-parameterized like bd_feed. Big cells -> 40GB fix.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(3))
STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16; N_FISHER = 200
KS = [32, 64, 128, 256]; EFRACS = [0.5, 0.9, 0.95]; KCAP = 256  # cap stored eigvecs: full U[col,col] OOMs 1.3b; max(KS) & all K_E fit under this
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt125m":    {"model": "facebook/opt-125m",     "ckpt": "opt125m_sst2_wA.pt",    "ck": OPT_C, "family": "OPT",    "size_m": 125},
    "opt350m":    {"model": "facebook/opt-350m",     "ckpt": "opt350m_sst2_wA.pt",    "ck": OPT_C, "family": "OPT",    "size_m": 350},
    "opt1_3b":    {"model": "facebook/opt-1.3b",     "ckpt": "opt1_3b_sst2_wA.pt",    "ck": OPT_C, "family": "OPT",    "size_m": 1300, "big": True},
    "pythia160m": {"model": "EleutherAI/pythia-160m", "ckpt": "pythia160m_sst2_wA.pt", "ck": PY_C,  "family": "PYTHIA", "size_m": 160},
    "pythia410m": {"model": "EleutherAI/pythia-410m", "ckpt": "pythia410m_sst2_subsetmin.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 410},
    "pythia1b":   {"model": "EleutherAI/pythia-1b",   "ckpt": "pythia1b_sst2_wA.pt",   "ck": PY_C,  "family": "PYTHIA", "size_m": 1000, "big": True},
    "pythia160m_step8k":  {"model": "EleutherAI/pythia-160m", "ckpt": "pythia160m_step8k_sst2_wA.pt",  "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia160m_step16k": {"model": "EleutherAI/pythia-160m", "ckpt": "pythia160m_step16k_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia160m_step50k": {"model": "EleutherAI/pythia-160m", "ckpt": "pythia160m_step50k_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia160m_dedup":   {"model": "EleutherAI/pythia-160m-deduped", "ckpt": "pythia160m_dedup_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia410m_dedup":   {"model": "EleutherAI/pythia-410m-deduped", "ckpt": "pythia410m_dedup_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 410},
}
CELL = os.environ.get("CELL", "")
assert CELL in CELLS, f"CELL must be one of {list(CELLS)}; got {CELL!r}"
cfg = CELLS[CELL]; BIG = bool(cfg.get("big"))
OUT = f"{G}/basisinv_{CELL}.json"


def _retry(fn, tries=5, wait=12):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e: last = e; print(f"[hf] retry {i+1}: {str(e)[:100]}", flush=True); time.sleep(wait)
    raise last


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def win(ds, k=900, w=4000):
    n = min(len(ds), w); return list(ds.shuffle(seed=0).select(range(n)))[: k]


def load_hf(key):
    from datasets import load_dataset
    LD = lambda *a, **k: _retry(lambda: load_dataset(*a, **k))
    if key == "emotion":
        return [(f"{r['text'].strip()}\nEmotion:", " " + ["sadness","joy","love","anger","fear","surprise"][r["label"]]) for r in win(LD("dair-ai/emotion")["train"])]
    if key == "dbpedia":
        L = ["company","school","artist","athlete","politics","transport","building","nature","village","animal","plant","album","film","book"]
        return [(f"{r['content'].strip()[:400]}\nCategory:", " " + L[r["label"]]) for r in win(LD("fancyzhx/dbpedia_14")["train"])]
    if key == "yahoo":
        L = ["society","science","health","education","computers","sports","business","entertainment","family","politics"]
        return [(f"{(r.get('question_title') or '').strip()}\nTopic:", " " + L[r["topic"]]) for r in win(LD("community-datasets/yahoo_answers_topics")["train"]) if r.get("question_title")]
    if key == "newscat":
        return [(f"{(r.get('headline') or '').strip()}\nCategory:", " " + (r.get("category") or "").strip().lower()) for r in win(LD("heegyu/news-category-dataset")["train"]) if r.get("headline") and r.get("category")]
    raise ValueError(key)


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = _retry(lambda: load_dataset("stanfordnlp/sst2"))[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def fisher_taskA(model, params, names, tok):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float32) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().float(); F[n] += g.T @ g
    return F


def childhood(model, tok, stream, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=(False if BIG else None)); g = torch.Generator().manual_seed(seed)
    tasks = list(stream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()


def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[BI:{CELL}] {cfg['model']} basis-invariance (deposit mapfrac vs basis def)", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    if BIG: model.gradient_checkpointing_enable(); model.config.use_cache = False; print(f"[BI:{CELL}] grad-checkpoint ON", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    ckpt_sd = {k: (v.detach().clone().cpu() if BIG else v.detach().clone()) for k, v in model.state_dict().items()}
    print(f"[BI:{CELL}] {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    # Fisher -> per-carrier eigenvalues (desc) + eigvecs (desc); energy-rank; dim
    FA = fisher_taskA(model, params, carriers, tok)
    EV = {}; U = {}; dim = {}; KE = {n: {} for n in carriers}
    for n in carriers:
        ev, vv = torch.linalg.eigh(FA[n].double()); ev = ev.clamp(min=0)
        order = torch.argsort(ev, descending=True)
        EV[n] = ev[order]; U[n] = vv[:, order[:KCAP]].contiguous(); dim[n] = ev.numel()  # top-KCAP eigvecs only (mem); EV kept full for K_E
        cum = torch.cumsum(EV[n], 0) / EV[n].sum().clamp(min=1e-30)
        for e in EFRACS: KE[n][e] = int((cum < e).sum().item()) + 1
        del FA[n]
    model.load_state_dict(ckpt_sd)
    print(f"[BI:{CELL}] Fisher/eig done ({time.time()-t0:.0f}s)", flush=True)

    ho = None  # childhood stream (same as bd_feed)
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    # accumulate deposit-energy spectra over seeds: colenergy[n] = sum_seed (dw @ U)^2 over rows, per eig-column
    colE = {n: torch.zeros(min(KCAP, dim[n]), dtype=torch.float64, device=dev) for n in carriers}
    totE = {n: 0.0 for n in carriers}
    for sd in SEEDS:
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        p = dict(model.named_parameters())
        with torch.no_grad():
            for n in carriers:
                dw = (p[n].detach() - ckpt_sd[n].to(dev)).double()          # [row, col]
                c = dw @ U[n]                                                # [row, col] coords in eig-basis (desc)
                colE[n] += (c * c).sum(0); totE[n] += float((dw * dw).sum())
        print(f"[BI:{CELL}] seed {sd} deposit done ({time.time()-t0:.0f}s)", flush=True)

    # per-carrier mapfrac@K (cumulative energy), mapfrac@E (top-K_E), enrichment = mapfrac@K / (K/d)
    rec = {"mapfrac_K": {k: [] for k in KS}, "enrich_K": {k: [] for k in KS},
           "mapfrac_E": {e: [] for e in EFRACS}, "KE": {e: [] for e in EFRACS},
           "KE_over_d": {e: [] for e in EFRACS}, "dim": []}
    for n in carriers:
        ce = colE[n] / max(totE[n], 1e-30); cum = torch.cumsum(ce, 0)        # cumulative deposit-energy by eig-rank (top-KCAP)
        d = dim[n]; nc = cum.numel(); rec["dim"].append(d)
        for k in KS:
            kk = min(k, d, nc); mf = float(cum[kk - 1]); rec["mapfrac_K"][k].append(mf)
            rec["enrich_K"][k].append(mf / (kk / d))
        for e in EFRACS:
            ke = min(KE[n][e], d, nc); rec["mapfrac_E"][e].append(float(cum[ke - 1]))
            rec["KE"][e].append(ke); rec["KE_over_d"][e].append(ke / d)

    def med(x): return float(np.median(x))
    summ = {"cell": CELL, "family": cfg["family"], "size_m": cfg["size_m"], "n_carriers": len(carriers),
            "dim_med": med(rec["dim"]),
            "mapfrac_K": {str(k): med(rec["mapfrac_K"][k]) for k in KS},
            "enrich_K": {str(k): med(rec["enrich_K"][k]) for k in KS},
            "mapfrac_E": {str(e): med(rec["mapfrac_E"][e]) for e in EFRACS},
            "KE": {str(e): med(rec["KE"][e]) for e in EFRACS},
            "KE_over_d": {str(e): med(rec["KE_over_d"][e]) for e in EFRACS}}
    json.dump(summ, open(OUT, "w"), indent=2)
    print(f"\n[BI:{CELL}] === BASIS-INVARIANCE ===", flush=True)
    print(f"  dim~{summ['dim_med']:.0f} | mapfrac@128={summ['mapfrac_K']['128']:.3f} chance(128/d)={128/summ['dim_med']:.3f} "
          f"ENRICH@128={summ['enrich_K']['128']:.2f}x", flush=True)
    print(f"  mapfrac@E90={summ['mapfrac_E']['0.9']:.3f} | K_E90={summ['KE']['0.9']:.0f} (KE90/d={summ['KE_over_d']['0.9']:.3f})", flush=True)
    print(f"[BI:{CELL}] saved basisinv_{CELL}.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
