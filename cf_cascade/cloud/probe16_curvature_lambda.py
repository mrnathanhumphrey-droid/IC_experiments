"""PROBE 16 (C1 curvature disambiguation) — does BLOOM's erosion reduce to the KNOWN flatness story (C3), or is it a genuinely
directional/conjunction effect (C1, novel)? Lit-check: C3 (forget ≈ ½ λ_max ‖Δθ‖²; flatter minima forget less) is KNOWN
(Mirzadeh 2020). If BLOOM erodes because (a) its task-A minimum is intrinsically SHARPER, or (b) its childhood SHARPENS the
minimum while protectors' FLATTENS it, then C1 collapses into C3. This probe measures the task-A Fisher SPECTRUM at w_A and after
childhood, per cell, to test that. PREREG_PROBE16_CURVATURE_DISAMBIGUATION.

Measures (scale-robust, cross-arch-comparable): per carrier eigen-spectrum of the task-A empirical Fisher F_A (same F as
probe13 V), aggregated to:
  - anisotropy  A  = mean_n( λ_max_n / trace_n )              [sharp/anisotropic minimum -> HIGH A]
  - eff_rank    ER = mean_n( trace_n^2 / Σ_k ev_{n,k}^2 )     [flat/isotropic -> HIGH ER; sharp -> LOW ER]
  - lam_max_raw = mean_n( λ_max_n )                           [scale-dependent; report with caveat]
BASELINE at w_A (deterministic). POST-CHILDHOOD at DOSES × NSEEDS -> fractional change (within-cell, scale-robust). Anchor gate
reused (VOID if w_A off). Incremental JSON dump per (dose,seed). NO alttask, NO interference-direction measurement (that's
probe13); this is the SCALAR/SPECTRAL curvature axis only.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
NSEEDS = int(os.environ.get("NSEEDS", "8")); SEEDS = list(range(NSEEDS))
DOSES = [float(x) for x in os.environ.get("DOSES", "1e-6,7e-6").split(",")]
STEPS_CHILD = 400; BATCH = 16; N_FISHER = 200; HEARTBEAT = 40; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753, "BLOOM": 0.83}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
BLOOM_C = ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
FALCON_C = ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",           "ck": OPT_C,   "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt", "ck": PY_C,    "family": "PYTHIA"},
    "bloom560m":  {"model": "bigscience/bloom-560m",   "ckpt": "bloom560m_sst2_wA.pt",         "ck": BLOOM_C, "family": "BLOOM"},
    "xglm564m":   {"model": "facebook/xglm-564M",      "ckpt": "xglm564m_sst2_wA.pt",          "ck": OPT_C,   "family": "XGLM"},
    "falcon_rw1b":{"model": "tiiuae/falcon-rw-1b",     "ckpt": "falcon_rw1b_sst2_wA.pt",       "ck": FALCON_C,"family": "FALCON"},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe16_curvature_{CELL}.json"


def _retry(fn, tries=5, wait=12):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e: last = e; print(f"[hf] retry {i+1}: {str(e)[:80]}", flush=True); time.sleep(wait)
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
def eval_sst(model, tok, evalset, posid, negid):
    model.eval(); correct = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(evalset), 32):
            chunk = evalset[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:256] for s, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pred = (last[:, posid] > last[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(chunk): correct += int(pred[j] == lab); n += 1
    return correct / max(n, 1)
def childhood(model, tok, stream, steps, lr, seed, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} childhood {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache()
def labels_of(items): return sorted(set(c for _, c in items))
def me(x): return [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)] if len(x) > 1 else [float(np.mean(x)) if x else 0.0, 0.0]


def spectrum(model, params, carriers, tok):
    """Task-A Fisher spectrum -> scale-robust sharpness aggregates. anisotropy A=mean(lmax/trace) [sharp->high];
    eff_rank ER=mean(trace^2/sum ev^2) [flat->high]; lam_max_raw=mean(lmax) [scale-dependent]."""
    F = fisher_taskA(model, params, carriers, tok)
    anis, er, lmx, tr = [], [], [], []
    for n in carriers:
        ev = torch.linalg.eigvalsh(F[n].double()).clamp_min(0.0)  # symmetric PSD; clamp tiny negatives
        s = float(ev.sum()); lm = float(ev[-1]); s2 = float((ev * ev).sum())
        if s > 1e-30:
            anis.append(lm / s); er.append((s * s) / (s2 + 1e-30)); lmx.append(lm); tr.append(s)
        del ev, F[n]
    return {"anisotropy": float(np.mean(anis)), "eff_rank": float(np.mean(er)),
            "lam_max_raw": float(np.mean(lmx)), "trace_mean": float(np.mean(tr))}


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P16:{CELL}] {cfg['model']} curvature spectrum; DOSES={DOSES} N={NSEEDS}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    _env_anchor = os.environ.get("ANCHOR", "")
    exp = float(_env_anchor) if _env_anchor else ANCHOR_WA.get(cfg["family"], 0.0)
    print(f"[P16:{CELL}] w_A acc={pre_sst:.3f} (exp {exp}); {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)
    if abs(pre_sst - exp) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f} vs {exp}"}, open(OUT, "w"), indent=2)
        print(f"[P16:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    base = spectrum(model, params, carriers, tok)
    print(f"[P16:{CELL}] BASELINE(w_A): anis={base['anisotropy']:.4f} eff_rank={base['eff_rank']:.2f} "
          f"lam_max={base['lam_max_raw']:.3e} ({time.time()-t0:.0f}s)", flush=True)
    try:
        stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
        stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}", "baseline": base}, open(OUT, "w"), indent=2)
        print(f"[P16:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return

    D = len(DOSES); acc = [{"anisotropy": [], "eff_rank": [], "lam_max_raw": [], "retA": [], "dA_frac": [], "dER_frac": []} for _ in range(D)]
    for si, sd in enumerate(SEEDS):
        for di, lr in enumerate(DOSES):
            model.load_state_dict(ckpt_sd)
            childhood(model, tok, stream, STEPS_CHILD, lr, sd, hb=(f"s{si+1}/{NSEEDS} d{di+1}/{D}({lr:.0e})" if si % 4 == 0 else ""))
            retA = eval_sst(model, tok, sst_eval, posid, negid)
            p2 = dict(model.named_parameters()); sp = spectrum(model, p2, carriers, tok)
            a = acc[di]; a["anisotropy"].append(sp["anisotropy"]); a["eff_rank"].append(sp["eff_rank"]); a["lam_max_raw"].append(sp["lam_max_raw"]); a["retA"].append(retA)
            a["dA_frac"].append((sp["anisotropy"] - base["anisotropy"]) / (base["anisotropy"] + 1e-30))
            a["dER_frac"].append((sp["eff_rank"] - base["eff_rank"]) / (base["eff_rank"] + 1e-30))
        rows = []
        for di, lr in enumerate(DOSES):
            a = acc[di]
            rows.append({"lr": lr, "n": len(a["retA"]), "retA": me(a["retA"]),
                         "anisotropy": me(a["anisotropy"]), "eff_rank": me(a["eff_rank"]), "lam_max_raw": me(a["lam_max_raw"]),
                         "dAnisotropy_frac": me(a["dA_frac"]), "dEffRank_frac": me(a["dER_frac"])})
        json.dump({"cell": CELL, "family": cfg["family"], "VOID": False, "w_A_acc": pre_sst, "n_seeds_done": si+1, "n_target": NSEEDS,
                   "doses": DOSES, "baseline_wA": base, "rows": rows}, open(OUT, "w"), indent=2)
        if si % 2 == 0 or si == NSEEDS-1:
            summ = " ".join(f"{r['lr']:.0e}:dA={r['dAnisotropy_frac'][0]:+.3f}/dER={r['dEffRank_frac'][0]:+.3f}@retA{r['retA'][0]:.3f}" for r in rows)
            print(f"[seed] s{si+1}/{NSEEDS} {summ} ({time.time()-_T0:.0f}s)", flush=True)
    print(f"[P16:{CELL}] DONE saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
