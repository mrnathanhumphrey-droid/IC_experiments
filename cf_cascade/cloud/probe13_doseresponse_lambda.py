"""PROBE 13 (Hunt 3) — dose-response of directional protection. PREREG_PROBE13_DOSERESPONSE (LOCK 49d14b48).
Is the cross-cell N_phase2 spread a DOSE artifact or a real architecture difference? Sweep childhood lr per cell; measure
N_phase2 (protection of the realized interference dir) + N_random (matched-mag control) vs childhood-effect retA_drop.
NO alttask (Hunt-1: S is a threat-magnitude artifact), NO calibration (dose SET via env DOSES). NoCH arm (anchor=w_A) is
dose-independent -> computed ONCE/seed, shared across doses. Incremental JSON dump per seed (crash-resilient).
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
NSEEDS = int(os.environ.get("NSEEDS", "50")); SEEDS = list(range(NSEEDS))
DOSES = [float(x) for x in os.environ.get("DOSES", "1e-6,3e-6,1e-5,3e-5").split(",")]
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; TOPK = 16; N_FISHER = 200; K_RAND = 3
HEARTBEAT = 40; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753, "BLOOM": 0.83}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
BLOOM_C = ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
FALCON_C = ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]  # Falcon-rw = ALiBi+English; same Megatron scheme as BLOOM for direct comparability
JAIS_C = ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"]  # 2026-07-26 BREAK-N=1: Jais GPT2-style Conv1D carriers (QKV + SwiGLU up c_fc/c_fc2 + MLP down c_proj); ndim==2 -> fisher_taskA handles orientation-agnostically
QWEN_C = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]  # RoPE+multi (deferred)
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C,   "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,    "family": "PYTHIA"},
    "bloom560m":  {"model": "bigscience/bloom-560m",   "ckpt": "bloom560m_sst2_wA.pt",          "ck": BLOOM_C, "family": "BLOOM"},
    "opt13b":     {"model": "facebook/opt-1.3b",        "ckpt": "opt13b_sst2_wA.pt",            "ck": OPT_C,   "family": "OPT", "anchor": 0.93},
    "xglm564m":   {"model": "facebook/xglm-564M",       "ckpt": "xglm564m_sst2_wA.pt",          "ck": OPT_C,   "family": "XGLM"},  # OPT-named submodules -> OPT carrier list drops in; anchor via ANCHOR env from buildwa
    "falcon_rw1b":{"model": "tiiuae/falcon-rw-1b",       "ckpt": "falcon_rw1b_sst2_wA.pt",       "ck": FALCON_C,"family": "FALCON"},  # ALiBi+English; BLOOM-scheme carriers; anchor via ANCHOR env from buildwa
    "bloom560m_noln":{"model": "bigscience/bloom-560m",  "ckpt": "bloom560m_noln_sst2_wA.pt",    "ck": BLOOM_C, "family": "BLOOM", "ablate_embln": True},  # post-embed LN removed; conjunction-vs-LN fork; anchor via ANCHOR env
    "jais590m":   {"model": "inceptionai/jais-family-590m", "ckpt": "jais590m_sst2_wA.pt",        "ck": JAIS_C,  "family": "JAIS"},   # BREAK-N=1: 2nd ALiBi+multilingual family (Conv1D carriers); anchor via ANCHOR env
    "qwen2_1p5b": {"model": "Qwen/Qwen2-1.5B",              "ckpt": "qwen2_1p5b_sst2_wA.pt",       "ck": QWEN_C,  "family": "QWEN"},   # RoPE+multi empty-cell fill (deferred; A100 for 8960-dim Fisher); anchor via ANCHOR env
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe13_doseresponse_{CELL}.json"


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
def train_phase(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()
def labels_of(items): return sorted(set(c for _, c in items))
def me(x): return [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)] if len(x) > 1 else [float(np.mean(x)) if x else 0.0, 0.0]


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P13:{CELL}] {cfg['model']} dose-response N={NSEEDS} DOSES={DOSES}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32, trust_remote_code=True).to(dev)
    if cfg.get("ablate_embln"):  # match the noln-trained ckpt structure (word_embeddings_layernorm -> Identity, no params)
        import torch.nn as nn; model.transformer.word_embeddings_layernorm = nn.Identity()
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    _env_anchor = os.environ.get("ANCHOR", "")   # buildwa-measured anchor for cells not in ANCHOR_WA (e.g. XGLM)
    exp = float(_env_anchor) if _env_anchor else cfg.get("anchor", ANCHOR_WA.get(cfg["family"], 0.0))
    print(f"[P13:{CELL}] w_A acc={pre_sst:.3f} (exp {exp}); {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)
    if abs(pre_sst - exp) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f} vs {exp}"}, open(OUT, "w"), indent=2)
        print(f"[P13:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    try:
        ho_tr = load_jsonB("medium_subj")[:-200]; assert len(ho_tr) > 100
        stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
        stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P13:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return
    print(f"[P13:{CELL}] datasets loaded (phase-2 subj {len(ho_tr)}; childhood {len(stream)})", flush=True)
    V16 = {n: V[n][:, :TOPK].contiguous() for n in carriers}

    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def perturb_eval(anchor_sd, fn):
        model.load_state_dict(anchor_sd)
        with torch.no_grad():
            p = dict(model.named_parameters())
            for n in carriers: p[n].add_(fn(n).to(p[n].dtype))
        return eval_sst(model, tok, sst_eval, posid, negid)
    def measure_np(anchor_sd, sd):
        """From anchor: realize phase-2 -> in-V dir dVp/magp. dp2=acc0-acc(anchor+dVp); drk=mean random@magp. Restores anchor."""
        model.load_state_dict(anchor_sd); acc0 = eval_sst(model, tok, sst_eval, posid, negid)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd)
        cur = dict(model.named_parameters()); dVp = {}; magp = {}
        with torch.no_grad():
            for n in carriers:
                d = (cur[n].detach().double() - anchor_sd[n].to(dev).double()); dv = (d @ V[n]) @ V[n].T; dVp[n] = dv; magp[n] = float(dv.norm())
        dp2 = acc0 - perturb_eval(anchor_sd, lambda n: dVp[n])
        drk = []
        for k in range(K_RAND):
            gen = torch.Generator(device="cpu").manual_seed(100000 * sd + 777 * k + 13)
            raws = {n: torch.randn(dict(model.named_parameters())[n].shape, generator=gen).double().to(dev) for n in carriers}
            def rfn(n, raws=raws):
                rv = (raws[n] @ V16[n]) @ V16[n].T; return rv / (float(rv.norm()) + 1e-30) * magp[n]
            drk.append(acc0 - perturb_eval(anchor_sd, rfn))
        model.load_state_dict(anchor_sd)
        return float(acc0), float(dp2), float(np.mean(drk))

    D = len(DOSES)
    acc = [{"dp2_n": [], "dp2_c": [], "drk_n": [], "drk_c": [], "retA": [], "fracV": []} for _ in range(D)]
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        acc0_n, dp2_n, drk_n = measure_np(ckpt_sd, sd)   # NoCH once (dose-independent)
        for di, lr in enumerate(DOSES):
            model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, lr, sd,
                                                      hb=(f"s{si+1}/{ns} d{di+1}/{D}({lr:.0e})" if si % 10 == 0 else ""))
            ch_sd = snap(); retA = eval_sst(model, tok, sst_eval, posid, negid)
            # MECHANISM (read-only, does not perturb measurement): energy-weighted fraction of the childhood
            # weight displacement (ch - w_A) that lands inside the task-A curvature subspace V. Tests the
            # readout hypothesis: protectors reorganize INSIDE V (high frac); BLOOM reorganizes generically (low).
            with torch.no_grad():
                num = 0.0; den = 0.0
                for n in carriers:
                    dch = ch_sd[n].to(dev).double() - ckpt_sd[n].to(dev).double()
                    proj = (dch @ V[n]) @ V[n].T
                    num += float((proj * proj).sum()); den += float((dch * dch).sum())
            frac_inV = num / (den + 1e-30)
            _, dp2_c, drk_c = measure_np(ch_sd, sd)
            a = acc[di]; a["dp2_n"].append(dp2_n); a["dp2_c"].append(dp2_c); a["drk_n"].append(drk_n); a["drk_c"].append(drk_c); a["retA"].append(retA); a["fracV"].append(frac_inV)
        # incremental dump (crash-resilient)
        rows = []
        for di, lr in enumerate(DOSES):
            a = acc[di]; arr = np.array
            Np = me((arr(a["dp2_n"]) - arr(a["dp2_c"])).tolist()); Nr = me((arr(a["drk_n"]) - arr(a["drk_c"])).tolist())
            rt = me(a["retA"]); rows.append({"lr": lr, "n": len(a["retA"]), "retA": rt, "retA_drop": [pre_sst - rt[0], rt[1]],
                                             "N_phase2": Np, "N_random": Nr, "N_conc": [Np[0]-Nr[0], (Np[1]**2+Nr[1]**2)**0.5],
                                             "frac_child_in_V": me(a["fracV"])})
        json.dump({"cell": CELL, "family": cfg["family"], "VOID": False, "w_A_acc": pre_sst, "n_seeds_done": si+1,
                   "n_target": ns, "doses": DOSES, "rows": rows}, open(OUT, "w"), indent=2)
        if si % 5 == 0 or si == ns-1:
            summ = " ".join(f"{r['lr']:.0e}:Np={r['N_phase2'][0]:+.3f}@drop{r['retA_drop'][0]:.3f}" for r in rows)
            print(f"[seed] s{si+1}/{ns} {summ} ({time.time()-_T0:.0f}s)", flush=True)
    print(f"[P13:{CELL}] DONE saved {OUT} wall={time.time()-t0:.0f}s", flush=True)
    for r in rows:
        print(f"[P13:{CELL}] lr={r['lr']:.0e} retA_drop={r['retA_drop'][0]:+.3f} N_phase2={r['N_phase2'][0]:+.4f}±{r['N_phase2'][1]:.4f} N_conc(ph2-rnd)={r['N_conc'][0]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
