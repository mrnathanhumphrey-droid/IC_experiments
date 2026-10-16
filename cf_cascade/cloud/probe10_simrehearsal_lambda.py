"""PROBE 10 — is specificity (8v4) DRIVEN BY rehearsal-similarity? PREREG_PROBE10_SPECIFICITY_SIMILARITY_2026_06_29 (LOCK c83884a7).
8v4 found OPT AIMS childhood-protection at the interference direction; WHY is untested. Hypothesis: OPT can aim because the
childhood rehearsed that task (or a same-family task). CONSTRUCTED causal contrast (NOT a measured covariate): 4 childhoods
identical except one swapped slot (backbone {trec,subj} + {agnews|dbpedia|yahoo|emotion}); probe the realized interference
direction of T in {agnews,yahoo} against each. DV (8v4 N-estimator, magnitude-matched -> not forget-confounded):
  N(d) = dprop_NoCH(d) - dprop_CH(d);  S_T(CH) = N(T-realized-dir) - N(randtop16-dir).
NoCH anchor = w_A (SHARED across the 4 conditions -> measured once/seed). CH anchor = w_postCH(condition).
Prediction (OPT): S grades S(IDENTICAL) > S(SAME-FAMILY) > S(UNRELATED), for both T. Pythia: FLAT (null control).
fixed childhood lr via env CHILD_LR (report ret_A per condition; VOID-soft if destructive). HARD-STOP (VOID) on dataset/anchor fail.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
NSEEDS = int(os.environ.get("NSEEDS", "80")); SEEDS = list(range(NSEEDS))
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; TOPK = 16; N_FISHER = 200; K_RAND = 5
NONDESTRUCT_TOL = 0.05; EPS = 1e-6; HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02
DEFAULT_CHILD_LR = {"OPT": 1e-5, "PYTHIA": 3e-5}
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
# 4 childhoods: backbone {trec,subj} + one swapped slot. #tasks=3, steps=400 fixed (only swap differs).
CONDITIONS = {
    "CH_agnews":  ["agnews", "trec", "subj"],
    "CH_dbpedia": ["dbpedia", "trec", "subj"],
    "CH_yahoo":   ["yahoo", "trec", "subj"],
    "CH_emotion": ["emotion", "trec", "subj"],
}
PROBE_T = os.environ.get("PROBE_T", "agnews,yahoo").split(",")   # realized interference directions to probe
# rehearsal-similarity rank of each childhood to each probe T (2=identical,1=same-family topic,0=unrelated affect)
SIM = {
    "agnews": {"CH_agnews": 2, "CH_dbpedia": 1, "CH_yahoo": 1, "CH_emotion": 0},
    "yahoo":  {"CH_yahoo": 2, "CH_dbpedia": 1, "CH_agnews": 1, "CH_emotion": 0},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe10_simrehearsal_{CELL}.json"
CHILD_LR = float(os.environ.get("CHILD_LR", DEFAULT_CHILD_LR[cfg["family"]]))


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
def load_trec():
    from datasets import load_dataset
    L6 = ["abbreviation", "entity", "description", "human", "location", "number"]
    ds = _retry(lambda: load_dataset("SetFit/TREC-QC"))["train"]; cols = ds.column_names; tcol = "text" if "text" in cols else cols[0]
    for lc in ("label_coarse_text", "label_text", "coarse_label_text"):
        if lc in cols: return [(f"{str(r[tcol]).strip()}\nType:", " " + str(r[lc]).strip().lower()) for r in win(ds)]
    for lc in ("label_coarse", "coarse_label", "label-coarse", "label"):
        if lc in cols: return [(f"{str(r[tcol]).strip()}\nType:", " " + L6[int(r[lc])]) for r in win(ds) if int(r[lc]) < 6]
    raise ValueError(f"TREC-QC columns: {cols}")
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
def load_task(key):
    if key == "agnews": return load_jsonB("low_agnews")
    if key == "subj": return load_jsonB("medium_subj")[:-200]
    if key == "trec": return load_trec()
    return load_hf(key)
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
        t = tasks[step % len(tasks)]; items = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} childhood {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache()
def train_phase(model, tok, items, steps, lr, seed, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache()


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P10:{CELL}] {cfg['model']} specificity~rehearsal-similarity N={NSEEDS} probes={PROBE_T} child_lr={CHILD_LR:.1e} ({time.ctime()})", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P10:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    if abs(pre_sst - ANCHOR_WA[cfg["family"]]) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f}"}, open(OUT, "w"), indent=2)
        print(f"[P10:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    try:
        allkeys = sorted(set(k for ks in CONDITIONS.values() for k in ks) | set(PROBE_T))
        items = {k: load_task(k) for k in allkeys}
        for k, it in items.items(): assert len(it) > 50, k
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P10:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return
    streams = {c: {k: items[k] for k in ks} for c, ks in CONDITIONS.items()}
    print(f"[P10:{CELL}] datasets loaded ({ {k: len(v) for k, v in items.items()} })", flush=True)
    V16 = {n: V[n][:, :TOPK].contiguous() for n in carriers}

    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def perturb_eval(anchor_sd, fn):
        model.load_state_dict(anchor_sd)
        with torch.no_grad():
            p = dict(model.named_parameters())
            for n in carriers: p[n].add_(fn(n).to(p[n].dtype))
        return eval_sst(model, tok, sst_eval, posid, negid)

    def measure(anchor_sd, sd, T_items, hb):
        """From anchor: realize T (250 steps) -> in-V displacement dV (mag). dprop(T)=acc0-acc(anchor+dV);
        dprop(rand)=acc0-acc(anchor+randtop16 @ matched mag). Returns (dphaseT, drand). Restores anchor."""
        model.load_state_dict(anchor_sd); acc0 = eval_sst(model, tok, sst_eval, posid, negid)
        train_phase(model, tok, T_items, STEPS_P2, LR_DEPTH, sd, hb=hb)
        cur = dict(model.named_parameters()); dV = {}; mag = {}
        with torch.no_grad():
            for n in carriers:
                d = (cur[n].detach().double() - anchor_sd[n].to(dev).double()); dv = (d @ V[n]) @ V[n].T; dV[n] = dv; mag[n] = float(dv.norm())
        dT = acc0 - perturb_eval(anchor_sd, lambda n: dV[n])
        drk = []
        for k in range(K_RAND):
            gen = torch.Generator(device="cpu").manual_seed(100000 * sd + 777 * k + 13)
            raws = {n: torch.randn(dict(model.named_parameters())[n].shape, generator=gen).double().to(dev) for n in carriers}
            def rfn(n, raws=raws):
                rv = (raws[n] @ V16[n]) @ V16[n].T; return rv / (float(rv.norm()) + 1e-30) * mag[n]
            drk.append(acc0 - perturb_eval(anchor_sd, rfn))
        model.load_state_dict(anchor_sd)
        return float(dT), float(np.mean(drk))

    # accumulators: per (condition, probe T): paired arrays of dT and drand for NoCH and CH
    acc = {c: {T: {"dT_n": [], "dT_c": [], "rk_n": [], "rk_c": []} for T in PROBE_T} for c in CONDITIONS}
    retA = {c: [] for c in CONDITIONS}
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        # NoCH arm (anchor = w_A) — SHARED across conditions, computed once per seed per probe T
        noch = {}
        for T in PROBE_T:
            dT, rk = measure(ckpt_sd, sd, items[T], hb=f"s{si+1}/{ns} NoCH {T}")
            noch[T] = (dT, rk)
        # CH arms (one per condition)
        for c in CONDITIONS:
            model.load_state_dict(ckpt_sd); childhood(model, tok, streams[c], STEPS_CHILD, CHILD_LR, sd, hb=f"s{si+1}/{ns} {c}")
            ch_sd = snap(); retA[c].append(eval_sst(model, tok, sst_eval, posid, negid))
            for T in PROBE_T:
                dT, rk = measure(ch_sd, sd, items[T], hb=f"s{si+1}/{ns} {c}+{T}")
                acc[c][T]["dT_n"].append(noch[T][0]); acc[c][T]["dT_c"].append(dT)
                acc[c][T]["rk_n"].append(noch[T][1]); acc[c][T]["rk_c"].append(rk)
        # running S per condition for the primary probe
        line = []
        for T in PROBE_T:
            for c in CONDITIONS:
                d = acc[c][T]; S = float(np.mean((np.array(d["dT_n"]) - np.array(d["dT_c"])) - (np.array(d["rk_n"]) - np.array(d["rk_c"]))))
                line.append(f"{T[:3]}:{c.split('_')[1][:3]}={S:+.3f}")
        print(f"[seed] s{si+1}/{ns} retA={ {c: round(np.mean(retA[c]),3) for c in CONDITIONS} } S[{' '.join(line)}] ({time.time()-_T0:.0f}s)", flush=True)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]; a = np.array
    def rank(x):
        x = a(x, float); o = x.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r
    def spear(x, y):
        rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
        return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))
    out = {"cell": CELL, "family": cfg["family"], "VOID": False, "n_seeds": ns, "w_A_acc": pre_sst, "child_lr": CHILD_LR,
           "conditions": CONDITIONS, "probe_T": PROBE_T, "sim_rank": SIM,
           "ret_anchor_ch": {c: me(retA[c]) for c in CONDITIONS},
           "childhood_nondestructive": {c: bool(np.mean(retA[c]) >= pre_sst - NONDESTRUCT_TOL) for c in CONDITIONS},
           "results": {}, "spearman_S_vs_similarity": {}, "ident_minus_unrelated": {}}
    for T in PROBE_T:
        out["results"][T] = {}
        Svals = {}
        for c in CONDITIONS:
            d = acc[c][T]
            NT = me((a(d["dT_n"]) - a(d["dT_c"])).tolist()); Nr = me((a(d["rk_n"]) - a(d["rk_c"])).tolist())
            S = me(((a(d["dT_n"]) - a(d["dT_c"])) - (a(d["rk_n"]) - a(d["rk_c"]))).tolist())
            out["results"][T][c] = {"N_T": NT, "N_rand": Nr, "S": S, "sim": SIM[T][c], "dT_noch": me(d["dT_n"]), "dT_ch": me(d["dT_c"])}
            Svals[c] = S
        sims = [SIM[T][c] for c in CONDITIONS]; Ss = [Svals[c][0] for c in CONDITIONS]
        out["spearman_S_vs_similarity"][T] = spear(Ss, sims)
        ident = [c for c in CONDITIONS if SIM[T][c] == 2][0]; unrel = [c for c in CONDITIONS if SIM[T][c] == 0][0]
        gap = a(((a(acc[ident][T]["dT_n"]) - a(acc[ident][T]["dT_c"])) - (a(acc[ident][T]["rk_n"]) - a(acc[ident][T]["rk_c"]))) -
                ((a(acc[unrel][T]["dT_n"]) - a(acc[unrel][T]["dT_c"])) - (a(acc[unrel][T]["rk_n"]) - a(acc[unrel][T]["rk_c"]))))
        out["ident_minus_unrelated"][T] = me(gap.tolist())
    json.dump(out, open(OUT, "w"), indent=2)
    for T in PROBE_T:
        g = out["ident_minus_unrelated"][T]; sg = g[0] / max(g[1], 1e-9)
        print(f"[P10:{CELL}] probe={T} rho(S,sim)={out['spearman_S_vs_similarity'][T]:+.3f} "
              f"S[{' '.join(c.split('_')[1][:3]+'='+format(out['results'][T][c]['S'][0],'+.3f')+'(sim'+str(SIM[T][c])+')' for c in CONDITIONS)}] "
              f"ident-unrel={g[0]:+.4f}({sg:+.1f}s)", flush=True)
    print(f"[P10:{CELL}] nondestruct={out['childhood_nondestructive']} saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
