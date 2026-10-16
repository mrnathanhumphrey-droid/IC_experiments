"""RECIPE × SEVERITY-MATCHED — the designed open test for the dedup lean.
The dedup B-D nudge (+0.027/160m, +0.021/410m) is confounded: both dedup cells also forget ~3pp less, and B-D tracks A_forget.
This SPLITS them: run std vs dedup across a held-out-intensity ladder (STEPS_DEPTH = the A_forget knob), N=40 seeds, so we read
B-D at MATCHED A_forget by comparing the (A_forget, B-D) curves at equal A_forget instead of at one regime.
  - dedup ABOVE std at equal A_forget -> dedup genuinely engages the map (severity-controlled). REAL.
  - curves coincide -> the lean was the severity confound. NULL confirmed.
Deposits (childhood B/D) computed ONCE per seed; only the held-out retrain varies across the ladder (cheap). PY_C carriers, none big.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(40))
LADDER = [180, 250, 350]      # held-out STEPS_DEPTH = the A_forget knob (matched-severity axis)
LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "pythia160m":       {"model": "EleutherAI/pythia-160m",         "ckpt": "pythia160m_sst2_wA.pt",       "ck": PY_C, "family": "PYTHIA", "size_m": 160, "recipe": "std"},
    "pythia160m_dedup": {"model": "EleutherAI/pythia-160m-deduped", "ckpt": "pythia160m_dedup_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160, "recipe": "dedup"},
    "pythia410m":       {"model": "EleutherAI/pythia-410m",         "ckpt": "pythia410m_sst2_subsetmin.pt","ck": PY_C, "family": "PYTHIA", "size_m": 410, "recipe": "std"},
    "pythia410m_dedup": {"model": "EleutherAI/pythia-410m-deduped", "ckpt": "pythia410m_dedup_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 410, "recipe": "dedup"},
}
CELL = os.environ.get("CELL", "")
assert CELL in CELLS, f"CELL must be one of {list(CELLS)}; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/recipematched_{CELL}.json"


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


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt


def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_((gd @ Vproj[n]) @ Vproj[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt


def snap(model, carriers): p = dict(model.named_parameters()); return {n: p[n].detach().clone() for n in carriers}
def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[RM:{CELL}] {cfg['model']} ({cfg['recipe']}) recipe×severity-matched, ladder={LADDER} N={len(SEEDS)}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[RM:{CELL}] w_A SST acc={pre_sst:.3f}; {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).float().contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd)
    print(f"[RM:{CELL}] V basis done ({time.time()-t0:.0f}s)", flush=True)

    ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    # ret[sdep][arm] = list over seeds
    ret = {sd: {a: [] for a in ["A", "B", "D"]} for sd in LADDER}
    for seed in SEEDS:
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, seed); wB = snap(model, carriers)
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, seed, Vproj=V, carriers=carriers); wD = snap(model, carriers)
        starts = {"A": ckpt_sd, "B": {**ckpt_sd, **wB}, "D": {**ckpt_sd, **wD}}
        for sdep in LADDER:
            for a in ["A", "B", "D"]:
                model.load_state_dict(starts[a]); train_single(model, tok, ho_tr, sdep, LR_DEPTH, seed)
                ret[sdep][a].append(eval_sst(model, tok, sst_eval, posid, negid))
        if seed % 5 == 0 or seed == SEEDS[-1]:
            s = LADDER[1]; print(f"[RM:{CELL}] seed {seed}: @{s} A/B/D={ret[s]['A'][-1]:.3f}/{ret[s]['B'][-1]:.3f}/{ret[s]['D'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5)]
    curve = {}
    for sdep in LADDER:
        A = np.array(ret[sdep]["A"]); B = np.array(ret[sdep]["B"]); D = np.array(ret[sdep]["D"])
        curve[str(sdep)] = {"A_forget": float(pre_sst - A.mean()), "B_minus_A": me((B - A).tolist()),
                            "B_minus_D": me((B - D).tolist()), "D_minus_A": me((D - A).tolist()),
                            "retA": me(A), "retB": me(B), "retD": me(D)}
    rep = {"cell": CELL, "model": cfg["model"], "recipe": cfg["recipe"], "size_m": cfg["size_m"],
           "pre_sst": pre_sst, "n_seeds": len(SEEDS), "ladder": LADDER, "curve": curve}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[RM:{CELL}] === (A_forget, B-D) CURVE [{cfg['recipe']}] ===", flush=True)
    for sdep in LADDER:
        c = curve[str(sdep)]; print(f"  steps={sdep:4d}  A_forget={c['A_forget']:+.3f}  B-D={c['B_minus_D'][0]:+.3f}±{c['B_minus_D'][1]:.3f}  B-A={c['B_minus_A'][0]:+.3f}", flush=True)
    print(f"[RM:{CELL}] saved recipematched_{CELL}.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
