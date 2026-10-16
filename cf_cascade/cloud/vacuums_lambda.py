"""VACUUMS (lane 4) — projection-strength ladder. Is the non-destructive protection (D-A) entirely OUT-of-V?
Arm A (no childhood) + arm D_K (childhood with gradient projected out of top-K task-A Fisher eigvecs) for K in {0,32,128,512}.
K=0 == arm B (no projection). Measure retention(D_K) - A at each K. FLAT across K = protection independent of how much V you
evacuate = lives out-of-V (the room). FALLS as K grows = some protection lives in V.
CONFOUND BAKED IN (same species as the toggle death): projecting out more shrinks the childhood update, so D-A could fall from
"did less childhood" not "V matters". We TRACK childhood loss-drop per K; the flat-D-A read only counts where task-B learning is
matched across K. CELL-parameterized (opt350m/pythia410m). Light: A once + 4 D-variants, N=15.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(15))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16; N_FISHER = 200
KPROJ = 512; KSET = [0, 16, 32, 64, 128, 512]   # K=0 == arm B; serves BOTH vacuums (D_K-A, out-of-V gate) AND probe-2 (B-D_k rank ablation; 16 ~ OPT K_E knot)
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt", "ck": PY_C,  "family": "PYTHIA"},
}
CELL = os.environ.get("CELL", "")
assert CELL in CELLS, f"CELL must be one of {list(CELLS)}; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/vacuums_{CELL}.json"


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
    """Vproj[n] = top-Kp eigvecs to project the childhood gradient OUT of (None => arm B, no projection)."""
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); losses = []; params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers:
                gd = params[n].grad; V = Vproj[n]
                if V.shape[1] > 0: gd.sub_((gd @ V) @ V.T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    del opt
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))


def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[VAC:{CELL}] {cfg['model']} projection-strength ladder K={KSET}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[VAC:{CELL}] w_A acc={pre:.3f}; {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    FA = fisher_taskA(model, params, carriers, tok); Vfull = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); Vfull[n] = vv[:, -KPROJ:].flip(dims=[1]).float().contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd)
    print(f"[VAC:{CELL}] V basis (top-{KPROJ}) done ({time.time()-t0:.0f}s)", flush=True)

    ho = load_jsonB("medium_subj"); ho_tr = ho[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    retA = []; retDK = {k: [] for k in KSET}; drop = {k: [] for k in KSET}
    for sd in SEEDS:
        # arm A (no childhood)
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        retA.append(eval_sst(model, tok, sst_eval, posid, negid))
        # arm D_K for each projection strength
        for k in KSET:
            model.load_state_dict(ckpt_sd)
            Vk = None if k == 0 else {n: Vfull[n][:, :k] for n in carriers}
            l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=Vk, carriers=carriers)
            train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
            retDK[k].append(eval_sst(model, tok, sst_eval, posid, negid)); drop[k].append(l0 - l1)
        if sd % 5 == 0 or sd == SEEDS[-1]:
            print(f"[VAC:{CELL}] seed {sd}: A={retA[-1]:.3f} | " +
                  " ".join(f"D{k}={retDK[k][-1]:.3f}" for k in KSET) + f" ({time.time()-t0:.0f}s)", flush=True)

    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5)]
    A = np.array(retA)
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "pre_sst": pre, "n_carriers": len(carriers),
           "n_seeds": len(SEEDS), "KSET": KSET, "retA": me(A),
           "DK_minus_A": {str(k): me((np.array(retDK[k]) - A).tolist()) for k in KSET},
           "BD_at_k": {str(k): float(np.mean(retDK[0]) - np.mean(retDK[k])) for k in KSET if k != 0},  # probe-2: load carried by top-k deposit (B - D_k)
           "retDK": {str(k): me(retDK[k]) for k in KSET},
           "childhood_lossdrop": {str(k): me(drop[k]) for k in KSET}}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[VAC:{CELL}] === VACUUMS (D_K - A; flat across K = out-of-V; check lossdrop matched) ===", flush=True)
    for k in KSET:
        dm = rep["DK_minus_A"][str(k)]; ld = rep["childhood_lossdrop"][str(k)]
        bd = rep["BD_at_k"].get(str(k), 0.0)
        print(f"  K={k:4d}  D-A={dm[0]:+.3f}±{dm[1]:.3f}  B-D_k={bd:+.3f}  childhood_lossdrop={ld[0]:.2f}", flush=True)
    print(f"[VAC:{CELL}] saved vacuums_{CELL}.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
