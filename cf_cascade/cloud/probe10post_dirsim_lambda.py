"""PROBE 10 post-hoc — interference-DIRECTION overlap vs task-FORMAT. PREREG_PROBE10POST_DIRSIM_2026_07_06 (LOCK 080dbe88).
Probe-10 found "same-family by format" unreliable (dbpedia a weak yahoo-sibling). Test: does aimed protection S track the
overlap of tasks' realized interference DIRECTIONS in V, not their format? Realize each task's 250-step in-V direction from
w_A over K seeds; emit the cosine overlap matrix + probe columns {agnews,yahoo}. OPT only, cheap (no childhood, no paired seeds).
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
KDIR = int(os.environ.get("KDIR", "8")); LR_DEPTH = 5e-5; STEPS_P2 = 250; BATCH = 16; KPROJ = 128; N_FISHER = 200
HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
TASKS = ["agnews", "dbpedia", "yahoo", "emotion", "trec", "subj"]
PROBES = ["agnews", "yahoo"]
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe10post_dirsim_{CELL}.json"


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
    print(f"[P10post:{CELL}] {cfg['model']} interference-direction overlap  KDIR={KDIR} tasks={TASKS} ({time.ctime()})", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P10post:{CELL}] w_A acc={pre_sst:.3f}; {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)
    if abs(pre_sst - ANCHOR_WA[cfg["family"]]) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f}"}, open(OUT, "w"), indent=2)
        print(f"[P10post:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    try:
        items = {k: load_task(k) for k in TASKS}
        for k, it in items.items(): assert len(it) > 50, k
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P10post:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return
    print(f"[P10post:{CELL}] datasets loaded ({ {k: len(v) for k, v in items.items()} })", flush=True)

    def realize_vcoord(task_items, seed):
        """From w_A: train 250 steps on task -> in-V coordinate direction (cat_carrier (d@V).reshape(-1)), unit-normalized (CPU)."""
        model.load_state_dict(ckpt_sd); train_phase(model, tok, task_items, STEPS_P2, LR_DEPTH, seed)
        cur = dict(model.named_parameters()); parts = []
        with torch.no_grad():
            for n in carriers:
                d = (cur[n].detach().double() - ckpt_sd[n].to(dev).double()); parts.append((d @ V[n]).reshape(-1))
        vec = torch.cat(parts); vec = vec / (vec.norm() + 1e-30)
        model.load_state_dict(ckpt_sd)
        return vec.float().cpu()

    meandir = {}; self_consistency = {}
    for X in TASKS:
        vs = []
        for s in range(KDIR):
            vs.append(realize_vcoord(items[X], 1000 + s))
            print(f"[hb] dir {X} {s+1}/{KDIR} ({time.time()-_T0:.0f}s)", flush=True)
        Vm = torch.stack(vs)  # (K, D) on CPU
        # self-consistency = mean pairwise cosine among the K unit dirs
        Gc = Vm @ Vm.T; iu = torch.triu_indices(KDIR, KDIR, offset=1)
        self_consistency[X] = float(Gc[iu[0], iu[1]].mean())
        md = Vm.mean(0); md = md / (md.norm() + 1e-30); meandir[X] = md
        print(f"[P10post:{CELL}] {X:8s} self_consistency={self_consistency[X]:+.3f} ({time.time()-_T0:.0f}s)", flush=True)

    overlap = {X: {Y: float(meandir[X] @ meandir[Y]) for Y in TASKS} for X in TASKS}
    probe_cols = {T: {X: overlap[X][T] for X in TASKS} for T in PROBES}
    rep = {"cell": CELL, "family": cfg["family"], "VOID": False, "KDIR": KDIR, "tasks": TASKS, "probes": PROBES,
           "w_A_acc": pre_sst, "self_consistency": self_consistency, "overlap_matrix": overlap, "probe_columns": probe_cols}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[P10post:{CELL}] OVERLAP with probe dirs (cos of realized in-V directions):", flush=True)
    for T in PROBES:
        print(f"   probe={T}: " + "  ".join(f"{X}={overlap[X][T]:+.3f}" for X in TASKS), flush=True)
    print(f"[P10post:{CELL}] self_consistency={ {k: round(v,3) for k,v in self_consistency.items()} } saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
