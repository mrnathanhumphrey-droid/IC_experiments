"""PROBE 4 — disjoint-subspace dual deposits. Do two deposits in non-overlapping V-subspaces REINFORCE / INTERFERE /
act INDEPENDENTLY when the phase-2 surge arrives? Pre-reg PREREG_PROBE4_DUAL_2026_06_23.md (LOCK a036d7b9). Band ONLY.
S1={agnews,dbpedia,yahoo} S2={newscat,emotion} (LOCKED). DISJOINTNESS GATE (HARD-STOP): top principal-angle cosine
between span(S_dep1) and span(S_dep2) (top-r=8 per carrier); carrier-median overlap > 0.35 => HARD-STOP. Arms: single1,
single2, dual(800 interleaved), single800 (compute-matched). Reproduce-anchor: single800 B-A vs banked.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
LR_DEPTH = 5e-5; STEPS_DEPTH = 250; LR_CHILD = 3e-5; BATCH = 16; KPROJ = 128; N_FISHER = 200; RSUB = 8
S1 = ["agnews", "dbpedia", "yahoo"]; S2 = ["newscat", "emotion"]; OVERLAP_STOP = 0.35
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT",    "anchor_BmA": 0.089},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA", "anchor_BmA": 0.073},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe4_dual_{CELL}.json"


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
def childhood(model, tok, substream, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(substream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = substream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()
def phase2(model, tok, ho_tr, sst_eval, posid, negid, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_DEPTH); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_DEPTH):
        idx = torch.randperm(len(ho_tr), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [ho_tr[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; r = eval_sst(model, tok, sst_eval, posid, negid); torch.cuda.empty_cache(); return r
def labels_of(items): return sorted(set(c for _, c in items))
def subspace_V(model, carriers, ckpt_sd, V):
    """per carrier: top-RSUB right-singular dirs (din-space, in span V) of P_V(deposit). returns {n: (RSUB,din) double}."""
    P = dict(model.named_parameters()); out = {}
    for n in carriers:
        df = (P[n].detach().double() - ckpt_sd[n].to(dev).double()); dv = (df @ V[n]) @ V[n].T
        _, _, Vh = torch.linalg.svd(dv, full_matrices=False); out[n] = Vh[:RSUB].contiguous()
    return out


def main():
    t0 = time.time(); print(f"[P4:{CELL}] {cfg['model']} dual deposits S1={S1} S2={S2}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P4:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    ho_tr = load_jsonB("medium_subj")[:-200]
    raw = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in S1 + S2}
    stream = {sk: (it, labels_of(it)) for sk, it in raw.items()}
    sub1 = {k: stream[k] for k in S1}; sub2 = {k: stream[k] for k in S2}; subdual = stream  # all 5 interleaved

    # ---- DISJOINTNESS GATE (seed 0) ----
    model.load_state_dict(ckpt_sd); childhood(model, tok, sub1, 400, LR_CHILD, 0); B1g = subspace_V(model, carriers, ckpt_sd, V)
    model.load_state_dict(ckpt_sd); childhood(model, tok, sub2, 400, LR_CHILD, 0); B2g = subspace_V(model, carriers, ckpt_sd, V)
    overlaps = []
    for n in carriers:
        M = B1g[n] @ B2g[n].T  # (RSUB,RSUB) = inner products of orthonormal rows
        overlaps.append(float(torch.linalg.svdvals(M)[0]))  # top principal-angle cosine
    ov_med = float(np.median(overlaps)); model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    print(f"[P4:{CELL}] disjointness gate: carrier-median top-PA cos={ov_med:.3f} (STOP if >{OVERLAP_STOP})", flush=True)
    if ov_med > OVERLAP_STOP:
        json.dump({"cell": CELL, "VOID": True, "reason": "disjointness gate failed", "overlap_median": ov_med,
                   "overlap_stop": OVERLAP_STOP, "S1": S1, "S2": S2}, open(OUT, "w"), indent=2)
        print(f"[P4:{CELL}] HARD-STOP: streams not disjoint (overlap {ov_med:.3f} > {OVERLAP_STOP}). VOID, no run.", flush=True); return

    retA = []; r1 = []; r2 = []; rdual = []; r800 = []; surv1 = []; surv2 = []
    for sd in SEEDS:
        model.load_state_dict(ckpt_sd); retA.append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
        # single1 -> capture deposit subspace + state
        model.load_state_dict(ckpt_sd); childhood(model, tok, sub1, 400, LR_CHILD, sd)
        Bd1 = subspace_V(model, carriers, ckpt_sd, V); w1 = {n: dict(model.named_parameters())[n].detach().double().clone() for n in carriers}
        r1.append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
        model.load_state_dict(ckpt_sd); childhood(model, tok, sub2, 400, LR_CHILD, sd)
        Bd2 = subspace_V(model, carriers, ckpt_sd, V); w2 = {n: dict(model.named_parameters())[n].detach().double().clone() for n in carriers}
        r2.append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
        # dual (800 interleaved) -> survival of each deposit subspace then phase2
        model.load_state_dict(ckpt_sd); childhood(model, tok, subdual, 800, LR_CHILD, sd)
        Pdual = dict(model.named_parameters()); s1n = []; s2n = []
        for n in carriers:
            dd = (Pdual[n].detach().double() - ckpt_sd[n].to(dev).double())
            d1 = (w1[n] - ckpt_sd[n].to(dev).double()); d2 = (w2[n] - ckpt_sd[n].to(dev).double())
            num1 = float((dd @ Bd1[n].T).norm()); den1 = float((d1 @ Bd1[n].T).norm()) + 1e-30
            num2 = float((dd @ Bd2[n].T).norm()); den2 = float((d2 @ Bd2[n].T).norm()) + 1e-30
            s1n.append(num1 / den1); s2n.append(num2 / den2)
        surv1.append(float(np.median(s1n))); surv2.append(float(np.median(s2n)))
        rdual.append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
        # single800 (compute-matched: S1 alone, 800)
        model.load_state_dict(ckpt_sd); childhood(model, tok, sub1, 800, LR_CHILD, sd); r800.append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
        if sd % 5 == 0 or sd == SEEDS[-1]:
            print(f"[P4:{CELL}] seed {sd}: A={retA[-1]:.3f} s1={r1[-1]:.3f} s2={r2[-1]:.3f} dual={rdual[-1]:.3f} s800={r800[-1]:.3f} surv1/2={surv1[-1]:.2f}/{surv2[-1]:.2f} ({time.time()-t0:.0f}s)", flush=True)

    ns = len(SEEDS); A = np.array(retA); me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    B1mA = me((np.array(r1) - A).tolist()); B2mA = me((np.array(r2) - A).tolist())
    BdmA = me((np.array(rdual) - A).tolist()); B8mA = me((np.array(r800) - A).tolist())
    anchor_ok = abs(B8mA[0] - cfg["anchor_BmA"]) < 0.03
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst,
           "S1": S1, "S2": S2, "disjointness_overlap_median": ov_med, "VOID": False, "retA": me(A),
           "B1_minus_A": B1mA, "B2_minus_A": B2mA, "Bdual_minus_A": BdmA, "B800_minus_A": B8mA,
           "sum_singles_minus_A": [B1mA[0] + B2mA[0], None], "surv_dep1_in_dual": me(surv1), "surv_dep2_in_dual": me(surv2),
           "anchor_BmA_expected": cfg["anchor_BmA"], "ANCHOR_OK": bool(anchor_ok)}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[P4:{CELL}] === ANCHOR single800 B-A={B8mA[0]:+.3f}(exp {cfg['anchor_BmA']:+.3f}) -> {'OK' if anchor_ok else 'MISMATCH!!'} ===", flush=True)
    print(f"   B1-A={B1mA[0]:+.3f}  B2-A={B2mA[0]:+.3f}  sum={B1mA[0]+B2mA[0]:+.3f}  Bdual-A={BdmA[0]:+.3f}  B800-A={B8mA[0]:+.3f}", flush=True)
    print(f"   survival in dual: dep1={rep['surv_dep1_in_dual'][0]:.2f} dep2={rep['surv_dep2_in_dual'][0]:.2f}  | REINFORCE if dual>max(singles)&>B800; INTERFERE if dual<sum or deposits erode", flush=True)
    print(f"[P4:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
