"""TOGGLE REDESIGN — OPTION 1: PIN DOWN THE ANTAGONISM SPACE (measurement-only, no arms, no toggle).

The toggle Phase-1 construction failed gate (a): synthetic-label cosine in top-128 Fisher (V) sat at +0.56-0.62 across all
nominal c, never reaching the −0.23 target. Diagnosis: (i) the label knob doesn't steer the weight-gradient direction in V
(Jacobian common-mode dominates), and (ii) rung-2's −0.23 antagonism was measured in a DIFFERENT space — g_A(held-out SST) vs
g_B0(task-B step-0) cosine in the TOP-16 CURVATURE dirs per carrier (KF=16), median over carriers — not the top-128 Fisher.

This run reproduces rung-2's measure EXACTLY at the toggle's actual pinned w_A, then measures the natural antagonism of a bank of
real task-Bs across THREE spaces, to decide: (1) which space −0.23 lives in, (2) whether negative cosine is reachable at all and
where, (3) what the natural antagonism of the DEPTH task (subj, the dispositive's held-out) actually is. Gates the toggle redesign.

REPRODUCE-KNOWN ANCHOR: agnews antagonism in curv16 should be ≈ −0.23 (Pythia rung-2). If it doesn't reproduce at the pinned w_A,
the pinned checkpoint differs from rung-2's w_A regime — flag (informative either way).

Faithful to rung2b_fragile_opt_lambda.py: curv_basis = top-KB Fisher eigvecs (col0=highest λ); g_A = summed held-out SST-val
gradient on carriers; g_B0 = task-B first-batch (step-0) gradient; antag = cos(g_A@Vt, g_B0@Vt), Vt = V[:, :KF]. Pythia (the
toggle substrate). CPU-light but needs GPU for gradients; short (~10-15 min).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
MODEL = "facebook/opt-350m"; CKPT = f"{G}/opt350m_sst2_wA.pt"
OUT = f"{G}/opt_antag_space.json"; NPZ = f"{G}/opt_antag_space.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
BATCH = 16; N_FISHER = 150; N_HELDOUT = 96; KB = 128; KF = 16
TASKBANK = ["agnews", "subj", "emotion", "dbpedia", "yahoo"]   # agnews = rung-2 anchor; subj = dispositive depth task


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def win(ds, k=900, w=4000):
    n = min(len(ds), w); return list(ds.shuffle(seed=0).select(range(n)))[: k]


def load_hf(key):
    from datasets import load_dataset
    LD = lambda *a, **k: load_dataset(*a, **k)
    if key == "emotion":
        return [(f"{r['text'].strip()}\nEmotion:", " " + ["sadness","joy","love","anger","fear","surprise"][r["label"]]) for r in win(LD("dair-ai/emotion")["train"])]
    if key == "dbpedia":
        Lc = ["company","school","artist","athlete","politics","transport","building","nature","village","animal","plant","album","film","book"]
        return [(f"{r['content'].strip()[:400]}\nCategory:", " " + Lc[r["label"]]) for r in win(LD("fancyzhx/dbpedia_14")["train"])]
    if key == "yahoo":
        Lc = ["society","science","health","education","computers","sports","business","entertainment","family","politics"]
        return [(f"{(r.get('question_title') or '').strip()}\nTopic:", " " + Lc[r["topic"]]) for r in win(LD("community-datasets/yahoo_answers_topics")["train"]) if r.get("question_title")]
    raise ValueError(key)


def taskB(key):
    if key == "agnews": return load_jsonB("low_agnews")
    if key == "subj": return load_jsonB("medium_subj")
    return load_hf(key)


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:128]; c = tok(co, add_special_tokens=False)["input_ids"][:8]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher_of(model, params, names, tok, itemsA):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    g = torch.Generator().manual_seed(0)
    for _ in range(N_FISHER):
        idx = torch.randperm(len(itemsA), generator=g)[:1].tolist()
        inp, lb, am = b_pairs(tok, [itemsA[i] for i in idx]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: gr = params[n].grad.detach().double(); F[n] += gr.T @ gr
    return F


def curv_basis(model, tok, names, itemsA):
    params = dict(model.named_parameters()); FA = fisher_of(model, params, names, tok, itemsA); V = {}
    for n in names:
        ev, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KB:].flip(dims=[1]).double().contiguous(); del FA[n]   # col0=highest λ
    return V


def heldout_gA(model, params, names, tok, ev):
    """summed per-example held-out SST-val gradient on carriers (rung-2's g_A)."""
    gA = {n: torch.zeros_like(params[n], dtype=torch.double) for n in names}
    for s, l in ev[:N_HELDOUT]:
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: gA[n] = gA[n] + params[n].grad.detach().double()
    model.zero_grad(set_to_none=True)
    return gA


def step0_gB(model, params, names, tok, items):
    """task-B first-batch (step-0) gradient at w_A."""
    g = torch.Generator().manual_seed(0); idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
    inp, lb, am = b_pairs(tok, [items[i] for i in idx]); model.zero_grad(set_to_none=True)
    model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
    gB = {n: params[n].grad.detach().double().clone() for n in names}
    model.zero_grad(set_to_none=True)
    return gB


def cos_spaces(gA, gB, names, V):
    """per-carrier cosine in curv16 / curv128 / raw, return medians."""
    c16, c128, craw = [], [], []
    for n in names:
        V16 = V[n][:, :KF]; V128 = V[n][:, :KB]
        a16, b16 = gA[n] @ V16, gB[n] @ V16; c16.append(float((a16 * b16).sum() / (a16.norm() * b16.norm() + 1e-12)))
        a128, b128 = gA[n] @ V128, gB[n] @ V128; c128.append(float((a128 * b128).sum() / (a128.norm() * b128.norm() + 1e-12)))
        ar, br = gA[n].reshape(-1), gB[n].reshape(-1); craw.append(float((ar * br).sum() / (ar.norm() * br.norm() + 1e-12)))
    return {"curv16": c16, "curv128": c128, "raw": craw}


def eval_sst(model, tok, ev, posid, negid):
    model.eval(); cor = 0
    with torch.no_grad():
        for i in range(0, len(ev), 32):
            ch = ev[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:128] for s, _ in ch]
            L = max(len(x) for x in seqs); inp = np.full((len(seqs), L), tok.pad_token_id, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pr = (lg[:, posid] > lg[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(ch): cor += int(pr[j] == lab)
    return cor / len(ev)


def main():
    t0 = time.time(); print("[ANTAG-OPT] RUN A — matched-w_A antagonism parity at OPT dispositive w_A (measurement-only)", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); print(f"[ANTAG] loaded pinned w_A {CKPT}", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    ev = sst_rows("validation", 300, 0); accA = eval_sst(model, tok, ev, posid, negid)
    print(f"[ANTAG] w_A SST acc={accA:.3f} (rung-1 ~0.77, dispositive 0.753); {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    sstA = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", 2000, 0)]
    V = curv_basis(model, tok, carriers, sstA); print(f"[ANTAG] curv basis (top-{KB}) done ({time.time()-t0:.0f}s)", flush=True)
    gA = heldout_gA(model, params, carriers, tok, ev); print(f"[ANTAG] g_A (held-out SST val, {N_HELDOUT} ex) done ({time.time()-t0:.0f}s)", flush=True)

    rows = {}
    for key in TASKBANK:
        items = taskB(key); gB = step0_gB(model, params, carriers, tok, items); cs = cos_spaces(gA, gB, carriers, V)
        med = {sp: float(np.median(cs[sp])) for sp in cs}; rows[key] = {"median": med, "per_carrier": cs}
        print(f"[ANTAG] {key:8s}: curv16={med['curv16']:+.3f} | curv128={med['curv128']:+.3f} | raw={med['raw']:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    anchor = rows["agnews"]["median"]["curv16"]; anchor_ok = anchor < -0.10
    # which space spans negative across the bank?
    spans = {}
    for sp in ["curv16", "curv128", "raw"]:
        vals = [rows[k]["median"][sp] for k in TASKBANK]; spans[sp] = {"min": float(min(vals)), "max": float(max(vals)), "reaches_neg": bool(min(vals) < -0.05)}
    rep = {"phase": "antag_space", "model": MODEL, "wA_sst_acc": accA, "KF": KF, "KB": KB, "taskbank": TASKBANK,
           "rows": {k: rows[k]["median"] for k in rows}, "anchor_agnews_curv16": anchor, "anchor_reproduces_neg023": bool(anchor_ok), "spans": spans}
    np.savez(NPZ, **{f"{k}_{sp}": np.array(rows[k]["per_carrier"][sp]) for k in rows for sp in ["curv16", "curv128", "raw"]})
    print(f"\n[ANTAG] === ANTAGONISM SPACE MAP (median over carriers) ===", flush=True)
    print(f"  reproduce-known: agnews curv16 = {anchor:+.3f} (rung-2 Pythia −0.23) -> {'REPRODUCES' if anchor_ok else 'DOES NOT reproduce — pinned w_A differs from rung-2 regime, flag'}", flush=True)
    for sp in ["curv16", "curv128", "raw"]:
        print(f"  {sp:7s}: " + " | ".join(f"{k}={rows[k]['median'][sp]:+.3f}" for k in TASKBANK) + f"   [range {spans[sp]['min']:+.3f},{spans[sp]['max']:+.3f}; neg-reachable={spans[sp]['reaches_neg']}]", flush=True)
    print(f"  subj (dispositive depth task) curv16={rows['subj']['median']['curv16']:+.3f} -> the antagonism level the toggle must hit at the −0.23 end", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[ANTAG] saved antag_space.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
