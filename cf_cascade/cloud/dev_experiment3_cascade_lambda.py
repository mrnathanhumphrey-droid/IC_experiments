"""#3 CASCADED-BROAD CONSTRUCTION — does building the funnel in DEPTH-ORDER protect task-A better than all-at-once breadth?

** STATUS: UNTESTED — needs GPU (Lambda A100). Written by the CPU-only spiral agent, derived line-for-line from the
   working dev_experiment_lambda.py (#2). The ONLY new machinery is `childhood_cascade` (band-restricted breadth) + arms C/D.
   The spiral agent did NOT run this (CPU-only lane). User/main-lane fires it. ~35-45 min / ~$1.5 on A100. **

IDEA: SPIRAL-1 found the network naturally collapses fine->coarse with depth (early=separated, deep=universal core),
implicitly as a side-effect of training. SPIRAL-3/5 found that OBSERVED depth-geometry is INERT (doesn't drive the mechanism).
But #2b's lesson: CONSTRUCTION has causal power where observation doesn't (the observed map was inert; non-destructive
construction protected). Cascaded-broad is the depth-axis construction protocol: build the collapse band-by-band, in depth-order.

ARMS (A,B reproduce #2 for paired comparison; C,D are new):
  A no-childhood        : w_A -> depth-train subj                                               (yank floor; #2: 0.527)
  B all-at-once broad   : w_A -> breadth stream on WHOLE net (400 steps) -> depth-train subj     (#2's childhood: 0.653)
  C cascaded-broad NEW  : w_A -> breadth in depth-BANDS early->mid->deep (matched 400 steps total) -> depth subj
  D anti-cascade  NEW   : w_A -> breadth in depth-bands REVERSE deep->mid->early (matched 400)   -> depth subj

DECISIVE: C>B (clears 2SE paired) => depth-ordered CONSTRUCTION of the funnel is LOAD-BEARING (flips depth-structure from
inert-observation to causal-construction). C>D => the early->deep DIRECTION matters (build it the way the net collapses), not
just any staging. C~B => construction inert too (consistent w/ SPIRAL-3/5; clean negative). C>B,C>D but task-B starved => rigidity.

MATCHED COMPUTE (the whole point): B=C=D all see 400 breadth batches (same data/FLOPs). They differ ONLY in WHICH params each
step updates and in what ORDER. NOTE (design-honest): in B each step updates ALL params; in C/D each step updates ONLY the
current band (~1/3 params), so each band gets ~133 steps vs B's effective 400. This is inherent to "distribute breadth over
depth" — match is on DATA/STEPS not params-updated. The D control isolates ORDER from this distribution.

BANDS = L0-7 / L8-15 / L16-23 — aligned to SPIRAL-1's natural collapse (curv-ctrl eff_rank 5.72 -> 5.08 -> 4.82, high->low).
Cascade trains ONE band at a time (others frozen); embeddings/final-LN/head frozen during phase-1 cascade (not in any band).
Phase-2 depth on subj is IDENTICAL across all arms (whole net, 250 steps) — only phase-1 differs.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/dev_experiment3.json"; NPZ = f"{G}/dev_experiment3.npz"
dev = "cuda"; SEEDS = list(range(7))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
BANDS = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]  # early / mid / deep, per SPIRAL-1 collapse

# ---- helpers copied verbatim from dev_experiment_lambda.py (#2) ----
def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]
def win(ds, k=900, w=4000):
    n = min(len(ds), w); return list(ds.shuffle(seed=0).select(range(n)))[: k]
def load_hf(key):
    from datasets import load_dataset
    if key == "emotion":
        return [(f"{r['text'].strip()}\nEmotion:", " " + ["sadness","joy","love","anger","fear","surprise"][r["label"]]) for r in win(load_dataset("dair-ai/emotion")["train"])]
    if key == "dbpedia":
        L = ["company","school","artist","athlete","politics","transport","building","nature","village","animal","plant","album","film","book"]
        return [(f"{r['content'].strip()[:400]}\nCategory:", " " + L[r["label"]]) for r in win(load_dataset("fancyzhx/dbpedia_14")["train"])]
    if key == "yahoo":
        L = ["society","science","health","education","computers","sports","business","entertainment","family","politics"]
        return [(f"{(r.get('question_title') or '').strip()}\nTopic:", " " + L[r["topic"]]) for r in win(load_dataset("community-datasets/yahoo_answers_topics")["train"]) if r.get("question_title")]
    if key == "newscat":
        return [(f"{(r.get('headline') or '').strip()}\nCategory:", " " + (r.get("category") or "").strip().lower()) for r in win(load_dataset("heegyu/news-category-dataset")["train"]) if r.get("headline") and r.get("category")]
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
def eval_clf(model, tok, eval_items, label_ids):
    model.eval(); correct = 0; n = 0; lid = torch.tensor(label_ids).to(dev)
    with torch.no_grad():
        for i in range(0, len(eval_items), 32):
            chunk = eval_items[i:i + 32]; seqs = [tok(p, add_special_tokens=False)["input_ids"][:256] for p, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            predpos = last[:, lid].argmax(1).cpu().numpy()
            for j, (_, co) in enumerate(chunk):
                true_id = tok(co, add_special_tokens=False)["input_ids"][0]
                correct += int(label_ids[predpos[j]] == true_id); n += 1
    return correct / max(n, 1)
def train_single(model, tok, items, steps, lr, seed):  # phase-2 depth (whole net), identical all arms
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed); last = float("nan")
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); last = float(loss.item())
    return last
def childhood(model, tok, stream, steps, lr, seed):  # arm B: all-at-once whole-net broad (= #2 arm B, scramble=False)
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, labels = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); batch = [items[i] for i in idx]
        inp, lb, am = b_pairs(tok, batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))

# ---- NEW: cascaded band-restricted breadth (arms C / D) ----
def childhood_cascade(model, tok, stream, steps, lr, seed, order):
    """Build the funnel band-by-band. order='forward' (early->mid->deep, arm C) or 'reverse' (deep->mid->early, arm D).
    Trains ONE band at a time (rest frozen); broad stream cycled CONTINUOUSLY across bands so each band sees breadth.
    Matched total compute = `steps` batches (same as arm B), split ~evenly across the 3 bands."""
    bands = BANDS if order == "forward" else [b for b in reversed(BANDS)]
    g = torch.Generator().manual_seed(seed); tasks = list(stream); losses = []; per = steps // len(bands); rem = steps - per * len(bands)
    step_global = 0
    for bi, band in enumerate(bands):
        for p in model.parameters(): p.requires_grad_(False)
        train_params = []
        for i in band:
            for p in model.gpt_neox.layers[i].parameters(): p.requires_grad_(True); train_params.append(p)
        opt = torch.optim.AdamW(train_params, lr=lr); model.train()
        nsteps = per + (rem if bi == len(bands) - 1 else 0)
        for _ in range(nsteps):
            t = tasks[step_global % len(tasks)]; items, _ = stream[t]
            idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); batch = [items[i] for i in idx]
            inp, lb, am = b_pairs(tok, batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
            losses.append(float(loss.item())); step_global += 1
    for p in model.parameters(): p.requires_grad_(True)  # restore for phase-2
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))

def labels_of(items): return sorted(set(c for _, c in items))
def first_ids(tok, lbls): return [tok(w, add_special_tokens=False)["input_ids"][0] for w in lbls]


def main():
    t0 = time.time(); print("[DEV3] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    from datasets import load_dataset
    _ds = load_dataset("stanfordnlp/sst2")["validation"]; _rows = [(r["sentence"].strip(), int(r["label"])) for r in _ds if r["label"] in (0, 1)]
    sst_eval = [_rows[i] for i in np.random.default_rng(0).permutation(len(_rows))[:300]]
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid); print(f"[DEV3] pre SST acc={pre_sst:.3f}", flush=True)

    # held-out = subj (forced; matches #2 for direct comparison). guard: confirm it still forgets >=0.10 in arm A.
    heldout = "subj"; ho_items = load_jsonB("medium_subj"); ho_tr, ho_ev = ho_items[:-200], ho_items[-200:]; ho_lids = first_ids(tok, labels_of(ho_items))
    # childhood stream = #2's exact non-sentiment in-neighborhood survey (subj excluded as held-out)
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {}
    for sk in stream_keys:
        it = load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)
        stream[sk] = (it, labels_of(it)); print(f"[DEV3]  stream +{sk} n={len(it)}", flush=True)
    assert len(stream) >= 4

    R = {a: {"sstA": [], "taskB": [], "post_child_sst": [], "child_l0": [], "child_l1": []} for a in ["A", "B", "C", "D"]}
    for sd in SEEDS:
        # A no childhood
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["A"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["A"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        # B all-at-once broad
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        R["B"]["post_child_sst"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["child_l0"].append(l0); R["B"]["child_l1"].append(l1)
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["B"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        # C cascaded-broad early->deep
        model.load_state_dict(ckpt_sd); l0, l1 = childhood_cascade(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, "forward")
        R["C"]["post_child_sst"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["child_l0"].append(l0); R["C"]["child_l1"].append(l1)
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["C"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        # D anti-cascade deep->early
        model.load_state_dict(ckpt_sd); l0, l1 = childhood_cascade(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, "reverse")
        R["D"]["post_child_sst"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["D"]["child_l0"].append(l0); R["D"]["child_l1"].append(l1)
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["D"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["D"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        print(f"[DEV3] seed {sd}: A/B/C/D sstA={R['A']['sstA'][-1]:.3f}/{R['B']['sstA'][-1]:.3f}/{R['C']['sstA'][-1]:.3f}/{R['D']['sstA'][-1]:.3f} "
              f"| taskB={R['A']['taskB'][-1]:.2f}/{R['B']['taskB'][-1]:.2f}/{R['C']['taskB'][-1]:.2f}/{R['D']['taskB'][-1]:.2f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"{a}_{m}": np.array(R[a][m]) for a in R for m in R[a]}, pre_sst=pre_sst, heldout=np.array([heldout]))
    A = {a: {m: [float(np.mean(R[a][m])), float(np.std(R[a][m]) / len(R[a][m]) ** 0.5)] for m in R[a] if R[a][m]} for a in R}
    rep = {"heldout": heldout, "stream": list(stream), "pre_sst": pre_sst, "bands": BANDS, "agg": A,
           "config": {"STEPS_DEPTH": STEPS_DEPTH, "STEPS_CHILD": STEPS_CHILD, "LR_DEPTH": LR_DEPTH, "LR_CHILD": LR_CHILD, "seeds": len(SEEDS)}}
    try:
        a, b, c, d = (np.array(R[x]["sstA"]) for x in "ABCD")
        def pair(x, y): m = float((x - y).mean()); se = float((x - y).std() / len(x) ** 0.5); return [m, se, m > 2 * se]
        CB, CD, BA = pair(c, b), pair(c, d), pair(b, a)
        bt, ct = np.array(R["B"]["taskB"]).mean(), np.array(R["C"]["taskB"]).mean()
        starved = ct < bt - 0.05
        if not (pre_sst - a.mean() >= 0.10):
            v = f"VOID: arm A doesn't forget (drop {pre_sst-a.mean():.3f}<0.10), no dynamic range."
        elif CB[2] and CD[2] and not starved:
            v = (f"CASCADE LOAD-BEARING: depth-ordered construction beats all-at-once (C-B={CB[0]:+.3f}>2SE) AND direction matters "
                 f"(C-D={CD[0]:+.3f}>2SE). Building the funnel early->deep protects — depth-structure flips inert->causal-when-built.")
        elif CB[2] and not CD[2]:
            v = f"STAGING NOT DIRECTION: C>B (C-B={CB[0]:+.3f}) but C~D (C-D={CD[0]:+.3f}) — distributing breadth helps, order doesn't."
        elif not CB[2]:
            v = f"CASCADE INERT (clean negative): C~B (C-B={CB[0]:+.3f}, 2SE {2*CB[1]:.3f}) — depth-distribution adds nothing over all-at-once. Consistent w/ SPIRAL-3/5 inert geometry."
        elif starved:
            v = f"RIGIDITY: C>B,C>D but task-B starved (C {ct:.2f}<B {bt:.2f}) — retains A by refusing B."
        else:
            v = f"MIXED: C-B={CB[0]:+.3f} C-D={CD[0]:+.3f} starved={starved}."
        rep["verdict"] = v; rep["paired"] = {"C_minus_B": CB, "C_minus_D": CD, "B_minus_A": BA, "starved": bool(starved)}
        print(f"\n[DEV3] === VERDICT (held-out=subj) ===\n  {v}", flush=True)
        print(f"  retention A/B/C/D = {a.mean():.3f}/{b.mean():.3f}/{c.mean():.3f}/{d.mean():.3f} (pre {pre_sst:.3f})", flush=True)
        print(f"  PAIRED C-B={CB[0]:+.3f}±{CB[1]:.3f} | C-D={CD[0]:+.3f}±{CD[1]:.3f} | B-A={BA[0]:+.3f}±{BA[1]:.3f}", flush=True)
        print(f"  task-B acc A/B/C/D = {np.array(R['A']['taskB']).mean():.2f}/{bt:.2f}/{ct:.2f}/{np.array(R['D']['taskB']).mean():.2f} (starved={starved})", flush=True)
        print(f"  childhood loss (master-none) B l0->l1={np.mean(R['B']['child_l0']):.2f}->{np.mean(R['B']['child_l1']):.2f} "
              f"C={np.mean(R['C']['child_l0']):.2f}->{np.mean(R['C']['child_l1']):.2f} D={np.mean(R['D']['child_l0']):.2f}->{np.mean(R['D']['child_l1']):.2f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[DEV3] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[DEV3] saved dev_experiment3.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
