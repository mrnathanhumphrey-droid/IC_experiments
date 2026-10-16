"""OPTION 2 — CONFIRMATORY 2nd NET (OPT-350m): does #2/#2b replicate on a DIFFERENT architecture family?

Pythia-410m (GPT-NeoX) gave: childhood slivers (B,D >> A), but it's NON-DESTRUCTIVENESS not the map (B~D; #2b). This run
re-runs the DECISIVE 4-arm battery on OPT-350m (different family, standard nn.Linear carriers so the Fisher/projection code path is
identical). Self-contained: trains its OWN task-A (SST-2) -> w_A first (no Pythia checkpoint to reuse), then A/B/C/D.

ARMS (same as #2b): A no-childhood / B real-childhood / C scrambled (noise-fit, destructive) / D non-destructive no-map
(childhood grads projected off task-A top-128 curvature subspace). DECISIVE: B-A>0 (childhood slivers replicates?), B~D
(map inert replicates?), D-A>0 (non-destructive-no-map helps replicates?). Guards on D: non-destructive (post-child >= B's),
no-map (mapfrac << B), trained (loss drops). PRE-COMMITTED: if B>A & B~D & D>A on OPT too -> #2b cross-net CONFIRMED (the survivor
generalizes). If B>D on OPT -> the map-inert finding was Pythia-specific (net-dependent). 5 paired seeds; save before readout.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "facebook/opt-350m"; AX = f"{G}/overlap_axis"
OUT = f"{G}/dev_net2_opt.json"; NPZ = f"{G}/dev_net2_opt.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]  # OPT standard nn.Linear (out,in)
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7))  # spec rung-3: ~6-8 paired seeds for error-bar separation (Pythia used 7)
STEPS_WA = 500; LR_WA = 5e-5; STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200


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


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def fisher_taskA(model, params, names, tok):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
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
                true_id = tok(co, add_special_tokens=False)["input_ids"][0]; correct += int(label_ids[predpos[j]] == true_id); n += 1
    return correct / max(n, 1)


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def train_sst(model, tok, steps, lr, seed):
    items = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", 2000, 7)]
    train_single(model, tok, items, steps, lr, seed)


def childhood(model, tok, stream, steps, lr, seed, scramble=False, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(1000 + seed); tasks = list(stream); losses = []; params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, labels = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); batch = [items[i] for i in idx]
        if scramble: batch = [(p, labels[rng.integers(len(labels))]) for (p, _) in batch]
        inp, lb, am = b_pairs(tok, batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_((gd @ Vproj[n]) @ Vproj[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))


def map_subspace_frac(model, ckpt_sd, carriers, V):
    num = 0.0; den = 0.0; params = dict(model.named_parameters())
    with torch.no_grad():
        for n in carriers:
            dw = (params[n].detach() - ckpt_sd[n].to(dev)).double(); proj = dw @ V[n].double()
            num += float((proj * proj).sum()); den += float((dw * dw).sum())
    return num / max(den, 1e-12)


def labels_of(items): return sorted(set(c for _, c in items))
def first_ids(tok, lbls): return [tok(w, add_special_tokens=False)["input_ids"][0] for w in lbls]


def main():
    t0 = time.time(); print(f"[N2] loading {MODEL}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    print(f"[N2] {len(carriers)} carriers (e.g. {carriers[0]}, {carriers[-1]})", flush=True)
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)

    # PHASE 0 — train task-A (SST-2) -> w_A (this net has no prior checkpoint)
    print(f"[N2] training task-A (SST) {STEPS_WA} steps...", flush=True)
    train_sst(model, tok, STEPS_WA, LR_WA, 0)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[N2] w_A SST acc={pre_sst:.3f} ({time.time()-t0:.0f}s)", flush=True)
    assert pre_sst > 0.68, f"w_A too weak ({pre_sst:.3f}) — net didn't learn task-A; abort"

    # task-A curvature subspace (for arm D projection)
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].float().contiguous()
    model.load_state_dict(ckpt_sd); params = dict(model.named_parameters())
    print(f"[N2] task-A curvature basis done ({time.time()-t0:.0f}s)", flush=True)

    # held-out depth task-B (precheck subj/emotion)
    cands = {"subj": load_jsonB("medium_subj"), "emotion": load_hf("emotion")}
    pick = {}
    for k, items in cands.items():
        tr, ev = items[:-200], items[-200:]; lids = first_ids(tok, labels_of(items))
        model.load_state_dict(ckpt_sd); train_single(model, tok, tr, STEPS_DEPTH, LR_DEPTH, 0)
        a_sst = eval_sst(model, tok, sst_eval, posid, negid); pick[k] = {"forget": pre_sst - a_sst, "taskB": eval_clf(model, tok, ev, lids)}
        print(f"[N2]   precheck {k}: forget={pick[k]['forget']:.3f} taskB={pick[k]['taskB']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    ok = {k: v for k, v in pick.items() if v["forget"] >= 0.10 and v["taskB"] > 0.55}
    heldout = max(ok or pick, key=lambda k: pick[k]["forget"])
    ho_items = cands[heldout]; ho_tr, ho_ev = ho_items[:-200], ho_items[-200:]; ho_lids = first_ids(tok, labels_of(ho_items))
    print(f"[N2] HELD-OUT = {heldout} (forget {pick[heldout]['forget']:.3f})", flush=True)

    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", ("emotion" if heldout == "subj" else "subj")]
    stream = {sk: ((load_jsonB("low_agnews") if sk == "agnews" else (load_jsonB("medium_subj") if sk == "subj" else load_hf(sk))), None) for sk in stream_keys}
    for sk in stream: stream[sk] = (stream[sk][0], labels_of(stream[sk][0]))
    print(f"[N2] stream={list(stream)} ({time.time()-t0:.0f}s)", flush=True)

    R = {a: {"sstA": [], "taskB": [], "post_child": [], "cl0": [], "cl1": [], "mapfrac": []} for a in ["A", "B", "C", "D"]}
    for sd in SEEDS:
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["A"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["A"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids)); R["A"]["post_child"].append(pre_sst)
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        R["B"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["cl0"].append(l0); R["B"]["cl1"].append(l1); R["B"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V))
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["B"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, scramble=True)
        R["C"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["cl0"].append(l0); R["C"]["cl1"].append(l1); R["C"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V))
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["C"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        R["D"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["D"]["cl0"].append(l0); R["D"]["cl1"].append(l1); R["D"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V))
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["D"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["D"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        print(f"[N2] seed {sd}: sstA A/B/C/D={R['A']['sstA'][-1]:.3f}/{R['B']['sstA'][-1]:.3f}/{R['C']['sstA'][-1]:.3f}/{R['D']['sstA'][-1]:.3f} | "
              f"postchild B/C/D={R['B']['post_child'][-1]:.3f}/{R['C']['post_child'][-1]:.3f}/{R['D']['post_child'][-1]:.3f} | "
              f"mapfrac B/D={R['B']['mapfrac'][-1]:.3f}/{R['D']['mapfrac'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"{a}_{m}": np.array(R[a][m]) for a in R for m in R[a] if R[a][m]}, pre_sst=pre_sst, heldout=np.array([heldout]))
    def ms(x): x = np.array(x, float); return [float(x.mean()), float(x.std() / max(len(x), 1) ** 0.5)]
    A = {a: {m: ms(R[a][m]) for m in R[a] if R[a][m]} for a in R}
    rep = {"model": MODEL, "heldout": heldout, "pre_sst": pre_sst, "precheck": pick, "agg": A}
    try:
        a, b, c, d = (np.array(R[x]["sstA"]) for x in "ABCD")
        def me(x): return [float(np.mean(x)), float(np.std(x) / max(len(x), 1) ** 0.5)]
        ba_m, ba_se = me(b - a); bd_m, bd_se = me(b - d); da_m, da_se = me(d - a)
        dpost = np.mean(R["D"]["post_child"]); bpost = np.mean(R["B"]["post_child"]); dmap = np.mean(R["D"]["mapfrac"]); bmap = np.mean(R["B"]["mapfrac"])
        guards_ok = (dpost >= bpost - 0.03) and (dmap < 0.5 * bmap) and (np.mean(R["D"]["cl0"]) - np.mean(R["D"]["cl1"]) > 0.3)
        slivers = ba_m > 2 * ba_se; map_inert = abs(bd_m) <= 2 * bd_se; nondestr_helps = da_m > 2 * da_se
        if not guards_ok:
            v = f"D GUARDS WEAK on OPT (post {dpost:.3f} vs B {bpost:.3f}, mapfrac D {dmap:.3f} vs B {bmap:.3f}) — interpret arm D cautiously."
        elif slivers and map_inert and nondestr_helps:
            v = (f"#2b CONFIRMED CROSS-NET (OPT-350m): childhood slivers (B-A={ba_m:+.3f}>2SE), it's NON-DESTRUCTIVENESS not the map "
                 f"(B~D, B-D={bd_m:+.3f} n.s.), non-destructive no-map helps (D-A={da_m:+.3f}>2SE). The survivor GENERALIZES across families.")
        elif slivers and not map_inert and bd_m > 0:
            v = (f"PARTIAL / NET-DEPENDENT: childhood slivers (B-A={ba_m:+.3f}) BUT B>D on OPT (B-D={bd_m:+.3f}>2SE) — the map-inert "
                 f"finding may be Pythia-specific; on OPT the map adds something. Flag for follow-up.")
        elif not slivers:
            v = (f"DOES NOT REPLICATE: childhood doesn't sliver on OPT (B-A={ba_m:+.3f} n.s.) — the #2 effect is net-dependent. "
                 f"Big caveat on generality.")
        else:
            v = f"MIXED on OPT: B-A={ba_m:+.3f} B-D={bd_m:+.3f} D-A={da_m:+.3f} — inspect."
        rep["verdict"] = v; rep["paired"] = {"B_minus_A": [ba_m, ba_se], "B_minus_D": [bd_m, bd_se], "D_minus_A": [da_m, da_se]}
        rep["guards"] = {"ok": bool(guards_ok), "D_postchild": float(dpost), "B_postchild": float(bpost), "D_mapfrac": float(dmap), "B_mapfrac": float(bmap)}
        print(f"\n[N2] === VERDICT (OPT-350m, held-out={heldout}) ===\n  {v}", flush=True)
        print(f"  retention: A={a.mean():.3f} B={b.mean():.3f} C={c.mean():.3f} D={d.mean():.3f} (pre {pre_sst:.3f})", flush=True)
        print(f"  PAIRED B-A={ba_m:+.3f}±{ba_se:.3f} | B-D={bd_m:+.3f}±{bd_se:.3f} | D-A={da_m:+.3f}±{da_se:.3f}", flush=True)
        print(f"  guards D: post-child {dpost:.3f} (B {bpost:.3f}) | mapfrac D {dmap:.3f} vs B {bmap:.3f}", flush=True)
        print(f"  task-B: A {np.mean(R['A']['taskB']):.3f} B {np.mean(R['B']['taskB']):.3f} D {np.mean(R['D']['taskB']):.3f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[N2] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[N2] saved dev_net2_opt.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
