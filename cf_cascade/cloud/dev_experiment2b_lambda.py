"""#2b NON-DESTRUCTIVE NO-MAP CONTROL — isolate "the map ROUTES the surge" (Mode 2) from "the map leaves a better START" (Mode 1).

#2 confirmed structured childhood slivers (B>A, B>C), but arm C (scrambled) is NOISE-FITTING = itself destructive (C post-child
SST 0.666 < B 0.722). So B's edge over C conflates: Mode 1 (real-structure training is LESS DESTRUCTIVE to task-A in phase-1 →
better starting point) and Mode 2 (the map ROUTES the phase-2 depth surge AROUND task-A — the actual sliver claim). This run adds a
NON-DESTRUCTIVE NO-MAP control (arm D) to isolate Mode 2.

ARM D construction (the crux): same stream as B, but childhood gradients are GPM-PROJECTED off task-A's top-128 Fisher (curvature)
subspace per carrier-layer each step. => (i) matched compute & SAME DATA as B; (ii) NON-DESTRUCTIVE by construction (can't move
task-A's curved/forgetting directions — the protection lever applied during the childhood); (iii) NO MAP by construction (can't
write structure into the top-128 task-A-curvature subspace where the v3 neighborhood map was MEASURED to live). The ONLY difference
B-vs-D is whether the childhood may write the map subspace. Note: projection makes D >= B non-destructive (>= B's start), so B>D is a
CONSERVATIVE Mode-2 test (B wins despite D's equal/better head start).

GUARDS (mandatory, else void): (a) NON-DESTRUCTIVE — D post-phase-1 SST ~>= B's pre-depth (NOT C's 0.666); (b) NO-MAP — fraction of
D's childhood WEIGHT-CHANGE energy inside the task-A curvature subspace ~0 (vs B substantial) = D wrote no map-subspace structure.
Plus: D's childhood loss still drops (non-degenerate, it DID train in the complement, not a no-op ~= arm A).

DECISIVE: B vs D (both non-destructive going in; B has map, D doesn't).
  B>D (clears 2SE) -> MODE 2 CONFIRMED: the map ROUTES the depth surge around task-A, over and above a good starting point. Strong claim.
  B~D -> Mode 1 only: B's edge was the non-destructive start, NOT routing. Narrows the claim hard (structured phase-1 = gentle phase-1).
  D~A (& B>both) -> non-destructive-no-map does nothing; the map is the WHOLE story (strongest).
Also report PHASE-2 DAMAGE per arm (post_child - final) = the direct routing measure. 7 PAIRED seeds; judge on error bars.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/dev_experiment2b.json"; NPZ = f"{G}/dev_experiment2b.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
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


def childhood(model, tok, stream, steps, lr, seed, scramble=False, Vproj=None, carriers=None):
    """breadth pass. scramble=random label/example (no-structure). Vproj!=None -> project carrier grads off task-A subspace (no-map, non-destructive)."""
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
    """fraction of childhood weight-change energy INSIDE the task-A curvature subspace (no-map guard: ~0 for D, large for B)."""
    num = 0.0; den = 0.0; params = dict(model.named_parameters())
    with torch.no_grad():
        for n in carriers:
            dw = (params[n].detach() - ckpt_sd[n].to(dev)).double(); proj = dw @ V[n].double()
            num += float((proj * proj).sum()); den += float((dw * dw).sum())
    return num / max(den, 1e-12)


def labels_of(items): return sorted(set(c for _, c in items))
def first_ids(tok, lbls): return [tok(w, add_special_tokens=False)["input_ids"][0] for w in lbls]


def main():
    t0 = time.time(); print("[2B] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[2B] pre SST-2 acc={pre_sst:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # task-A Fisher -> top-KPROJ curvature subspace per carrier (the projection / map subspace)
    FA = fisher_taskA(model, params, names=carriers, tok=tok)
    V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].float().contiguous()
    model.load_state_dict(ckpt_sd); params = dict(model.named_parameters())
    print(f"[2B] task-A curvature basis (top-{KPROJ}) done ({time.time()-t0:.0f}s)", flush=True)

    # held-out depth task-B (same as #2: subj; precheck confirms forgetting)
    cands = {"subj": load_jsonB("medium_subj"), "emotion": load_hf("emotion")}
    pick = {}
    for k, items in cands.items():
        tr, ev = items[:-200], items[-200:]; lids = first_ids(tok, labels_of(items))
        model.load_state_dict(ckpt_sd); train_single(model, tok, tr, STEPS_DEPTH, LR_DEPTH, 0)
        a_sst = eval_sst(model, tok, sst_eval, posid, negid); pick[k] = {"forget": pre_sst - a_sst, "taskB": eval_clf(model, tok, ev, lids)}
        print(f"[2B]   precheck {k}: forget={pick[k]['forget']:.3f} taskB={pick[k]['taskB']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    ok = {k: v for k, v in pick.items() if v["forget"] >= 0.10 and v["taskB"] > 0.55}
    heldout = max(ok or pick, key=lambda k: pick[k]["forget"])
    ho_items = cands[heldout]; ho_tr, ho_ev = ho_items[:-200], ho_items[-200:]; ho_lids = first_ids(tok, labels_of(ho_items))
    print(f"[2B] HELD-OUT = {heldout} (forget {pick[heldout]['forget']:.3f})", flush=True)

    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", ("emotion" if heldout == "subj" else "subj")]
    stream = {}
    for sk in stream_keys:
        it = load_jsonB("low_agnews") if sk == "agnews" else (load_jsonB("medium_subj") if sk == "subj" else load_hf(sk))
        stream[sk] = (it, labels_of(it))
    print(f"[2B] stream={list(stream)} (NON-sentiment, held-out excluded) ({time.time()-t0:.0f}s)", flush=True)

    R = {a: {"sstA": [], "taskB": [], "post_child": [], "cl0": [], "cl1": [], "mapfrac": []} for a in ["A", "B", "C", "D"]}
    for sd in SEEDS:
        # A — no childhood
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["A"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["A"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids)); R["A"]["post_child"].append(pre_sst)
        # B — real childhood
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        R["B"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["cl0"].append(l0); R["B"]["cl1"].append(l1); R["B"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V))
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["B"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        # C — scrambled childhood
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, scramble=True)
        R["C"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["cl0"].append(l0); R["C"]["cl1"].append(l1); R["C"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V))
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["C"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        # D — non-destructive no-map childhood (projected off task-A subspace)
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        R["D"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["D"]["cl0"].append(l0); R["D"]["cl1"].append(l1); R["D"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V))
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["D"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["D"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        print(f"[2B] seed {sd}: sstA A/B/C/D={R['A']['sstA'][-1]:.3f}/{R['B']['sstA'][-1]:.3f}/{R['C']['sstA'][-1]:.3f}/{R['D']['sstA'][-1]:.3f} | "
              f"postchild B/C/D={R['B']['post_child'][-1]:.3f}/{R['C']['post_child'][-1]:.3f}/{R['D']['post_child'][-1]:.3f} | "
              f"mapfrac B/C/D={R['B']['mapfrac'][-1]:.3f}/{R['C']['mapfrac'][-1]:.3f}/{R['D']['mapfrac'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"{a}_{m}": np.array(R[a][m]) for a in R for m in R[a] if R[a][m]}, pre_sst=pre_sst, heldout=np.array([heldout]))
    def ms(x): x = np.array(x); return [float(x.mean()), float(x.std() / len(x) ** 0.5)]
    A = {a: {m: ms(R[a][m]) for m in R[a] if R[a][m]} for a in R}
    rep = {"heldout": heldout, "stream": list(stream), "pre_sst": pre_sst, "precheck": pick, "KPROJ": KPROJ, "agg": A,
           "config": {"STEPS_DEPTH": STEPS_DEPTH, "STEPS_CHILD": STEPS_CHILD, "LR_DEPTH": LR_DEPTH, "LR_CHILD": LR_CHILD, "seeds": len(SEEDS)}}
    try:
        a, b, c, d = (np.array(R[x]["sstA"]) for x in "ABCD")
        bd, da, dc = b - d, d - a, d - c  # paired
        def me(x): return [float(np.mean(x)), float(np.std(x) / len(x) ** 0.5)]
        bd_m, bd_se = me(bd); da_m, da_se = me(da)
        # guards on D
        d_postchild = np.mean(R["D"]["post_child"]); b_postchild = np.mean(R["B"]["post_child"]); c_postchild = np.mean(R["C"]["post_child"])
        d_mapfrac = np.mean(R["D"]["mapfrac"]); b_mapfrac = np.mean(R["B"]["mapfrac"])
        d_clo = np.mean(R["D"]["cl0"]) - np.mean(R["D"]["cl1"])
        nondestructive = d_postchild >= b_postchild - 0.03   # D not worse than B going in
        nomap = d_mapfrac < 0.5 * b_mapfrac                  # D wrote far less into the map subspace than B
        trained = d_clo > 0.3                                 # D's childhood actually trained (non-degenerate)
        # phase-2 damage per arm
        dmg = {x: float(np.mean(R[x]["post_child"]) - np.mean(R[x]["sstA"])) for x in "ABCD"}
        if not (nondestructive and trained):
            v = (f"VOID: arm D guards failed — non-destructive={nondestructive} (D post-child {d_postchild:.3f} vs B {b_postchild:.3f}), "
                 f"trained={trained} (childhood loss drop {d_clo:.2f}). Not a clean Mode-2 control.")
        elif not nomap:
            v = (f"VOID: arm D no-map guard failed — D mapfrac {d_mapfrac:.3f} not << B {b_mapfrac:.3f}; projection didn't suppress map-subspace writing.")
        elif bd_m > 2 * bd_se:
            v = (f"MODE 2 CONFIRMED: the map ROUTES the surge. B>D on retention (B-D={bd_m:+.3f}>2SE {2*bd_se:.3f}) DESPITE D being "
                 f"non-destructive going in (D {d_postchild:.3f} >= B {b_postchild:.3f}, mapfrac D {d_mapfrac:.3f}<<B {b_mapfrac:.3f}). "
                 f"Phase-2 damage: A {dmg['A']:+.3f} B {dmg['B']:+.3f} C {dmg['C']:+.3f} D {dmg['D']:+.3f} — the map cuts depth damage. "
                 f"The traffic-guard knowing the corner changes how he routes the crowd, not just that he started calmer.")
        elif abs(bd_m) <= 2 * bd_se:
            v = (f"MODE 1 ONLY (claim narrows): B~D (B-D={bd_m:+.3f}, 2SE {2*bd_se:.3f}). B's whole edge was the NON-DESTRUCTIVE START, "
                 f"NOT surge-routing. A non-destructive no-map phase-1 slivers as well as the map -> 'the map routes the surge' does NOT "
                 f"hold beyond 'the map is a gentle phase-1'. D-vs-A={da_m:+.3f}. Phase-2 dmg A{dmg['A']:+.3f} B{dmg['B']:+.3f} D{dmg['D']:+.3f}.")
        else:
            v = f"D>B (B-D={bd_m:+.3f}): D's better start outweighed — inspect; Mode 2 not supported, Mode 1 dominates."
        rep["verdict"] = v
        rep["guards"] = {"nondestructive": bool(nondestructive), "nomap": bool(nomap), "trained": bool(trained),
                         "D_postchild": float(d_postchild), "B_postchild": float(b_postchild), "C_postchild": float(c_postchild),
                         "D_mapfrac": float(d_mapfrac), "B_mapfrac": float(b_mapfrac), "D_child_loss_drop": float(d_clo)}
        rep["paired"] = {"B_minus_D": [bd_m, bd_se], "D_minus_A": me(da)}
        rep["phase2_damage"] = dmg
        print(f"\n[2B] === VERDICT (held-out={heldout}, KPROJ={KPROJ}) ===\n  {v}", flush=True)
        print(f"  retention: A={a.mean():.3f} B={b.mean():.3f} C={c.mean():.3f} D={d.mean():.3f} (pre {pre_sst:.3f})", flush=True)
        print(f"  PAIRED B-D={bd_m:+.3f}±{bd_se:.3f} | D-A={da_m:+.3f}±{da_se:.3f}", flush=True)
        print(f"  GUARDS D: non-destructive post-child {d_postchild:.3f} (B {b_postchild:.3f}, C {c_postchild:.3f}) | "
              f"no-map mapfrac D {d_mapfrac:.3f} vs B {b_mapfrac:.3f} | trained drop {d_clo:.2f}", flush=True)
        print(f"  phase-2 damage (post_child - final): A {dmg['A']:+.3f} B {dmg['B']:+.3f} C {dmg['C']:+.3f} D {dmg['D']:+.3f}", flush=True)
        print(f"  task-B acc: A {np.mean(R['A']['taskB']):.3f} B {np.mean(R['B']['taskB']):.3f} C {np.mean(R['C']['taskB']):.3f} D {np.mean(R['D']['taskB']):.3f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[2B] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[2B] saved dev_experiment2b.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
