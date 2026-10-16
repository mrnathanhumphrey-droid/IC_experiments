"""#2 THE DEVELOPMENTAL EXPERIMENT — does a network that had a CHILDHOOD sliver where one that didn't yanks?

Claim under test: catastrophic forgetting = the missing childhood. A net given a BREADTH phase first (low-res survey of the
task-neighborhood, master none, build the map) should, when then DEPTH-trained on a new in-neighborhood task-B, retain task-A
(SST-2 sentiment) BETTER than a net that went straight to depth — because the childhood map lets new depth SLIVER (find its slot)
instead of YANK (overwrite). In-neighborhood scope ONLY (#1: topic/subjectivity/emotion reachable; STEM/code/conflict disconnected).

THREE ARMS (the scrambled-childhood control is the whole experiment):
  A no-childhood       : w_A -> depth-train held-out task-B                              -> baseline yank (floor)
  B real-childhood     : w_A -> breadth pass over neighborhood stream (master none) -> depth-train SAME task-B  -> sliver?
  C scrambled-childhood: w_A -> breadth pass over SAME stream, LABELS SHUFFLED (structure destroyed, compute matched)
                                -> depth-train SAME task-B                            -> is the benefit the MAP or just compute?

DESIGN GUARDS baked in:
  * CHILDHOOD STREAM EXCLUDES SENTIMENT datasets (yelp/imdb/tweet) — task-A is sentiment, so they'd be task-A REPLAY (CLS),
    not map-building. Stream = NON-sentiment in-neighborhood survey (agnews/dbpedia/yahoo/newscat + the non-held-out of subj/emotion).
  * HELD-OUT task-B is IN-neighborhood (subj or emotion, #1-confirmed reachable), NOT in the stream, and pre-checked to forget >=0.10.
  * MATCHED COMPUTE B vs C: identical steps/LR/exposure; ONLY difference = real vs shuffled labels in the breadth phase.
  * CHILDHOOD IS BREADTH NOT DEPTH: shallow, master none — report childhood loss (real should drop modestly & stay >> converged;
    scrambled can't fit noise, stays high) + post-childhood task-A acc (childhood shouldn't itself wreck A).
TWO METRICS/arm (sliver is two-sided): task-A retention (SST acc after depth) AND task-B accuracy (childhood must not STARVE B).
PRE-COMMITTED: B>A retention AND B>C AND task-B not starved -> childhood makes depth SLIVER, it's the MAP -> claim LANDS (in-nbhd).
  B~A -> frame wrong at its own experiment (clean negative). B>A but B~C -> compute not map (DND-adjacent). B>A,B>C but B starves task-B -> rigidity not sliver.
ROBUSTNESS: 7 PAIRED seeds (judge B-A & B-C on paired error-bar separation, not point means); save all arrays before readout.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/dev_experiment.json"; NPZ = f"{G}/dev_experiment.npz"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16


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
    """generic next-token-label accuracy: argmax over label first-token ids vs true completion first-token."""
    model.eval(); correct = 0; n = 0; lid = torch.tensor(label_ids).to(dev)
    with torch.no_grad():
        for i in range(0, len(eval_items), 32):
            chunk = eval_items[i:i + 32]; seqs = [tok(p, add_special_tokens=False)["input_ids"][:256] for p, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            predpos = last[:, lid].argmax(1).cpu().numpy()  # index into label_ids
            for j, (_, co) in enumerate(chunk):
                true_id = tok(co, add_special_tokens=False)["input_ids"][0]
                correct += int(label_ids[predpos[j]] == true_id); n += 1
    return correct / max(n, 1)


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed); last = float("nan")
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); last = float(loss.item())
    return last


def childhood(model, tok, stream, steps, lr, seed, scramble):
    """breadth pass over the stream (master none). stream: {task:(items,label_strs)}. scramble -> random label per example."""
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(1000 + seed); tasks = list(stream); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, labels = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); batch = [items[i] for i in idx]
        if scramble: batch = [(p, labels[rng.integers(len(labels))]) for (p, _) in batch]  # structure destroyed, compute matched
        inp, lb, am = b_pairs(tok, batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))


def labels_of(items): return sorted(set(c for _, c in items))
def first_ids(tok, lbls): return [tok(w, add_special_tokens=False)["input_ids"][0] for w in lbls]


def main():
    t0 = time.time(); print("[DEV] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[DEV] pre SST-2 acc={pre_sst:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # candidate held-out depth tasks (in-neighborhood, NOT sentiment)
    cands = {"subj": load_jsonB("medium_subj"), "emotion": load_hf("emotion")}
    for k in cands: print(f"[DEV]   cand {k} n={len(cands[k])}", flush=True)

    # PRE-CHECK: pick held-out = forgets >=0.10 in arm-A AND learns
    pick = {}
    for k, items in cands.items():
        tr, ev = items[:-200], items[-200:]; lids = first_ids(tok, labels_of(items))
        model.load_state_dict(ckpt_sd); train_single(model, tok, tr, STEPS_DEPTH, LR_DEPTH, 0)
        a_sst = eval_sst(model, tok, sst_eval, posid, negid); b_acc = eval_clf(model, tok, ev, lids)
        pick[k] = {"forget": pre_sst - a_sst, "taskB": b_acc}
        print(f"[DEV]   precheck {k}: forget={pre_sst-a_sst:.3f} taskB_acc={b_acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    ok = {k: v for k, v in pick.items() if v["forget"] >= 0.10 and v["taskB"] > 0.55}
    heldout = max(ok or pick, key=lambda k: pick[k]["forget"])
    print(f"[DEV] HELD-OUT depth task-B = {heldout} (forget {pick[heldout]['forget']:.3f}, taskB {pick[heldout]['taskB']:.3f})", flush=True)
    ho_items = cands[heldout]; ho_tr, ho_ev = ho_items[:-200], ho_items[-200:]; ho_lids = first_ids(tok, labels_of(ho_items))

    # childhood stream = NON-sentiment in-neighborhood survey, EXCLUDING the held-out
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", ("emotion" if heldout == "subj" else "subj")]
    stream = {}
    for sk in stream_keys:
        try:
            it = load_jsonB("low_agnews") if sk == "agnews" else (load_jsonB("medium_subj") if sk == "subj" else load_hf(sk))
            stream[sk] = (it, labels_of(it)); print(f"[DEV]   stream +{sk} n={len(it)} ({len(labels_of(it))} labels)", flush=True)
        except Exception as e:
            print(f"[DEV]   stream x{sk} FAILED: {str(e)[:90]}", flush=True)
    assert len(stream) >= 4, f"stream too small: {list(stream)}"
    print(f"[DEV] stream={list(stream)} (NON-sentiment; held-out={heldout} excluded) ({time.time()-t0:.0f}s)", flush=True)

    R = {a: {"sstA": [], "taskB": [], "post_child_sst": [], "child_l0": [], "child_l1": []} for a in ["A", "B", "C"]}
    for sd in SEEDS:
        # Arm A — no childhood
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["A"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["A"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        # Arm B — real childhood -> depth
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, scramble=False)
        R["B"]["post_child_sst"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["child_l0"].append(l0); R["B"]["child_l1"].append(l1)
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["B"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        # Arm C — scrambled childhood -> depth
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, scramble=True)
        R["C"]["post_child_sst"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["child_l0"].append(l0); R["C"]["child_l1"].append(l1)
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["C"]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["C"]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
        print(f"[DEV] seed {sd}: A_sstA={R['A']['sstA'][-1]:.3f} B_sstA={R['B']['sstA'][-1]:.3f} C_sstA={R['C']['sstA'][-1]:.3f} | "
              f"taskB A/B/C={R['A']['taskB'][-1]:.2f}/{R['B']['taskB'][-1]:.2f}/{R['C']['taskB'][-1]:.2f} | "
              f"postchild B/C={R['B']['post_child_sst'][-1]:.3f}/{R['C']['post_child_sst'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"{a}_{m}": np.array(R[a][m]) for a in R for m in R[a]}, pre_sst=pre_sst, heldout=np.array([heldout]))
    A = {a: {m: [float(np.mean(R[a][m])), float(np.std(R[a][m]) / len(R[a][m]) ** 0.5)] for m in R[a] if R[a][m]} for a in R}
    rep = {"heldout": heldout, "stream": list(stream), "pre_sst": pre_sst, "precheck": pick, "agg": A,
           "config": {"STEPS_DEPTH": STEPS_DEPTH, "LR_DEPTH": LR_DEPTH, "STEPS_CHILD": STEPS_CHILD, "LR_CHILD": LR_CHILD, "seeds": len(SEEDS)}}
    try:
        a, b, c = np.array(R["A"]["sstA"]), np.array(R["B"]["sstA"]), np.array(R["C"]["sstA"])
        ba, bc = b - a, b - c  # PAIRED (same seed)
        ba_m, ba_se = float(ba.mean()), float(ba.std() / len(ba) ** 0.5)
        bc_m, bc_se = float(bc.mean()), float(bc.std() / len(bc) ** 0.5)
        floor_forgets = (pre_sst - a.mean()) >= 0.10
        bt, at = np.array(R["B"]["taskB"]).mean(), np.array(R["A"]["taskB"]).mean()
        starved = bt < at - 0.05
        child_drop_real = float(np.mean(R["B"]["child_l0"]) - np.mean(R["B"]["child_l1"]))
        child_drop_scr = float(np.mean(R["C"]["child_l0"]) - np.mean(R["C"]["child_l1"]))
        b_gt_a = ba_m > 2 * ba_se; b_gt_c = bc_m > 2 * bc_se
        if not floor_forgets:
            v = f"VOID: arm A doesn't forget (pre {pre_sst:.3f} -> A {a.mean():.3f}, drop {pre_sst-a.mean():.3f} <0.10) — no dynamic range."
        elif b_gt_a and b_gt_c and not starved:
            v = (f"CLAIM LANDS (in-neighborhood): childhood RETAINS task-A better (B-A={ba_m:+.3f}>2SE {2*ba_se:.3f}) AND it's the MAP not "
                 f"compute (B-C={bc_m:+.3f}>2SE {2*bc_se:.3f}) AND task-B not starved (B {bt:.2f} vs A {at:.2f}). The childhood makes depth SLIVER.")
        elif not b_gt_a:
            v = (f"CLEAN NEGATIVE (the big one): childhood does NOT help even in-neighborhood (B-A={ba_m:+.3f}, 2SE {2*ba_se:.3f}). "
                 f"The frame fails at its own experiment.")
        elif b_gt_a and not b_gt_c:
            v = (f"COMPUTE NOT MAP (DND-adjacent): B>A (B-A={ba_m:+.3f}) BUT B~C (B-C={bc_m:+.3f}, 2SE {2*bc_se:.3f}) — the benefit is "
                 f"exposure/compute, the MAP did no work. Map-specificity dies.")
        elif starved:
            v = (f"RIGIDITY NOT SLIVER: B>A (B-A={ba_m:+.3f}) & B>C but task-B STARVED (B {bt:.2f} < A {at:.2f}) — childhood retains A by "
                 f"refusing to learn B, not by slivering.")
        else:
            v = f"MIXED: B-A={ba_m:+.3f} B-C={bc_m:+.3f} starved={starved} — inspect per-seed."
        rep["verdict"] = v; rep["paired"] = {"B_minus_A": [ba_m, ba_se], "B_minus_C": [bc_m, bc_se], "starved": bool(starved),
                                             "child_loss_drop_real": child_drop_real, "child_loss_drop_scrambled": child_drop_scr}
        print(f"\n[DEV] === VERDICT (held-out={heldout}) ===\n  {v}", flush=True)
        print(f"  task-A retention: A={a.mean():.3f} B={b.mean():.3f} C={c.mean():.3f} (pre {pre_sst:.3f})", flush=True)
        print(f"  PAIRED B-A={ba_m:+.3f}±{ba_se:.3f} | B-C={bc_m:+.3f}±{bc_se:.3f}", flush=True)
        print(f"  task-B acc: A={at:.3f} B={bt:.3f} C={np.array(R['C']['taskB']).mean():.3f} (starved={starved})", flush=True)
        print(f"  post-childhood SST (childhood's own effect on A): B={np.mean(R['B']['post_child_sst']):.3f} C={np.mean(R['C']['post_child_sst']):.3f}", flush=True)
        print(f"  childhood loss drop (master-none/structure check): real={child_drop_real:+.3f} scrambled={child_drop_scr:+.3f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[DEV] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[DEV] saved dev_experiment.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
