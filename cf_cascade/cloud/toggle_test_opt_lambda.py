"""RUNG-3.5 / TOGGLE TEST (OPT-350m) — does ANTAGONISM control MAP-LOAD-BEARINGNESS?

Cross-rung hypothesis (rungs 2+3): rung-2 found OPT has NO antagonism (shared steep core, but cosine~0 not head-on);
rung-3 found OPT's map is LOAD-BEARING (B-D=+0.118). Pythia is the opposite on BOTH (antagonistic -0.23 AND map-inert).
UNIFYING HYP: antagonism erodes a map deposited in the shared core; its absence lets the map survive & route. So
antagonism-level is the DIAL that sets map-load-bearingness. This toggles the dial WITHIN OPT (architecture held fixed).

DESIGN CRUX: vary antagonism while holding SHARING fixed. KNOB = depth-phase gradient surgery. The map is deposited in
CHILDHOOD; DEPTH erodes-or-spares it, so we rotate the DEPTH task-B gradient. Per carrier, with V=task-A top-curv subspace
(the shared core) and aHat=normalized P_V g_A (task-A grad direction in core, at depth-start weights):
  g_par = (g_B @ V) @ V.T          # core component  -> its energy IS "sharing"
  g_perp = g_B - g_par             # off-core, untouched
  r_hat  = normalize(g_par - <g_par,aHat>aHat)   # residual direction within core
  g_par' = |g_par| * (c*aHat + sqrt(1-c^2)*r_hat)  # SAME norm -> sharing fixed; cosine-with-task-A = c
  g_B(c) = g_par' + g_perp
Preserves |g_par| (sharing) and g_perp (off-core); sets in-core cosine to EXACTLY c. Moves ONLY antagonism.
LEVELS c in {+0.20 aligned, 0.0 orthogonal (OPT-natural anchor), -0.23 head-on (Pythia-like)} = antagonism INCREASING right.

ARMS at each level L: A_L no-childhood->depth(L) (yank floor) / B_L real-childhood(map)->depth(L) / D_L non-destructive
no-map->depth(L). Childhood (B map, D projected) is LEVEL-INDEPENDENT (built once/seed, reused across levels); only DEPTH
differs by L. B-D per level = the map's load-bearing contribution as a function of antagonism.

PRE-COMMITTED: B-D DECREASES as antagonism increases -> large at +0.20/0.0 (~+0.118, map survives), ~0 at -0.23 (eroded).
D-A roughly antagonism-INDEPENDENT (non-destr never needed the map to survive). MANDATORY GUARD: achieved cosine ~= c AND
sharing preserved by the rotation; arm A_L must forget >=0.10 each level (dynamic range; +0.20 may CEILING -> flagged, the
core claim rests on orthogonal-vs-head-on, both forget). VOID if knob dirty (cos!=c or sharing not preserved).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "facebook/opt-350m"; AX = f"{G}/overlap_axis"
OUT = f"{G}/toggle_test_opt.json"; NPZ = f"{G}/toggle_test_opt.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7))
LEVELS = [0.20, 0.0, -0.23]  # antagonism INCREASING left->right (cosine decreasing)
STEPS_WA = 500; LR_WA = 5e-5; STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; GUARD_EVERY = 60


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


def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); losses = []; params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, labels = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); batch = [items[i] for i in idx]
        inp, lb, am = b_pairs(tok, batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_((gd @ Vproj[n]) @ Vproj[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))


def gA_dir(model, tok, params, carriers, V):
    """task-A grad direction in core V, per carrier (unit Frobenius), at CURRENT (depth-start) weights."""
    rows = sst_rows("train", 96, 123); items = [(f"{s}\nSentiment:", LAB[l]) for s, l in rows]
    model.zero_grad(set_to_none=True); inp, lb, am = b_pairs(tok, items)
    model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
    aHat = {}
    for n in carriers:
        Vn = V[n].double(); g = params[n].grad.detach().double()
        gpar = (g @ Vn) @ Vn.T; nrm = gpar.norm()
        aHat[n] = (gpar / nrm) if nrm > 1e-12 else None
    model.zero_grad(set_to_none=True)
    return aHat


def depth_toggle(model, tok, items, steps, lr, seed, V, carriers, aHat, c):
    """Depth-train task-B with per-carrier gradient surgery rotating the in-core component to cosine c with task-A."""
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    params = dict(model.named_parameters()); sq = float((1.0 - c * c) ** 0.5)
    gd_pre_cos = []; gd_post_cos = []; gd_share = []  # guard samples (median over carriers, sampled steps)
    for st in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        sample = (st % GUARD_EVERY == 0); pre_c = []; post_c = []; shr = []
        for n in carriers:
            a = aHat[n]
            if a is None: continue
            gd = params[n].grad; Vn = V[n].double(); g64 = gd.detach().double()
            gpar = (g64 @ Vn) @ Vn.T; gperp = g64 - gpar; npar = gpar.norm()
            if npar < 1e-12: continue
            if sample:
                gtot = g64.norm()
                pre_c.append(float((gpar * a).sum() / (npar + 1e-30)))  # natural cosine (pre-surgery)
                shr.append(float((npar * npar) / (gtot * gtot + 1e-30)))
            along = (gpar * a).sum() * a; gres = gpar - along; nres = gres.norm()
            if nres < 1e-12: continue  # already parallel -> can't rotate cleanly, leave as-is
            rhat = gres / nres; gpar2 = npar * (c * a + sq * rhat); new = gpar2 + gperp
            if sample: post_c.append(float((gpar2 * a).sum() / (gpar2.norm() + 1e-30)))
            gd.copy_(new.to(gd.dtype))
        if sample and pre_c:
            gd_pre_cos.append(float(np.median(pre_c))); gd_post_cos.append(float(np.median(post_c))); gd_share.append(float(np.median(shr)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return {"pre_cos": float(np.mean(gd_pre_cos)), "post_cos": float(np.mean(gd_post_cos)), "share": float(np.mean(gd_share))}


def map_subspace_frac(model, ckpt_sd, carriers, V):
    num = 0.0; den = 0.0; params = dict(model.named_parameters())
    with torch.no_grad():
        for n in carriers:
            dw = (params[n].detach() - ckpt_sd[n].to(dev)).double(); proj = dw @ V[n].double()
            num += float((proj * proj).sum()); den += float((dw * dw).sum())
    return num / max(den, 1e-12)


def labels_of(items): return sorted(set(c for _, c in items))
def first_ids(tok, lbls): return [tok(w, add_special_tokens=False)["input_ids"][0] for w in lbls]
def cpu_clone(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
def to_dev(sd): return {k: v.to(dev) for k, v in sd.items()}


def main():
    t0 = time.time(); print(f"[TG] loading {MODEL}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    print(f"[TG] {len(carriers)} carriers; levels={LEVELS}", flush=True)
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)

    # PHASE 0 — train task-A (SST) -> w_A
    print(f"[TG] training task-A (SST) {STEPS_WA} steps...", flush=True)
    train_single(model, tok, [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", 2000, 7)], STEPS_WA, LR_WA, 0)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[TG] w_A SST acc={pre_sst:.3f} ({time.time()-t0:.0f}s)", flush=True)
    assert pre_sst > 0.68, f"w_A too weak ({pre_sst:.3f})"

    # task-A curvature subspace V (the shared core)
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].float().contiguous()
    model.load_state_dict(ckpt_sd); params = dict(model.named_parameters())
    print(f"[TG] task-A curvature basis done ({time.time()-t0:.0f}s)", flush=True)

    # held-out depth task-B (subj, like rung-3) + stream
    cands = {"subj": load_jsonB("medium_subj"), "emotion": load_hf("emotion")}
    pick = {}
    for k, items in cands.items():
        tr, ev = items[:-200], items[-200:]; lids = first_ids(tok, labels_of(items))
        model.load_state_dict(ckpt_sd); train_single(model, tok, tr, STEPS_DEPTH, LR_DEPTH, 0)
        a_sst = eval_sst(model, tok, sst_eval, posid, negid); pick[k] = {"forget": pre_sst - a_sst, "taskB": eval_clf(model, tok, ev, lids)}
        print(f"[TG]   precheck {k}: forget={pick[k]['forget']:.3f} taskB={pick[k]['taskB']:.3f}", flush=True)
    ok = {k: v for k, v in pick.items() if v["forget"] >= 0.10 and v["taskB"] > 0.55}
    heldout = max(ok or pick, key=lambda k: pick[k]["forget"])
    ho_items = cands[heldout]; ho_tr, ho_ev = ho_items[:-200], ho_items[-200:]; ho_lids = first_ids(tok, labels_of(ho_items))
    print(f"[TG] HELD-OUT = {heldout} (forget {pick[heldout]['forget']:.3f})", flush=True)

    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", ("emotion" if heldout == "subj" else "subj")]
    stream = {sk: ((load_jsonB("low_agnews") if sk == "agnews" else (load_jsonB("medium_subj") if sk == "subj" else load_hf(sk))), None) for sk in stream_keys}
    for sk in stream: stream[sk] = (stream[sk][0], labels_of(stream[sk][0]))
    print(f"[TG] stream={list(stream)} ({time.time()-t0:.0f}s)", flush=True)

    LK = [str(c) for c in LEVELS]
    R = {a: {lk: {"sstA": [], "taskB": []} for lk in LK} for a in ["A", "B", "D"]}
    CH = {"B": {"post_child": [], "mapfrac": [], "cl0": [], "cl1": []}, "D": {"post_child": [], "mapfrac": [], "cl0": [], "cl1": []}}
    GD = {a: {lk: {"pre_cos": [], "post_cos": [], "share": []} for lk in LK} for a in ["A", "B", "D"]}

    for sd in SEEDS:
        # childhood states (LEVEL-INDEPENDENT) — build once per seed
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        CH["B"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); CH["B"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V)); CH["B"]["cl0"].append(l0); CH["B"]["cl1"].append(l1)
        sdB = cpu_clone(model)
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        CH["D"]["post_child"].append(eval_sst(model, tok, sst_eval, posid, negid)); CH["D"]["mapfrac"].append(map_subspace_frac(model, ckpt_sd, carriers, V)); CH["D"]["cl0"].append(l0); CH["D"]["cl1"].append(l1)
        sdD = cpu_clone(model)

        for c in LEVELS:
            lk = str(c)
            # arm A — no childhood
            model.load_state_dict(ckpt_sd); pp = dict(model.named_parameters()); aH = gA_dir(model, tok, pp, carriers, V)
            gstat = depth_toggle(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd, V, carriers, aH, c)
            R["A"][lk]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["A"][lk]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
            for kk in gstat: GD["A"][lk][kk].append(gstat[kk])
            # arm B — real childhood (map)
            model.load_state_dict(to_dev(sdB)); pp = dict(model.named_parameters()); aH = gA_dir(model, tok, pp, carriers, V)
            gstat = depth_toggle(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd, V, carriers, aH, c)
            R["B"][lk]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["B"][lk]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
            for kk in gstat: GD["B"][lk][kk].append(gstat[kk])
            # arm D — non-destructive no-map
            model.load_state_dict(to_dev(sdD)); pp = dict(model.named_parameters()); aH = gA_dir(model, tok, pp, carriers, V)
            gstat = depth_toggle(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd, V, carriers, aH, c)
            R["D"][lk]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid)); R["D"][lk]["taskB"].append(eval_clf(model, tok, ho_ev, ho_lids))
            for kk in gstat: GD["D"][lk][kk].append(gstat[kk])
            print(f"[TG] seed {sd} c={c:+.2f}: sstA A/B/D={R['A'][lk]['sstA'][-1]:.3f}/{R['B'][lk]['sstA'][-1]:.3f}/{R['D'][lk]['sstA'][-1]:.3f} "
                  f"| B-D={R['B'][lk]['sstA'][-1]-R['D'][lk]['sstA'][-1]:+.3f} | cos pre/post={GD['A'][lk]['pre_cos'][-1]:+.3f}/{GD['A'][lk]['post_cos'][-1]:+.3f} share={GD['A'][lk]['share'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"{a}_{lk}_{m}": np.array(R[a][lk][m]) for a in R for lk in LK for m in R[a][lk]},
             **{f"CH_{a}_{m}": np.array(CH[a][m]) for a in CH for m in CH[a]},
             **{f"GD_{a}_{lk}_{m}": np.array(GD[a][lk][m]) for a in GD for lk in LK for m in GD[a][lk]},
             pre_sst=pre_sst, levels=np.array(LEVELS), heldout=np.array([heldout]))

    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std() / max(len(x), 1) ** 0.5)]
    rep = {"model": MODEL, "heldout": heldout, "pre_sst": pre_sst, "levels": LEVELS, "precheck": pick}
    rep["childhood"] = {a: {m: me(CH[a][m]) for m in CH[a]} for a in CH}
    per_level = {}
    try:
        bd_by_level = []
        for c in LEVELS:
            lk = str(c)
            a = np.array(R["A"][lk]["sstA"]); b = np.array(R["B"][lk]["sstA"]); d = np.array(R["D"][lk]["sstA"])
            bd = me(b - d); da = me(d - a); ba = me(b - a)
            forget = pre_sst - a.mean()
            guard_cos = me(GD["A"][lk]["post_cos"]); guard_pre = me(GD["A"][lk]["pre_cos"]); guard_share = me(GD["A"][lk]["share"])
            per_level[lk] = {"c": c, "ret_A": me(a.tolist()), "ret_B": me(b.tolist()), "ret_D": me(d.tolist()),
                             "B_minus_D": bd, "D_minus_A": da, "B_minus_A": ba, "A_forget": float(forget),
                             "taskB_A": me(R["A"][lk]["taskB"]), "taskB_B": me(R["B"][lk]["taskB"]), "taskB_D": me(R["D"][lk]["taskB"]),
                             "guard_post_cos": guard_cos, "guard_pre_cos": guard_pre, "guard_share": guard_share}
            bd_by_level.append((c, bd[0], bd[1], float(forget)))
        rep["per_level"] = per_level
        # GUARD: knob bit (|post_cos - c| small) and sharing preserved/stable
        cos_ok = all(abs(per_level[str(c)]["guard_post_cos"][0] - c) < 0.05 for c in LEVELS)
        shares = [per_level[str(c)]["guard_share"][0] for c in LEVELS]
        share_ok = (max(shares) - min(shares)) < 0.5 * (np.mean(shares) + 1e-9)  # rough stability across levels
        # ordered by antagonism INCREASING (cosine decreasing): LEVELS already +0.20,0.0,-0.23
        bds = [per_level[str(c)]["B_minus_D"][0] for c in LEVELS]
        ses = [per_level[str(c)]["B_minus_D"][1] for c in LEVELS]
        forgets = [per_level[str(c)]["A_forget"] for c in LEVELS]
        decreasing = bds[0] >= bds[1] >= bds[2]  # +0.20 >= 0.0 >= -0.23
        endpoint_sep = (bds[1] - bds[2]) > 2 * ((ses[1] ** 2 + ses[2] ** 2) ** 0.5)  # orthogonal vs head-on (both forget)
        ceiling_flag = forgets[0] < 0.10  # aligned may not forget
        if not cos_ok:
            v = f"VOID — knob did not bite (post_cos {[round(per_level[str(c)]['guard_post_cos'][0],3) for c in LEVELS]} vs targets {LEVELS})."
        elif not share_ok:
            v = f"VOID — sharing drifted across levels ({[round(s,3) for s in shares]}) — manipulation confounded."
        elif endpoint_sep and decreasing:
            v = (f"HYPOTHESIS CONFIRMED: B-D DECREASES with antagonism, sharing fixed (cos achieved, share stable). "
                 f"B-D by level {[round(x,3) for x in bds]} for c={LEVELS} -> map load-bearing at orthogonal, eroded at head-on. "
                 f"Rungs 2+3 collapse into ONE law: antagonism is the dial setting map-survival.{' (aligned CEILING-flagged, core claim = orthogonal vs head-on.)' if ceiling_flag else ''}")
        elif endpoint_sep:
            v = f"PARTIAL: orthogonal vs head-on separate ({bds[1]:.3f} vs {bds[2]:.3f}) but NOT monotone across all 3 (bds {[round(x,3) for x in bds]}) — antagonism matters, shape non-monotone. Inspect."
        elif (max(bds) - min(bds)) < 2 * max(ses):
            v = f"CLEAN KILL: B-D FLAT across antagonism ({[round(x,3) for x in bds]}) — antagonism does NOT control map-survival; rung-2/3 divergences are INDEPENDENT facts. Unifying hypothesis dead."
        else:
            v = f"MIXED: B-D {[round(x,3) for x in bds]} (SE {[round(x,3) for x in ses]}) — varies but not cleanly with antagonism. Report what it tracks."
        rep["verdict"] = v
        rep["guards"] = {"cos_ok": bool(cos_ok), "share_ok": bool(share_ok), "shares": shares, "ceiling_flag_aligned": bool(ceiling_flag)}
        print(f"\n[TG] === TOGGLE VERDICT (OPT-350m, held-out={heldout}) ===\n  {v}", flush=True)
        for c in LEVELS:
            lk = str(c); p = per_level[lk]
            print(f"  c={c:+.2f}: ret A/B/D={p['ret_A'][0]:.3f}/{p['ret_B'][0]:.3f}/{p['ret_D'][0]:.3f} | "
                  f"B-D={p['B_minus_D'][0]:+.3f}±{p['B_minus_D'][1]:.3f} | D-A={p['D_minus_A'][0]:+.3f}±{p['D_minus_A'][1]:.3f} | "
                  f"A_forget={p['A_forget']:.3f} | cos {p['guard_pre_cos'][0]:+.3f}->{p['guard_post_cos'][0]:+.3f} share={p['guard_share'][0]:.3f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[TG] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[TG] saved toggle_test_opt.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
