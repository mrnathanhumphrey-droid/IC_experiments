"""PYTHIA DISPOSITIVE — direct mirror of the OPT dispositive on Pythia-410m. Tests whether the cross-arch dissociation
(map-dominant OPT B−D=+0.055 sig vs non-destructiveness-dominant Pythia B−D≈+0.015 @7seeds) is REAL at matched power,
or a magnitude artifact between two underpowered estimates.

WHY: the OPT dispositive at N=30 reinstated rung-3's direction at half magnitude (B−D=+0.055 sig, MAP LOAD-BEARING on OPT).
Pythia's existing B−D≈+0.015 is from 7 seeds — the same underpowered regime that gave OPT three inconsistent 7-seed draws
(+0.118 / −0.003 / +0.021), all consistent with +0.055. The "substrates unify on #2b" inference was drawn from Pythia at
exactly the N now shown unreliable. This decides whether antagonism-controls-survival is a real cross-arch fact and gates the
framing of every downstream mechanism question (rung-4 M3, toggle redesign, second-spool).

DIRECT MIRROR of opt_dispositive_lambda.py — three swaps (model / pinned-w_A / carrier names); regime, V=task-A-Fisher-top-128,
forced-subj, stream, arms A/B/D, paired seeds, mapfrac guard, childhood-loss-drop coherence, save-before-readout — ALL identical.

PRE-COMMITTED FORK (mirrored for Pythia, keyed on OPT_REF=+0.055):
  HARD VOID: A_forget < 0.15 (regime drifted / no dynamic range).
  (1) B−D ≈ 0 AND CI excludes OPT's +0.055  -> CROSS-ARCH DISSOCIATION CONFIRMED at matched power (map-dominant OPT vs
      non-destructiveness-dominant Pythia is real). Antagonism-controls-survival reopens; rung-4 M3 / toggle have a real target.
  (2) B−D ≈ +0.04-0.06 overlapping OPT       -> SUBSTRATES UNIFY on a smaller-everywhere map effect. Cross-arch was a magnitude
      artifact between two underpowered estimates. rung-4 M3 / toggle die clean — no cross-arch effect to explain.
  (3) B−D ≈ +0.02-0.04 (borderline)          -> PARTIAL DISSOCIATION (gradient not binary). Most interesting; report CIs, no
      forced binary read. Also lands here if CI is too wide to exclude OR overlap OPT (still underpowered at 30).
  (4) B−D > +0.055 (CI above OPT)             -> UNEXPECTED (Pythia map MORE load-bearing than OPT). Flag and reopen.
Reproduce-anchor (soft, from #2b Pythia bank): expect A_forget~0.226, B−A~+0.126, D−A~+0.143 — reported vs bank as harness-sanity.
Mechanistic-coherence tell: childhood loss-drop B vs D — OPT showed D learns FASTER (13.54 vs 6.67). If the asymmetry is ABSENT
on Pythia, the substrates differ structurally and the cross-arch comparison is confounded by that, not just antagonism level.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"
OUT = f"{G}/pythia_dispositive.json"; NPZ = f"{G}/pythia_dispositive.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; FORGET_GATE = 0.15; OPT_REF = 0.055
# #2b Pythia reproduce-anchor bank (soft harness-sanity):
ANCHOR = {"A_forget": 0.226, "B_minus_A": 0.126, "D_minus_A": 0.143}


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
    if key == "newscat":
        return [(f"{(r.get('headline') or '').strip()}\nCategory:", " " + (r.get("category") or "").strip().lower()) for r in win(LD("heegyu/news-category-dataset")["train"]) if r.get("headline") and r.get("category")]
    raise ValueError(key)


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    Lm = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), Lm), pad, np.int64); lb = np.full((len(seqs), Lm), -100, np.int64); am = np.zeros((len(seqs), Lm), np.int64)
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
            Lm = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), Lm), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
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


def map_subspace_frac(params, ckpt_sd, carriers, V):
    num = 0.0; den = 0.0
    with torch.no_grad():
        for n in carriers:
            dw = (params[n].detach() - ckpt_sd[n].to(dev)).double(); proj = dw @ V[n].double()
            num += float((proj * proj).sum()); den += float((dw * dw).sum())
    return num / max(den, 1e-12)


def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[PDISP] PYTHIA DISPOSITIVE — 30-seed A/B/D retention, pinned w_A, forced subj (mirror of OPT)", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); print(f"[PDISP] loaded pinned w_A from {CKPT}", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[PDISP] pinned w_A SST acc={pre_sst:.3f} ({time.time()-t0:.0f}s)", flush=True)
    assert pre_sst > 0.65, f"w_A too weak ({pre_sst:.3f})"

    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].float().contiguous()
    model.load_state_dict(ckpt_sd)
    print(f"[PDISP] V basis done ({time.time()-t0:.0f}s)", flush=True)

    heldout = "subj"; ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in stream_keys}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
    print(f"[PDISP] held-out=subj FORCED; stream={list(stream)} ({time.time()-t0:.0f}s)", flush=True)

    R = {a: [] for a in ["A", "B", "D"]}; EX = {"mfB": [], "mfD": [], "clB": [], "clD": []}
    for sd in SEEDS:
        # A — no childhood
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["A"].append(eval_sst(model, tok, sst_eval, posid, negid))
        # B — real childhood
        model.load_state_dict(ckpt_sd); b0, b1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        EX["mfB"].append(map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)); EX["clB"].append(b0 - b1)
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["B"].append(eval_sst(model, tok, sst_eval, posid, negid))
        # D — non-destructive no-map
        model.load_state_dict(ckpt_sd); d0, d1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        EX["mfD"].append(map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)); EX["clD"].append(d0 - d1)
        train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd); R["D"].append(eval_sst(model, tok, sst_eval, posid, negid))
        if sd % 5 == 0 or sd == SEEDS[-1]:
            print(f"[PDISP] seed {sd}: A/B/D={R['A'][-1]:.3f}/{R['B'][-1]:.3f}/{R['D'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    A, B, D = np.array(R["A"]), np.array(R["B"]), np.array(R["D"])
    np.savez(NPZ, A=A, B=B, D=D, pre_sst=pre_sst, mfB=np.array(EX["mfB"]), mfD=np.array(EX["mfD"]))
    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5)]
    rep = {"model": MODEL, "heldout": heldout, "pre_sst": pre_sst, "n_seeds": len(SEEDS), "opt_ref_BD": OPT_REF,
           "regime": {"lr_depth": LR_DEPTH, "steps_depth": STEPS_DEPTH, "steps_child": STEPS_CHILD, "lr_child": LR_CHILD, "pinned_wA": True},
           "anchor_bank": ANCHOR,
           "retention": {"A": me(A), "B": me(B), "D": me(D)}, "guards": {"mapfrac_B": me(EX["mfB"]), "mapfrac_D": me(EX["mfD"]),
           "childB_drop": me(EX["clB"]), "childD_drop": me(EX["clD"])}}
    try:
        A_forget = pre_sst - A.mean()
        ba_m, ba_se = me((B - A).tolist()); da_m, da_se = me((D - A).tolist()); bd_m, bd_se = me((B - D).tolist())
        ba_pos = int((B - A > 0).sum()); da_pos = int((D - A > 0).sum()); bd_pos = int((B - D > 0).sum())
        n = len(SEEDS)
        rep["paired"] = {"B_minus_A": [ba_m, ba_se, f"{ba_pos}/{n}+"], "D_minus_A": [da_m, da_se, f"{da_pos}/{n}+"],
                         "B_minus_D": [bd_m, bd_se, f"{bd_pos}/{n}+"]}
        rep["A_forget"] = float(A_forget)
        bd_sig = bd_m > 2 * bd_se
        ci_lo, ci_hi = bd_m - 2 * bd_se, bd_m + 2 * bd_se
        excludes_opt = ci_hi < OPT_REF or ci_lo > OPT_REF
        overlaps_opt = ci_lo <= OPT_REF <= ci_hi
        if A_forget < FORGET_GATE:
            v = f"VOID — regime drifted: A_forget={A_forget:.3f} < {FORGET_GATE} (no dynamic range)."
        elif (not bd_sig) and ci_hi < OPT_REF:
            v = (f"CROSS-ARCH DISSOCIATION CONFIRMED at matched power: Pythia B−D={bd_m:+.3f}±{bd_se:.3f} ({bd_pos}/{n}+) ≈0, "
                 f"CI [{ci_lo:+.3f},{ci_hi:+.3f}] EXCLUDES OPT's +{OPT_REF:.3f}. Map-dominant OPT vs non-destructiveness-dominant "
                 f"Pythia is REAL. Antagonism-controls-survival reopens with a real target; rung-4 M3 / toggle have a cross-arch effect "
                 f"to predict against. The morning's 'substrates unify on #2b' is OVERTURNED at matched power.")
        elif bd_sig and overlaps_opt and bd_m >= 0.04:
            v = (f"SUBSTRATES UNIFY on a smaller-everywhere map effect: Pythia B−D={bd_m:+.3f}±{bd_se:.3f} ({bd_pos}/{n}+) SIG and "
                 f"CI [{ci_lo:+.3f},{ci_hi:+.3f}] OVERLAPS OPT's +{OPT_REF:.3f}. Cross-arch dissociation was a magnitude artifact between "
                 f"two underpowered estimates. Both archs show map-deposition protection at ~+0.05; rung-3's inversion was a "
                 f"both-substrates-underpowered illusion. rung-4 M3 / toggle die clean — no cross-arch effect to explain.")
        elif bd_m > OPT_REF and ci_lo > OPT_REF:
            v = (f"UNEXPECTED — Pythia map MORE load-bearing than OPT: B−D={bd_m:+.3f}±{bd_se:.3f} ({bd_pos}/{n}+), CI "
                 f"[{ci_lo:+.3f},{ci_hi:+.3f}] entirely ABOVE OPT's +{OPT_REF:.3f}. Opposite of the morning's framing. Flag and reopen substantively.")
        else:
            v = (f"PARTIAL DISSOCIATION / borderline: Pythia B−D={bd_m:+.3f}±{bd_se:.3f} ({bd_pos}/{n}+), CI [{ci_lo:+.3f},{ci_hi:+.3f}] "
                 f"vs OPT +{OPT_REF:.3f} (excludes_opt={excludes_opt}, overlaps_opt={overlaps_opt}). Both substrates have map components, "
                 f"OPT's larger; cross-arch is a gradient not a binary OR CI too wide to adjudicate at N={n}. Report magnitudes with CIs; "
                 f"NO forced binary read. Antagonism-controls-survival = partially right (controls magnitude not existence).")
        rep["verdict"] = v
        # reproduce-anchor soft check vs #2b bank
        anc = (f"anchor vs #2b bank: A_forget {A_forget:.3f} (bank {ANCHOR['A_forget']:.3f}), B−A {ba_m:+.3f} (bank "
               f"{ANCHOR['B_minus_A']:+.3f}), D−A {da_m:+.3f} (bank {ANCHOR['D_minus_A']:+.3f})")
        rep["anchor_check"] = anc
        print(f"\n[PDISP] === PYTHIA DISPOSITIVE VERDICT (N={n}, held-out=subj, pinned w_A {pre_sst:.3f}, OPT_REF +{OPT_REF:.3f}) ===\n  {v}", flush=True)
        print(f"  retention A/B/D = {A.mean():.3f}/{B.mean():.3f}/{D.mean():.3f} (pre {pre_sst:.3f}, A_forget {A_forget:.3f})", flush=True)
        print(f"  PAIRED B−A={ba_m:+.3f}±{ba_se:.3f} ({ba_pos}/{n}+) | D−A={da_m:+.3f}±{da_se:.3f} ({da_pos}/{n}+) | B−D={bd_m:+.3f}±{bd_se:.3f} ({bd_pos}/{n}+)", flush=True)
        print(f"  {anc}", flush=True)
        print(f"  guards: mapfrac B/D {np.mean(EX['mfB']):.3f}/{np.mean(EX['mfD']):.3f} | childhood loss-drop B/D {np.mean(EX['clB']):.2f}/{np.mean(EX['clD']):.2f} (OPT: D faster 13.54 vs 6.67)", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[PDISP] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[PDISP] saved pythia_dispositive.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
