"""OPT DISPOSITIVE — close the retention layer: is OPT protection genuinely NULL or noise-limited at 7 seeds?

rung-3's B−A +0.160 / B−D +0.118 (7/7) did NOT reproduce (3 OPT runs: +0.118/−0.003/+0.021; re-run B−A +0.027 n.s.) —
per-seed retention SD ~0.10 swamps ~0.02-0.15 effects. Retraction landed but bank says "noise-dominated at 7 seeds", leaving
open whether a small real effect (~0.03-0.05) exists below 7-seed resolution. This closes it dispositively at N=30 (SE~0.018;
+0.05 ≈ 2.8 SE detectable, +0.02 ≈ 1.1 SE borderline).

Pinned regime (no drift, no auto-pick): LOAD att-3's saved w_A (opt350m_sst2_wA.pt, acc 0.823, the regime that passed the
reproduce-anchor A_forget 0.296), held-out FORCED = subj, att-3 depth params (lr 5e-5 / 250 steps — NB the spec said 2e-5/200 but
att-3 ACTUALLY ran 5e-5/250; using the params that produced A_forget 0.296). Arms A/B/D only (C/map question settled inert ×3 runs).

PRE-COMMITTED FORK: B−A & D−A both ≥+0.05 (≥~2.8SE) → REAL but rung-3-inflated (rescope: OPT protects, smaller than reported).
both <+0.03 n.s. → GENUINELY NULL at any resolvable N (retraction clean: geometry/M2 replicate, retention protection Pythia-specific).
one clears not other → asymmetric partial dissociation. 0.03-0.05 → borderline, report CIs. HARD-STOP: A_forget<0.15 → VOID (regime drifted).
Per-seed sign fractions reported (real small effect → most seeds same-signed; null → ~50/50 around 0).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
MODEL = "facebook/opt-350m"; CKPT = f"{G}/opt350m_sst2_wA.pt"
OUT = f"{G}/opt_dispositive.json"; NPZ = f"{G}/opt_dispositive.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; FORGET_GATE = 0.15


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
    t0 = time.time(); print(f"[DISP] OPT DISPOSITIVE — 30-seed A/B/D retention, pinned w_A, forced subj", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); print(f"[DISP] loaded pinned w_A from {CKPT}", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[DISP] pinned w_A SST acc={pre_sst:.3f} ({time.time()-t0:.0f}s)", flush=True)
    assert pre_sst > 0.68, f"w_A too weak ({pre_sst:.3f})"

    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].float().contiguous()
    model.load_state_dict(ckpt_sd)
    print(f"[DISP] V basis done ({time.time()-t0:.0f}s)", flush=True)

    heldout = "subj"; ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in stream_keys}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
    print(f"[DISP] held-out=subj FORCED; stream={list(stream)} ({time.time()-t0:.0f}s)", flush=True)

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
            print(f"[DISP] seed {sd}: A/B/D={R['A'][-1]:.3f}/{R['B'][-1]:.3f}/{R['D'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    A, B, D = np.array(R["A"]), np.array(R["B"]), np.array(R["D"])
    np.savez(NPZ, A=A, B=B, D=D, pre_sst=pre_sst, mfB=np.array(EX["mfB"]), mfD=np.array(EX["mfD"]))
    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5)]
    rep = {"model": MODEL, "heldout": heldout, "pre_sst": pre_sst, "n_seeds": len(SEEDS),
           "regime": {"lr_depth": LR_DEPTH, "steps_depth": STEPS_DEPTH, "steps_child": STEPS_CHILD, "lr_child": LR_CHILD, "pinned_wA": True},
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
        ba_sig = ba_m > 2 * ba_se; da_sig = da_m > 2 * da_se
        if A_forget < FORGET_GATE:
            v = f"VOID — regime drifted: A_forget={A_forget:.3f} < {FORGET_GATE}."
        elif (ba_m >= 0.05 and ba_sig) and (da_m >= 0.05 and da_sig):
            v = (f"OPT PROTECTION REAL but rung-3-INFLATED: B−A={ba_m:+.3f}±{ba_se:.3f} ({ba_pos}/{n}+), D−A={da_m:+.3f}±{da_se:.3f} "
                 f"({da_pos}/{n}+) both clear at N={n}. Rescope: OPT protects like Pythia at SMALLER magnitude than rung-3's +0.16.")
        elif (ba_m < 0.03 and not ba_sig) and (da_m < 0.03 and not da_sig):
            v = (f"OPT PROTECTION GENUINELY NULL at N={n}: B−A={ba_m:+.3f}±{ba_se:.3f} ({ba_pos}/{n}+), D−A={da_m:+.3f}±{da_se:.3f} "
                 f"({da_pos}/{n}+) both n.s. Retraction CLEAN: geometry/M2 replicate, retention protection is Pythia-specific at any affordable N.")
        elif ba_sig != da_sig:
            v = (f"ASYMMETRIC partial dissociation: B−A={ba_m:+.3f}({'sig' if ba_sig else 'ns'}), D−A={da_m:+.3f}({'sig' if da_sig else 'ns'}) "
                 f"— only one half of the protection layer replicates on OPT.")
        else:
            v = (f"BORDERLINE (0.03-0.05 / ~2SE): B−A={ba_m:+.3f}±{ba_se:.3f}, D−A={da_m:+.3f}±{da_se:.3f} — real but tiny, "
                 f"smaller than 7-seed apparent magnitudes. Report CIs.")
        rep["verdict"] = v
        print(f"\n[DISP] === OPT DISPOSITIVE VERDICT (N={n}, held-out=subj, pinned w_A {pre_sst:.3f}) ===\n  {v}", flush=True)
        print(f"  retention A/B/D = {A.mean():.3f}/{B.mean():.3f}/{D.mean():.3f} (pre {pre_sst:.3f}, A_forget {A_forget:.3f})", flush=True)
        print(f"  PAIRED B−A={ba_m:+.3f}±{ba_se:.3f} ({ba_pos}/{n}+) | D−A={da_m:+.3f}±{da_se:.3f} ({da_pos}/{n}+) | B−D={bd_m:+.3f}±{bd_se:.3f} ({bd_pos}/{n}+)", flush=True)
        print(f"  guards: mapfrac B/D {np.mean(EX['mfB']):.3f}/{np.mean(EX['mfD']):.3f} | childhood loss-drop B/D {np.mean(EX['clB']):.2f}/{np.mean(EX['clD']):.2f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[DISP] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[DISP] saved opt_dispositive.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
