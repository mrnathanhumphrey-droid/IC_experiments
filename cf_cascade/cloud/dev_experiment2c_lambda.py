"""#2c — WHY does non-destructive warm-up protect task-A? BASIN-FLATTENING (M1) vs PRE-MOVED-CAPACITY (M2).

#2b survivor: a non-destructive prior phase (even map-free, arm D) leaves task-A ~3x robust to depth (D-A +0.144). Black box.
Two candidate mechanisms, distinguishable with two measurements in a COMMON frame = task-A's top-curvature subspace V at w_A:

M1 (flattening test): task-A curvature IN V, at w_A vs at D's post-childhood weights w_D. Recompute task-A Fisher at each point,
   project onto V (directions fixed at w_A), compare mean eigen-energy.
     basin-flattening -> curvature DROPS at w_D (flatter basin, same surge perturbs less).
     pre-moved-capacity -> curvature UNCHANGED (D trained the complement, didn't touch V).
M2 (capacity/redirection test): phase-2 depth (subj) for arm A (from w_A) vs arm D (from w_D); decompose the phase-2 weight-change
   Δw into V vs the complement.
     pre-moved-capacity -> D's Δw lands LESS in V, MORE in the exercised complement (the surge has a sink). Redirection.
     basin-flattening -> D's Δw lands in the SAME place as A's; task-A survives because V is flatter. Attenuation, not redirection.

These are NOT circular: forgetting ~ Σ_V λ_i (Δw·v_i)^2 = (curvature in V) × (movement in V). M1 = the λ factor, M2 = the Δw factor.
Same damage-drop can come from either; reading BOTH lands the cell.

2x2 VERDICT (joint, corroboration-over-scalar — do NOT call off one axis):
  curv DROPS + surge REDIRECTED  -> MIXED (both levers live)
  curv DROPS + surge SAME-dest    -> BASIN FLATTENING (clean) -> bridge to flat-minima/SAM
  curv UNCHANGED + surge REDIRECTED -> PRE-MOVED CAPACITY (clean) -> buildable sink-intervention (pre-exercise the drain, not GPM)
  curv UNCHANGED + surge SAME-dest  -> NEITHER (reopen; would be surprising given top-curvature parsimony)
Primary basis = top-16 (the curved/fragile core where forgetting lives); top-128 reported too. Reuse #2b setup (subj, non-sentiment
stream, projected childhood). Per-seed, save before readout. Guard: D mapfrac~0.018 reproduces (D didn't write V) — so an M1 drop
DESPITE that = a pure landscape effect (interesting), consistent flag.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/dev_experiment2c.json"; NPZ = f"{G}/dev_experiment2c.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(5))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; KFRAG = 16; N_FISHER = 200


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


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_((gd @ Vproj[n]) @ Vproj[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def dw_frac(model, ref_sd, carriers, V):
    num = 0.0; den = 0.0; params = dict(model.named_parameters())
    with torch.no_grad():
        for n in carriers:
            dw = (params[n].detach() - ref_sd[n].to(dev)).double(); proj = dw @ V[n].double()
            num += float((proj * proj).sum()); den += float((dw * dw).sum())
    return num / max(den, 1e-12)


def curv_in(model, params, carriers, tok, V16, V128):
    """mean over carriers of mean diag(V^T F V), F=task-A fisher at current weights. Returns (c16, c128)."""
    FA = fisher_taskA(model, params, carriers, tok); v16 = []; v128 = []
    for n in carriers:
        F = FA[n]; A = V16[n].double(); B = V128[n].double()
        v16.append(float((A.T @ F @ A).diagonal().mean())); v128.append(float((B.T @ F @ B).diagonal().mean()))
    return float(np.mean(v16)), float(np.mean(v128))


def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print("[2C] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); pre_sst = eval_sst(model, tok, sst_eval, posid, negid)

    # task-A curvature basis at w_A (the COMMON frame)
    FA0 = fisher_taskA(model, params, carriers, tok)
    V16 = {}; V128 = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA0[n]); V128[n] = vv[:, -KPROJ:].float().contiguous(); V16[n] = vv[:, -KFRAG:].float().contiguous()
    curvA_16 = float(np.mean([float((V16[n].double().T @ FA0[n] @ V16[n].double()).diagonal().mean()) for n in carriers]))
    curvA_128 = float(np.mean([float((V128[n].double().T @ FA0[n] @ V128[n].double()).diagonal().mean()) for n in carriers]))
    print(f"[2C] pre SST {pre_sst:.3f} | w_A curvature in V: top16={curvA_16:.4e} top128={curvA_128:.4e} ({time.time()-t0:.0f}s)", flush=True)

    heldout = "subj"; ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {sk: ((load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)), None) for sk in stream_keys}
    print(f"[2C] stream={list(stream)} held-out={heldout} ({time.time()-t0:.0f}s)", flush=True)

    R = {k: [] for k in ["curvD16", "curvD128", "curvB16", "curvB128", "mapfracD",
                          "fracV16_A", "fracV128_A", "fracV16_D", "fracV128_D", "sstA_finalA", "sstA_finalD",
                          "sst_postchildD"]}
    for sd in SEEDS:
        # --- arm D childhood (projected, non-destructive no-map) -> w_D ; M1 at w_D
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V128, carriers=carriers)
        params = dict(model.named_parameters())
        R["mapfracD"].append(dw_frac(model, ckpt_sd, carriers, V128))
        R["sst_postchildD"].append(eval_sst(model, tok, sst_eval, posid, negid))
        c16, c128 = curv_in(model, params, carriers, tok, V16, V128); R["curvD16"].append(c16); R["curvD128"].append(c128)
        wD_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # --- arm B childhood (real map) -> w_B ; M1 at w_B (contrast)
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        params = dict(model.named_parameters())
        c16, c128 = curv_in(model, params, carriers, tok, V16, V128); R["curvB16"].append(c16); R["curvB128"].append(c128)
        # --- M2 arm A: depth from w_A, decompose Δw
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["fracV16_A"].append(dw_frac(model, ckpt_sd, carriers, V16)); R["fracV128_A"].append(dw_frac(model, ckpt_sd, carriers, V128))
        R["sstA_finalA"].append(eval_sst(model, tok, sst_eval, posid, negid))
        # --- M2 arm D: depth from w_D, decompose Δw (vs w_D, in w_A-defined V)
        model.load_state_dict(wD_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        R["fracV16_D"].append(dw_frac(model, wD_sd, carriers, V16)); R["fracV128_D"].append(dw_frac(model, wD_sd, carriers, V128))
        R["sstA_finalD"].append(eval_sst(model, tok, sst_eval, posid, negid))
        print(f"[2C] seed {sd}: curvD16/A={R['curvD16'][-1]/curvA_16:.3f} curvB16/A={R['curvB16'][-1]/curvA_16:.3f} | "
              f"fracV16 A={R['fracV16_A'][-1]:.4f} D={R['fracV16_D'][-1]:.4f} | sstA A={R['sstA_finalA'][-1]:.3f} D={R['sstA_finalD'][-1]:.3f} "
              f"| mapfracD={R['mapfracD'][-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, curvA_16=curvA_16, curvA_128=curvA_128, **{k: np.array(v) for k, v in R.items()})
    def ms(x): x = np.array(x, float); return [float(x.mean()), float(x.std() / max(len(x), 1) ** 0.5)]
    rep = {"heldout": heldout, "pre_sst": pre_sst, "curvA_16": curvA_16, "curvA_128": curvA_128, "KFRAG": KFRAG, "KPROJ": KPROJ,
           "agg": {k: ms(v) for k, v in R.items()}}
    try:
        cD16 = np.array(R["curvD16"]); fA16 = np.array(R["fracV16_A"]); fD16 = np.array(R["fracV16_D"])
        ratio_curvD = float(cD16.mean() / curvA_16)                 # M1: curvature at w_D relative to w_A (in V16)
        ratio_curvB = float(np.array(R["curvB16"]).mean() / curvA_16)
        # M2: paired frac difference (D vs A) in V16
        df = fA16 - fD16; df_m = float(df.mean()); df_se = float(df.std() / max(len(df), 1) ** 0.5)
        redirected = df_m > 2 * df_se and df_m > 0          # D lands LESS in V than A (clearing 2SE)
        flattened = ratio_curvD < 0.85                       # curvature dropped meaningfully at w_D
        protect = float(np.array(R["sstA_finalD"]).mean() - np.array(R["sstA_finalA"]).mean())
        if flattened and redirected:
            cell = "MIXED — both levers live: curvature DROPS (ratio {:.2f}) AND surge REDIRECTED (D lands {:.4f} in V16 vs A {:.4f}, Δ{:+.4f}>2SE). Report relative contribution.".format(ratio_curvD, fD16.mean(), fA16.mean(), df_m)
        elif flattened and not redirected:
            cell = "BASIN FLATTENING (clean): curvature DROPS at w_D (ratio {:.2f}<0.85) AND surge SAME destination (D {:.4f} ~ A {:.4f} in V16, Δ{:+.4f} n.s.). Forgetting-robustness is a FLATNESS phenomenon -> bridge to flat-minima/SAM.".format(ratio_curvD, fD16.mean(), fA16.mean(), df_m)
        elif (not flattened) and redirected:
            cell = "PRE-MOVED CAPACITY (clean): curvature UNCHANGED at w_D (ratio {:.2f}~1) AND surge REDIRECTED into the complement (D {:.4f} < A {:.4f} in V16, Δ{:+.4f}>2SE). The exercised complement is a SINK -> buildable intervention (pre-exercise the drain, not GPM).".format(ratio_curvD, fD16.mean(), fA16.mean(), df_m)
        else:
            cell = "NEITHER (reopen): curvature UNCHANGED (ratio {:.2f}) AND surge SAME destination (Δ{:+.4f} n.s.) yet task-A protected (Δsst {:+.3f}) — mechanism is outside M1/M2. Surprising given top-curvature parsimony.".format(ratio_curvD, df_m, protect)
        rep["verdict"] = cell
        rep["M1"] = {"curvD16_over_A": ratio_curvD, "curvB16_over_A": ratio_curvB, "flattened": bool(flattened),
                     "curvD128_over_A": float(np.array(R["curvD128"]).mean() / curvA_128)}
        rep["M2"] = {"fracV16_A": ms(R["fracV16_A"]), "fracV16_D": ms(R["fracV16_D"]), "paired_A_minus_D": [df_m, df_se], "redirected": bool(redirected),
                     "fracV128_A": ms(R["fracV128_A"]), "fracV128_D": ms(R["fracV128_D"])}
        rep["protection_reproduced"] = protect; rep["mapfracD"] = ms(R["mapfracD"])
        print(f"\n[2C] === 2x2 VERDICT (held-out={heldout}, V=top{KFRAG}) ===\n  {cell}", flush=True)
        print(f"  M1 curvature ratio (in V16): D/A={ratio_curvD:.3f} B/A={ratio_curvB:.3f} (flattened if <0.85)", flush=True)
        print(f"  M2 phase-2 Δw frac in V16: A={fA16.mean():.4f} D={fD16.mean():.4f} (paired A-D={df_m:+.4f}±{df_se:.4f}; redirected if >0 & >2SE)", flush=True)
        print(f"  protection reproduced (D-A final SST): {protect:+.3f} | D mapfrac {np.mean(R['mapfracD']):.4f} (D didn't write V)", flush=True)
        print(f"  [top128] curvD128/A={np.array(R['curvD128']).mean()/curvA_128:.3f} | fracV128 A={np.array(R['fracV128_A']).mean():.4f} D={np.array(R['fracV128_D']).mean():.4f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[2C] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[2C] saved dev_experiment2c.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
