"""PROBE 7 — suppression vs reorganization (Probe-6 confound-kill). Pre-reg PREREG_PROBE7_REORG_2026_06_25.md (LOCK 7bdfcbc6).
Probe-6 falsified pinning (childhood protects task-A but ends FARTHER from w_A in V). Confound: CH trains 400 extra steps,
so displacement-from-w_A is step-count-inflated. FIX: measure PHASE-2-ONLY displacement from each arm's pre-phase-2 anchor.
Per seed, paired off w_A:  NoCH: w_A -> phase-2 (anchor=w_A).   CH: w_A -> childhood -> [anchor=w_postCH] -> phase-2.
V = top-128 task-A Fisher at w_A (the Probe-5 damage subspace), SAME for both arms.
  D2 = mean_carrier ||P_V(w_end - w_anchor)||  (phase-2-only in-V displacement)
  forget = ret_anchor - ret_end             (task-A forgetting from phase-2)
  robust = forget / max(D2, eps)            (forgetting per unit movement = reorganization quantity)
Contrasts: dD2=D2_noch-D2_ch (>0 suppression); dforget (>0 protection); drobust=robust_noch-robust_ch (>0 reorganization).
HARD-STOP (VOID) on dataset load fail or w_A anchor off-band; flag childhood_nondestructive.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100))   # floor go-forward: N>=100; paired contrasts cancel the ~0.02 floor
LR_DEPTH = 5e-5; STEPS_P2 = 250; LR_CHILD = 3e-5; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; N_FISHER = 200
EPS = 1e-6; HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02; NONDESTRUCT_TOL = 0.05
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe7_displacement_{CELL}.json"


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
def childhood(model, tok, stream, steps, lr, seed, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} childhood {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache(); return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
def train_phase(model, tok, items, steps, lr, seed, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache()
def labels_of(items): return sorted(set(c for _, c in items))


def disp(params, anchor_sd, V, carriers):
    """Per-carrier in-V displacement ||P_V(w_end - w_anchor)|| averaged over carriers. anchor_sd: cpu state dict."""
    s = 0.0
    with torch.no_grad():
        for n in carriers:
            df = (params[n].detach().double() - anchor_sd[n].to(dev).double())
            dv = (df @ V[n]) @ V[n].T; s += float(dv.norm())
    return s / len(carriers)


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P7:{CELL}] {cfg['model']} suppression-vs-reorganization (phase-2-only displacement), N={len(SEEDS)}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P7:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    exp = ANCHOR_WA[cfg["family"]]
    if abs(pre_sst - exp) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off: {pre_sst:.3f} vs {exp:.3f}+-{ANCHOR_TOL}"}, open(OUT, "w"), indent=2)
        print(f"[P7:{CELL}] HARD-STOP: w_A anchor off-band -> VOID, no run.", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    try:
        ho_tr = load_jsonB("medium_subj")[:-200]; assert len(ho_tr) > 100
        stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
        stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
        for sk, (it, _) in stream.items(): assert len(it) > 50, sk
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P7:{CELL}] HARD-STOP: datasets failed to load -> VOID, no run. {e}", flush=True); return
    print(f"[P7:{CELL}] datasets loaded (phase-2 subj {len(ho_tr)}; childhood {len(stream)} streams)", flush=True)

    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    D2n = []; D2c = []; fn = []; fc = []; rbn = []; rbc = []; retA_ch = []; rend_n = []; rend_c = []
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"s{si+1}/{ns}"
        # --- NoCH: anchor = w_A ---
        model.load_state_dict(ckpt_sd)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} NoCH_p2")
        d2 = disp(dict(model.named_parameters()), ckpt_sd, V, carriers); re = eval_sst(model, tok, sst_eval, posid, negid)
        D2n.append(d2); rend_n.append(re); fgn = pre_sst - re; fn.append(fgn); rbn.append(fgn / max(d2, EPS))
        # --- CH: childhood -> anchor=w_postCH -> phase-2 ---
        model.load_state_dict(ckpt_sd)
        childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, hb=f"{lab} CH_child")
        anchor_ch = snap(); ra = eval_sst(model, tok, sst_eval, posid, negid); retA_ch.append(ra)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} CH_p2")
        d2 = disp(dict(model.named_parameters()), anchor_ch, V, carriers); re = eval_sst(model, tok, sst_eval, posid, negid)
        D2c.append(d2); rend_c.append(re); fgc = ra - re; fc.append(fgc); rbc.append(fgc / max(d2, EPS))
        dD2 = float(np.mean(np.array(D2n) - np.array(D2c))); dfg = float(np.mean(np.array(fn) - np.array(fc))); drb = float(np.mean(np.array(rbn) - np.array(rbc)))
        print(f"[seed] {lab} DONE  D2_noch={D2n[-1]:.4f} D2_ch={D2c[-1]:.4f} fgt_n={fn[-1]:+.3f} fgt_c={fc[-1]:+.3f} retA_ch={ra:.3f}  run(dD2)={dD2:+.4f} run(dforget)={dfg:+.4f} run(drobust)={drb:+.4f}  ({time.time()-_T0:.0f}s)", flush=True)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    D2nA = np.array(D2n); D2cA = np.array(D2c); fnA = np.array(fn); fcA = np.array(fc)
    r_ch = float(np.corrcoef(D2cA, fcA)[0, 1]) if ns > 2 and D2cA.std() > 0 and fcA.std() > 0 else 0.0
    nondestruct = bool(np.mean(retA_ch) >= pre_sst - NONDESTRUCT_TOL)
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst, "VOID": False,
           "D2_noch": me(D2n), "D2_ch": me(D2c), "delta_D2": me((D2nA - D2cA).tolist()),
           "forget_noch": me(fn), "forget_ch": me(fc), "delta_forget": me((fnA - fcA).tolist()),
           "robust_noch": me(rbn), "robust_ch": me(rbc), "delta_robust": me((np.array(rbn) - np.array(rbc)).tolist()),
           "ret_anchor_ch": me(retA_ch), "ret_end_noch": me(rend_n), "ret_end_ch": me(rend_c),
           "childhood_nondestructive": nondestruct, "corr_D2ch_forgetch": r_ch,
           "SUPPRESSION_if": "delta_D2>0 (>=2sigma) AND delta_robust~=0",
           "REORG_if": "delta_forget>0 (>=2sigma) AND delta_robust>0 (>=2sigma) AND delta_D2<=0",
           "MIXED_if": "delta_D2>0 AND delta_robust>0"}
    json.dump(rep, open(OUT, "w"), indent=2)
    g = lambda k: (rep[k][0], rep[k][1], rep[k][0] / max(rep[k][1], 1e-9))
    for k in ("delta_D2", "delta_forget", "delta_robust"):
        m, e, z = g(k); print(f"[P7:{CELL}] {k} = {m:+.4f} +- {e:.4f} ({z:+.1f}sigma)", flush=True)
    print(f"[P7:{CELL}] D2_noch={rep['D2_noch'][0]:.4f} D2_ch={rep['D2_ch'][0]:.4f}  robust_noch={rep['robust_noch'][0]:.4f} robust_ch={rep['robust_ch'][0]:.4f}  r_ch={r_ch:+.3f}  nondestruct={nondestruct}", flush=True)
    print(f"[P7:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
