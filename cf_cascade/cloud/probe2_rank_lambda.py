"""PROBE 2 (clean rank ablation) — does map load-bearingness live in the tiny task-A curvature CORE (K_E~5-14) or in
the broad V shell? KILLS the vacuums confound: vacuums projected the childhood GRADIENT out of V per K, so childhood
learning (lossdrop) was NOT matched across K (6.66->15.41) -> rank vs learning-amount confounded. Here childhood is run
ONCE per seed (full), then the deposit is truncated to its top-k task-A-Fisher V-dims and RE-NORMALIZED to the full in-V
Frobenius magnitude (isolates rank from magnitude, per prereg). Identical childhood across all k => lossdrop identical by
construction => confound dead. Endpoint retention only (no per-step logging).

Per seed:
  childhood ONCE (full, from w_A) -> wB_full; per carrier dfull = wB_full[c] - w_A[c]; d_inV = P_V(dfull); M=||d_inV||_F;
  d_outV = dfull - d_inV.
  arm A    : phase-2 from w_A                                                 (no-deposit baseline; retA)
  arm B_k  : carriers = w_A + d_outV + (M/||P_{V[:k]}(dfull)||)*P_{V[:k]}(dfull); non-carriers = wB_full; phase-2  for k in KSET
             k=0   == arm D  (out-of-V deposit only, no in-V content)
             k=128 == arm B  (full deposit = wB_full exactly)  -> REPRODUCE-ANCHOR (B-A must hit banked within 0.03)
  arm SWAP : carriers = w_A + d_outV + M*(random rank-knot in V, unit Frob); non-carriers = wB_full; phase-2
             (curvature-matched random rank-knot, matched magnitude -> deposit-specificity AT the core rank)

Reads:
  B_k - A   : protection delivered by a rank-k (magnitude-matched) deposit. saturates by k~K_E => load lives in the core
              (monolithic-in-core); still climbing 64->128 => load in the broad shell OUTSIDE the core.
  B - B_k   : load LOST by truncating to rank k (= B128 - Bk).
  childhood lossdrop logged once/seed; identical across k (single childhood) -> the confound vacuums had is gone.
Env: CELL in {opt350m, pythia410m} ONLY (live band; dead at 1B -> HARD-STOP).
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT",
                   "anchor": {"B_minus_A": 0.089, "B_minus_D": 0.055}, "KSET": [0, 14, 32, 64, 128], "knot": 14},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA",
                   "anchor": {"B_minus_A": 0.073, "B_minus_D": -0.006}, "KSET": [0, 5, 9, 32, 64, 128], "knot": 9},
}
CELL = os.environ.get("CELL", "")
assert CELL in CELLS, f"CELL must be opt350m|pythia410m (live band ONLY; dead at 1B). got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe2_rank_{CELL}.json"; KSET = cfg["KSET"]; KNOT = cfg["knot"]


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
def childhood(model, tok, stream, steps, lr, seed):
    """full childhood (no projection). returns (early_loss, late_loss) for the lossdrop record."""
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    del opt; torch.cuda.empty_cache()
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
def phase2(model, tok, ho_tr, sst_eval, posid, negid, seed):
    """uninstrumented phase-2 (subj forgetting train) from the current state; returns task-A retention."""
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_DEPTH); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_DEPTH):
        idx = torch.randperm(len(ho_tr), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [ho_tr[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; r = eval_sst(model, tok, sst_eval, posid, negid); torch.cuda.empty_cache(); return r
def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[P2:{CELL}] {cfg['model']} clean rank ladder K={KSET} (knot={KNOT}); childhood-once -> confound dead", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)
    ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P2:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)

    # V = top-128 task-A Fisher per carrier (at w_A), descending eigval
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]  # (din, 128) double
    # fixed curvature-matched random rank-KNOT subspace in V (NOT deposit-derived), per carrier, unit Frobenius
    grng = torch.Generator(device="cpu").manual_seed(12345); S_swap = {}
    for n in carriers:
        din = V[n].shape[0]; dout = params[n].shape[0]
        cin = torch.randn(KPROJ, KNOT, generator=grng).double(); vin = (V[n].cpu() @ cin)            # (din, knot) in span(V)
        vin, _ = torch.linalg.qr(vin); vin = vin.to(dev)                                              # orthonormal cols
        vout = torch.randn(dout, KNOT, generator=grng).double().to(dev); vout, _ = torch.linalg.qr(vout)
        Sw = vout @ vin.T; S_swap[n] = (Sw / (Sw.norm() + 1e-30))                                     # (dout,din) unit Frob
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    print(f"[P2:{CELL}] V + swap(rank-{KNOT}) done ({time.time()-t0:.0f}s)", flush=True)

    ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    ret = {"A": []}; [ret.setdefault(f"k{k}", []) for k in KSET]; ret["swap"] = []
    lossdrops = []; Mnorm = {n: [] for n in carriers}  # record full in-V magnitude per carrier (diagnostic)

    def build_state(wB_full, dout_dep, inV_term):
        """state with non-carriers = wB_full, carriers = w_A + d_outV + inV_term[n]."""
        sd = {k: v.clone() for k, v in wB_full.items()}
        for n in carriers: sd[n] = (ckpt_sd[n].to(dev).double() + dout_dep[n] + inV_term[n]).float()
        return sd

    for sd_i in SEEDS:
        # --- childhood ONCE (full) ---
        model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd_i); lossdrops.append(l0 - l1)
        wB_full = {k: v.detach().to(dev).clone() for k, v in model.state_dict().items()}
        Pc = dict(model.named_parameters())
        dfull = {}; d_inV = {}; d_outV = {}; M = {}
        for n in carriers:
            df = (Pc[n].detach().double() - ckpt_sd[n].to(dev).double())           # (dout,din)
            dv = (df @ V[n]) @ V[n].T                                              # P_V(deposit) full in-V
            dfull[n] = df; d_inV[n] = dv; d_outV[n] = df - dv
            M[n] = float(dv.norm()); Mnorm[n].append(M[n])
        # arm A (no deposit) from w_A
        model.load_state_dict(ckpt_sd); ret["A"].append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd_i))
        # arm B_k for each rank k (magnitude-matched to full in-V)
        for k in KSET:
            if k == 0:
                inV_term = {n: torch.zeros_like(d_inV[n]) for n in carriers}        # arm D: out-of-V only
            else:
                inV_term = {}
                for n in carriers:
                    dk = (dfull[n] @ V[n][:, :k]) @ V[n][:, :k].T                    # P_{V[:k]}(deposit)
                    nk = float(dk.norm()) + 1e-30
                    inV_term[n] = dk * (M[n] / nk)                                   # renorm to full in-V magnitude
            model.load_state_dict(build_state(wB_full, d_outV, inV_term))
            ret[f"k{k}"].append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd_i))
        # arm SWAP (curvature-matched random rank-knot, matched magnitude) at the knot
        inV_swap = {n: S_swap[n] * M[n] for n in carriers}
        model.load_state_dict(build_state(wB_full, d_outV, inV_swap)); ret["swap"].append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd_i))
        del wB_full; torch.cuda.empty_cache()
        if sd_i % 5 == 0 or sd_i == SEEDS[-1]:
            msg = " ".join(f"k{k}={ret[f'k{k}'][-1]:.3f}" for k in KSET)
            print(f"[P2:{CELL}] seed {sd_i}: A={ret['A'][-1]:.3f} {msg} swap={ret['swap'][-1]:.3f} ld={lossdrops[-1]:.2f} ({time.time()-t0:.0f}s)", flush=True)

    ns = len(SEEDS); A = np.array(ret["A"])
    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    B128 = np.array(ret["k128"]); D0 = np.array(ret["k0"])
    BmA = me(B128 - A); BmD = me(B128 - D0)
    anchor_ok = (abs(BmA[0] - cfg["anchor"]["B_minus_A"]) < 0.03)
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns,
           "w_A_acc": pre_sst, "KSET": KSET, "knot": KNOT, "retA": me(A),
           "retBk": {str(k): me(ret[f"k{k}"]) for k in KSET},
           "Bk_minus_A": {str(k): me((np.array(ret[f"k{k}"]) - A).tolist()) for k in KSET},
           "B_minus_Bk": {str(k): me((B128 - np.array(ret[f"k{k}"])).tolist()) for k in KSET},
           "swap_ret": me(ret["swap"]), "swap_minus_A": me((np.array(ret["swap"]) - A).tolist()),
           "B_minus_A_k128": BmA, "B_minus_D_k0": BmD, "anchor_expected": cfg["anchor"], "ANCHOR_OK": bool(anchor_ok),
           "childhood_lossdrop_mean": me(lossdrops), "note_lossdrop": "single childhood/seed -> identical across all k (confound dead)"}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[P2:{CELL}] === ANCHOR B-A(k128)={BmA[0]:+.3f}(exp {cfg['anchor']['B_minus_A']:+.3f}) -> {'OK' if anchor_ok else 'MISMATCH!!'} ===", flush=True)
    print(f"[P2:{CELL}] retA={rep['retA'][0]:.3f}  lossdrop={rep['childhood_lossdrop_mean'][0]:.2f} (identical across k)", flush=True)
    print(f"[P2:{CELL}] rank ladder (Bk - A; saturate by k~{KNOT}=core-monolithic, climb 64->128=broad-shell):", flush=True)
    for k in KSET:
        bka = rep["Bk_minus_A"][str(k)]; bmb = rep["B_minus_Bk"][str(k)]
        tag = " (=D,out-of-V only)" if k == 0 else (" (=B,full)" if k == 128 else ("  <-KNOT/core" if k == KNOT else ""))
        print(f"   k={k:4d}  Bk-A={bka[0]:+.3f}±{bka[1]:.3f}   B-Bk={bmb[0]:+.3f}{tag}", flush=True)
    print(f"[P2:{CELL}] swap(rank-{KNOT}, matched-mag) - A = {rep['swap_minus_A'][0]:+.3f}±{rep['swap_minus_A'][1]:.3f}"
          f"  vs real k{KNOT}-A={rep['Bk_minus_A'][str(KNOT)][0]:+.3f}  (real>>swap => core deposit-specific)", flush=True)
    print(f"[P2:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
