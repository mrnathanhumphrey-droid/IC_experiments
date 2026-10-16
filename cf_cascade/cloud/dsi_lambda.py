"""DSI — Deposit-Surge Interaction (fly-through, probe 1). Per-step mechanism of map load-bearingness, cross-arch, at
the live band. PRE-REG (parallel-agent). Env: CELL in {opt350m, pythia410m} ONLY (HARD-STOP otherwise; dead at 1B).

Reproduces the dispositive regime EXACTLY (lr_child 3e-5/400, lr_depth 5e-5/250, pinned w_A, held-out subj, N=30 paired
seeds, V=top-128 task-A Fisher) -> must reproduce banked B-A/B-D (OPT +0.089/+0.055; Pythia +0.073/-0.006) within noise,
else harness invalid. Then instruments phase-2 (the task-B/subj forgetting train) per-step, per-(layer,carrier):

  #1 cos(g_t, S_dep)        surge-deposit alignment        (Frobenius; S_dep unit)
  #2 ||P_V(g_t)||^2/||g_t||^2  surge in-V fraction
  #3 <w_t - w_A, S_dep>      deposit survival (norm to t=0)
  #4 task-A damage localized in S_dep / V\\S_dep / complement  (leave-component-out loss at checkpoints)
  + ||g_t|| per arm/step (MAGNITUDE-MATCH gating: compare cos/frac, never raw; a B<A surge-norm drop IS a deflect signal)
  + SWAP control (built-in): cos(g_t, S_swap) with S_swap = curvature-matched random rank-1 in V (NOT deposit-derived).

Arms: A (no deposit, start w_A) vs B (deposit, start wB); D (deposit proj-out of V) for the B-D anchor + absorb/deflect ref.
Per-step logging on A & B (the deposit contrast). Signatures: DEFLECT/ABSORB/ERASE/PASS-THROUGH per pre-reg.
Stores per-(arm,step,carrier) mean/std over seeds -> dsi_<cell>.npz; retention anchor + summary -> dsi_<cell>.json.
"""
import os, time, json, traceback, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; CKPTS = [125, 250]   # phase-2 damage-attribution checkpoints (gated to every 5th seed)
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",        "ck": OPT_C, "family": "OPT",
                   "anchor": {"B_minus_A": 0.089, "B_minus_D": 0.055}},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt", "ck": PY_C, "family": "PYTHIA",
                   "anchor": {"B_minus_A": 0.073, "B_minus_D": -0.006}},
}
CELL = os.environ.get("CELL", "")
assert CELL in CELLS, f"CELL must be opt350m|pythia410m (live band ONLY; dead at 1B). got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/dsi_{CELL}.json"; NPZ = f"{G}/dsi_{CELL}.npz"


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
def sst_loss(model, tok, items):
    model.eval(); tot = 0.0; nb = 0
    with torch.no_grad():
        for i in range(0, len(items), 32):
            ch = items[i:i + 32]; inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l]) for s, l in ch])
            tot += float(model(input_ids=inp, attention_mask=am, labels=lb).loss.item()); nb += 1
    return tot / max(nb, 1)
def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_((gd @ Vproj[n]) @ Vproj[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()
def snap(model, carriers): p = dict(model.named_parameters()); return {n: p[n].detach().clone() for n in carriers}
def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[DSI:{CELL}] {cfg['model']} band-ONLY; reproduce dispositive then per-step DSI", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); sst_dmg = sst_rows("validation", 128, 1)   # held-out for damage loss
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[DSI:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)

    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).float().contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    # SWAP: fixed curvature-matched random rank-1 in V per carrier (uniform combo of top-128 dirs; NOT deposit-derived)
    grng = torch.Generator(device="cpu").manual_seed(12345); S_swap = {}
    for n in carriers:
        din = V[n].shape[0]; dout = params[n].shape[0]
        cin = torch.randn(KPROJ, 1, generator=grng); vin = (V[n].cpu().double() @ cin.double()); vin = (vin / vin.norm()).to(dev)
        vout = torch.randn(dout, 1, generator=grng).double().to(dev); vout = vout / vout.norm()
        S_swap[n] = (vout @ vin.T).float()   # (dout,din) unit Frobenius, input-dir in span(V)
    print(f"[DSI:{CELL}] V + swap done ({time.time()-t0:.0f}s)", flush=True)

    ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    # accumulators: per arm x step x carrier  -> mean over seeds. ARMS: A(no deposit) B(real deposit) Bswap(SWAP control)
    ARMS = ["A", "B", "Bswap"]; T = STEPS_DEPTH
    acc = {a: {q: np.zeros((T, NC)) for q in ["cos_dep", "cos_swap", "inV", "gnorm", "surv"]} for a in ARMS}
    acc2 = {a: {q: np.zeros((T, NC)) for q in ["cos_dep", "cos_swap", "inV", "gnorm", "surv"]} for a in ARMS}
    ret = {a: [] for a in ["A", "B", "Bswap", "D"]}

    def phase2_logged(arm, dep_dir, surv0, seed):
        """instrumented phase-2 from the CURRENT model state (caller sets it). dep_dir = the deposit direction this arm
        tracks (S_dep for A/B; S_swap for the SWAP control arm). cos_swap always vs the fixed curvature-matched S_swap.
        NON-PERTURBING: only no_grad reads. w_A ref for survival = ckpt_sd. returns retention."""
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR_DEPTH); g = torch.Generator().manual_seed(seed)
        P = dict(model.named_parameters())
        for step in range(T):
            idx = torch.randperm(len(ho_tr), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [ho_tr[i] for i in idx])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            with torch.no_grad():
                for ci, n in enumerate(carriers):
                    gt = P[n].grad.detach().double(); gn = float((gt * gt).sum()) ** 0.5 + 1e-30
                    acc[arm]["gnorm"][step, ci] += gn; acc2[arm]["gnorm"][step, ci] += gn * gn
                    cdep = float((gt * dep_dir[n].double()).sum()) / gn
                    cswp = float((gt * S_swap[n].double()).sum()) / gn
                    gV = gt @ V[n].double(); inv = float((gV * gV).sum()) / (gn * gn)
                    for q, val in [("cos_dep", cdep), ("cos_swap", cswp), ("inV", inv)]:
                        acc[arm][q][step, ci] += val; acc2[arm][q][step, ci] += val * val
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            with torch.no_grad():
                for ci, n in enumerate(carriers):
                    dw = (P[n].detach() - ckpt_sd[n]).double()
                    sv = float((dw * dep_dir[n].double()).sum()) / (surv0[n] + 1e-30)
                    acc[arm]["surv"][step, ci] += sv; acc2[arm]["surv"][step, ci] += sv * sv
        del opt
        return eval_sst(model, tok, sst_eval, posid, negid)

    for sd in SEEDS:
        # --- B childhood (full) -> S_dep + FULL post-childhood state (dispositive method) ---
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        wB_full = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # FULL state, not carriers-only
        Pc = dict(model.named_parameters()); S_dep = {}; surv0 = {}
        for n in carriers:
            dfull = (Pc[n].detach() - ckpt_sd[n]).double(); d = (dfull @ V[n].double()) @ V[n].double().T   # P_V(deposit)
            nrm = float((d * d).sum()) ** 0.5 + 1e-30; S_dep[n] = (d / nrm).float()
            surv0[n] = float((dfull * S_dep[n].double()).sum())   # = ||d|| (t=0 survival ref)
        # arm A: phase-2 from w_A (full); arm B: phase-2 from full post-childhood state
        model.load_state_dict(ckpt_sd); ret["A"].append(phase2_logged("A", S_dep, surv0, sd))
        model.load_state_dict(wB_full); ret["B"].append(phase2_logged("B", S_dep, surv0, sd))
        # arm Bswap (SWAP CONTROL): inject a curvature-matched random deposit (S_swap) of MATCHED in-V magnitude
        # (surv0[n]) into carriers only; phase-2 from there. If it reproduces B's absorb/deflect -> generic curvature
        # routing (deposit-specific reading dies); if not -> the real deposit's interaction is deposit-specific.
        model.load_state_dict(ckpt_sd)
        with torch.no_grad():
            Ps = dict(model.named_parameters())
            for n in carriers: Ps[n].add_((surv0[n] * S_swap[n]).to(Ps[n].dtype))
        ret["Bswap"].append(phase2_logged("Bswap", S_swap, surv0, sd))
        # arm D: Vproj childhood (full) then phase-2 directly from the full D-state (dispositive method)
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        opt = torch.optim.AdamW(model.parameters(), lr=LR_DEPTH); gg = torch.Generator().manual_seed(sd)
        for _ in range(T):
            idx = torch.randperm(len(ho_tr), generator=gg)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [ho_tr[i] for i in idx])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        del opt; ret["D"].append(eval_sst(model, tok, sst_eval, posid, negid)); torch.cuda.empty_cache()
        if sd % 5 == 0 or sd == SEEDS[-1]:
            print(f"[DSI:{CELL}] seed {sd}: ret A/B/D={ret['A'][-1]:.3f}/{ret['B'][-1]:.3f}/{ret['D'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- reproduce-anchor ----
    A, B, D = np.array(ret["A"]), np.array(ret["B"]), np.array(ret["D"]); ns = len(SEEDS)
    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / ns ** 0.5)]
    BmA = me(B - A); BmD = me(B - D)
    anchor_ok = (abs(BmA[0] - cfg["anchor"]["B_minus_A"]) < 0.03 and abs(BmD[0] - cfg["anchor"]["B_minus_D"]) < 0.03)

    # ---- per-step mean/std over seeds, carrier-median trajectories ----
    def traj(arm, q):
        m = acc[arm][q] / ns; return m  # (T,NC)
    out_npz = {}
    for a in ARMS:
        for q in ["cos_dep", "cos_swap", "inV", "gnorm", "surv"]:
            out_npz[f"{a}__{q}"] = acc[a][q] / ns      # (T, NC) mean over seeds
    out_npz["carriers"] = np.array(carriers)
    np.savez(NPZ, **out_npz)

    # carrier-median per-step summary (for the verdict read)
    def cmed(a, q): return np.median(acc[a][q] / ns, axis=1)  # (T,)
    summ = {}
    for a in ARMS:
        summ[a] = {q: cmed(a, q).tolist() for q in ["cos_dep", "cos_swap", "inV", "gnorm", "surv"]}

    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns,
           "w_A_acc": pre_sst, "retention": {"A": me(A), "B": me(B), "D": me(D), "Bswap": me(ret["Bswap"]),
                                             "B_minus_A": BmA, "B_minus_D": BmD},
           "anchor_expected": cfg["anchor"], "ANCHOR_OK": bool(anchor_ok),
           "per_step_carrier_median": summ}
    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"\n[DSI:{CELL}] === ANCHOR B-A={BmA[0]:+.3f}(exp {cfg['anchor']['B_minus_A']:+.3f}) "
          f"B-D={BmD[0]:+.3f}(exp {cfg['anchor']['B_minus_D']:+.3f}) -> {'OK' if anchor_ok else 'MISMATCH!!'} ===", flush=True)
    for a in ARMS:
        cd = summ[a]["cos_dep"]; iv = summ[a]["inV"]; sv = summ[a]["surv"]; gn = summ[a]["gnorm"]
        print(f"  arm {a:5s}: cos_dep {cd[0]:+.3f}->{cd[-1]:+.3f} | inV {iv[0]:.3f}->{iv[-1]:.3f} | surv {sv[0]:.2f}->{sv[-1]:.2f} | gnorm {gn[0]:.3f}->{gn[-1]:.3f}", flush=True)
    # SWAP-GATE read: real-deposit (B) vs swap-deposit (Bswap) in-V late; if equal -> generic curvature, if B<Bswap -> deposit-specific
    print(f"  SWAP-GATE inV late  B={summ['B']['inV'][-1]:.3f}  Bswap={summ['Bswap']['inV'][-1]:.3f}  A={summ['A']['inV'][-1]:.3f}"
          f"  (B<Bswap => real deposit deflects MORE than random => deposit-specific)", flush=True)
    print(f"saved {OUT} + {NPZ} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
