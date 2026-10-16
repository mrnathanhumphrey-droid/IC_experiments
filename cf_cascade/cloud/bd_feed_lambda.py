"""B-D FEED — per-model dispositive (A/B/D, N=30) producing model-level B-D + per-carrier resid, for the RMD-SRC panel.
Parameterized by CELL env var. One cell per Lambda instance (fire 7 in parallel). Generalizes opt_m3 across the matrix+depth cells.

Outputs bd_<cell>.{json,npz}: retention A/B/D + B-A/B-D/D-A + A_forget + mapfrac + per-carrier resid_B-D (the panel inputs).
Anchor softened (no prior B-D ref for new cells): VOID only if A_forget < 0.15 (no dynamic range). Big cells (opt1_3b/pythia1b)
use gradient_checkpointing to fit 40GB. Depth cells load config.num_hidden_layers then load_state_dict.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; K_M = 16; K_GEN = 16; CHANCE = K_M / KPROJ; FORGET_GATE = 0.15
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt125m":    {"model": "facebook/opt-125m",      "ckpt": "opt125m_sst2_wA.pt",    "ck": OPT_C, "family": "OPT",    "size_m": 125},
    "opt1_3b":    {"model": "facebook/opt-1.3b",       "ckpt": "opt1_3b_sst2_wA.pt",    "ck": OPT_C, "family": "OPT",    "size_m": 1300, "big": True},
    "pythia160m": {"model": "EleutherAI/pythia-160m",  "ckpt": "pythia160m_sst2_wA.pt", "ck": PY_C,  "family": "PYTHIA", "size_m": 160},
    "pythia1b":   {"model": "EleutherAI/pythia-1b",    "ckpt": "pythia1b_sst2_wA.pt",   "ck": PY_C,  "family": "PYTHIA", "size_m": 1000, "big": True},
    "opt_d6":     {"model": "facebook/opt-350m",       "ckpt": "opt350m_d6_sst2_wA.pt",  "ck": OPT_C, "family": "OPT", "size_m": 350, "n_layers": 6,  "depth": 6},
    "opt_d12":    {"model": "facebook/opt-350m",       "ckpt": "opt350m_d12_sst2_wA.pt", "ck": OPT_C, "family": "OPT", "size_m": 350, "n_layers": 12, "depth": 12},
    "opt_d18":    {"model": "facebook/opt-350m",       "ckpt": "opt350m_d18_sst2_wA.pt", "ck": OPT_C, "family": "OPT", "size_m": 350, "n_layers": 18, "depth": 18},
    "pythia160m_step16k": {"model": "EleutherAI/pythia-160m", "ckpt": "pythia160m_step16k_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia160m_step50k": {"model": "EleutherAI/pythia-160m", "ckpt": "pythia160m_step50k_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia160m_step8k":  {"model": "EleutherAI/pythia-160m", "ckpt": "pythia160m_step8k_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia160m_dedup":   {"model": "EleutherAI/pythia-160m-deduped", "ckpt": "pythia160m_dedup_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 160},
    "pythia410m_dedup":   {"model": "EleutherAI/pythia-410m-deduped", "ckpt": "pythia410m_dedup_sst2_wA.pt", "ck": PY_C, "family": "PYTHIA", "size_m": 410},
}
CELL = os.environ.get("CELL", "")
assert CELL in CELLS, f"CELL env must be one of {list(CELLS)}; got {CELL!r}"
cfg = CELLS[CELL]
BIG = bool(cfg.get("big"))  # big cells (opt1_3b/pythia1b): CPU ckpt + foreach=False AdamW to fit 40GB.
# NOTE: foreach=False is the SAME AdamW update (per-tensor vs fused kernel; fp diffs ~1e-7) applied IDENTICALLY to A/B/D
# arms -> common-mode, cannot bias the within-model B-D contrast. Childhood/deposit/projection regime is UNCHANGED.
OUT = f"{G}/bd_{CELL}.json"; NPZ = f"{G}/bd_{CELL}.npz"


def _retry(fn, tries=5, wait=12):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e: last = e; print(f"[hf] retry {i+1}: {str(e)[:100]}", flush=True); time.sleep(wait)
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


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=(False if BIG else None)); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt


def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=(False if BIG else None)); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); losses = []; params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_((gd @ Vproj[n]) @ Vproj[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    del opt
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))


def map_subspace_frac(params, ckpt_sd, carriers, V):
    num = 0.0; den = 0.0
    with torch.no_grad():
        for n in carriers:
            dw = (params[n].detach() - ckpt_sd[n].to(dev)).double(); proj = dw @ V[n].double()
            num += float((proj * proj).sum()); den += float((dw * dw).sum())
    return num / max(den, 1e-12)


def map_dirs(wB, ckpt_sd, carriers, V):
    M = {}
    with torch.no_grad():
        for n in carriers:
            dwB = (wB[n].detach() - ckpt_sd[n].to(dev)).double(); coords = dwB @ V[n].double()
            U, S, Vh = torch.linalg.svd(coords, full_matrices=False); M[n] = Vh[:K_M].T.contiguous()
    return M


def surge_pcs(post, start_sd, carriers, V, k):
    Gd = {}
    with torch.no_grad():
        for n in carriers:
            dw = (post[n].detach() - start_sd[n].to(dev)).double(); coords = dw @ V[n].double()
            U, S, Vh = torch.linalg.svd(coords, full_matrices=False); Gd[n] = Vh[:k].T.contiguous()
    return Gd


def residualize(M, Gd, eps=1e-2):
    R = {}
    for n in M:
        P = M[n] - Gd[n] @ (Gd[n].T @ M[n]); U, S, Vh = torch.linalg.svd(P, full_matrices=False)
        keep = int((S > eps * float(S.max())).sum().item()) if (S.numel() and float(S.max()) > 0) else 0
        R[n] = U[:, :keep]
    return R


def align_resid(post, start_sd, carriers, V, Qresid):
    ar = {}
    with torch.no_grad():
        for n in carriers:
            dw = (post[n].detach() - start_sd[n].to(dev)).double(); coords = dw @ V[n].double()
            inV = float((coords * coords).sum())
            ar[n] = (float(((coords @ Qresid[n]) ** 2).sum()) / max(inV, 1e-12)) if Qresid[n].shape[1] > 0 else 0.0
    return ar


def grass(A, B):
    k = min(A.shape[1], B.shape[1]); return float((A.T @ B).pow(2).sum() / max(k, 1)) if k > 0 else 0.0


def snap(model, carriers): p = dict(model.named_parameters()); return {n: p[n].detach().clone() for n in carriers}
def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[BD:{CELL}] {cfg['model']} dispositive A/B/D N=30 -> model B-D + per-carrier resid", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    if cfg.get("n_layers"):
        config = AutoConfig.from_pretrained(cfg["model"]); config.num_hidden_layers = cfg["n_layers"]
        model = AutoModelForCausalLM.from_config(config).float().to(dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    if cfg.get("big"):
        model.gradient_checkpointing_enable(); model.config.use_cache = False
        print(f"[BD:{CELL}] gradient checkpointing ON (big model)", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: (v.detach().clone().cpu() if BIG else v.detach().clone()) for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[BD:{CELL}] w_A SST acc={pre_sst:.3f}; {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).float().contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd)
    print(f"[BD:{CELL}] V basis done ({time.time()-t0:.0f}s)", flush=True)

    ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    ret = {a: [] for a in ["A", "B", "D"]}; arN = {a: {n: [] for n in carriers} for a in ["A", "B", "D"]}
    EX = {"mapfrac_B": [], "mapfrac_D": [], "clB": [], "clD": [], "Mstab": []}; Mprev = None
    for sd in SEEDS:
        model.load_state_dict(ckpt_sd); b0, b1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        wB = snap(model, carriers); EX["mapfrac_B"].append(map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)); EX["clB"].append(b0 - b1)
        Mcur = map_dirs(wB, ckpt_sd, carriers, V)
        model.load_state_dict(ckpt_sd); d0, d1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        wD = snap(model, carriers); EX["mapfrac_D"].append(map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)); EX["clD"].append(d0 - d1)
        if Mprev is not None: EX["Mstab"].append(float(np.mean([grass(Mprev[n], Mcur[n]) for n in carriers])))
        Mprev = Mcur
        starts = {"A": ckpt_sd, "B": {**ckpt_sd, **wB}, "D": {**ckpt_sd, **wD}}; posts = {}
        for a in ["A", "B", "D"]:
            model.load_state_dict(starts[a]); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
            posts[a] = snap(model, carriers); ret[a].append(eval_sst(model, tok, sst_eval, posid, negid))
        G_A = surge_pcs(posts["A"], starts["A"], carriers, V, K_GEN); M_resid = residualize(Mcur, G_A)
        for a in ["A", "B", "D"]:
            ar = align_resid(posts[a], starts[a], carriers, V, M_resid)
            for n in carriers: arN[a][n].append(ar[n])
        if sd % 5 == 0 or sd == SEEDS[-1]:
            print(f"[BD:{CELL}] seed {sd}: ret A/B/D={ret['A'][-1]:.3f}/{ret['B'][-1]:.3f}/{ret['D'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    A, B, D = np.array(ret["A"]), np.array(ret["B"]), np.array(ret["D"])
    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5)]
    resid_BD_per_carrier = [float(np.mean(arN["B"][n]) - np.mean(arN["D"][n])) for n in carriers]
    A_forget = pre_sst - A.mean()
    rep = {"cell": CELL, "model": cfg["model"], "family": cfg["family"], "size_m": cfg["size_m"], "depth": cfg.get("depth"),
           "pre_sst": pre_sst, "n_carriers": len(carriers), "n_seeds": len(SEEDS), "A_forget": float(A_forget),
           "VOID": bool(A_forget < FORGET_GATE),
           "retention": {"A": me(A), "B": me(B), "D": me(D), "B_minus_A": me((B - A).tolist()),
                         "B_minus_D": me((B - D).tolist()), "D_minus_A": me((D - A).tolist())},
           "guards": {"mapfrac_B": me(EX["mapfrac_B"]), "mapfrac_D": me(EX["mapfrac_D"]),
                      "childB_drop": me(EX["clB"]), "childD_drop": me(EX["clD"]), "M_stability": me(EX["Mstab"]) if EX["Mstab"] else [0, 0]},
           "resid_BD_per_carrier_median": float(np.median(resid_BD_per_carrier)),
           "carriers": carriers}
    np.savez(NPZ, A=A, B=B, D=D, resid_BD_per_carrier=np.array(resid_BD_per_carrier),
             residB=np.array([[arN["B"][n][s] for n in carriers] for s in range(len(SEEDS))]),
             residD=np.array([[arN["D"][n][s] for n in carriers] for s in range(len(SEEDS))]),
             mapfrac_B=np.array(EX["mapfrac_B"]), mapfrac_D=np.array(EX["mapfrac_D"]), carriers=np.array(carriers))
    bd = rep["retention"]["B_minus_D"]
    print(f"\n[BD:{CELL}] === B-D FEED ===\n  ret A/B/D={A.mean():.3f}/{B.mean():.3f}/{D.mean():.3f} (pre {pre_sst:.3f}, A_forget {A_forget:.3f}{' VOID' if rep['VOID'] else ''})", flush=True)
    print(f"  B-A={rep['retention']['B_minus_A'][0]:+.3f} | B-D={bd[0]:+.3f}±{bd[1]:.3f} | D-A={rep['retention']['D_minus_A'][0]:+.3f} | "
          f"mapfrac B/D {np.mean(EX['mapfrac_B']):.3f}/{np.mean(EX['mapfrac_D']):.3f} | resid_B-D(medCarr) {rep['resid_BD_per_carrier_median']:+.3f}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[BD:{CELL}] saved bd_{CELL}.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
