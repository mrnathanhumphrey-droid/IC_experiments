"""PROBE 18 (controllability / positive-direction) — is the protect/erode effect CAUSALLY the childhood's movement along the
interference direction dVp? Steer the childhood's GRADIENT relative to dVp and see if the effect nullifies. PREREG_PROBE18_STEER.

Arms (per cell, per dose), all paired vs the shared NoCH measure:
  - baseline    : unsteered childhood (native N_phase2).
  - orth_dVp    : childhood gradient projected ORTHOGONAL to the w_A interference direction dVp (forbid movement along dVp).
  - orth_rand   : same projection against a MATCHED RANDOM in-V direction (the control — rules out "any projection weakens it").
Pre-registered read: orth_dVp collapses N_phase2 -> ~0 (effect is causally the childhood moving along dVp); orth_rand ~ baseline.
This CAUSALLY proves the directional mechanism (elimination -> positive) + demonstrates controllability (steer away -> switch off).
dVp for steering is realized from w_A (NoCH interference direction), ONCE per seed. NO amplify arm (add later if nullify holds).
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
NSEEDS = int(os.environ.get("NSEEDS", "15")); SEEDS = list(range(NSEEDS))
DOSES = [float(x) for x in os.environ.get("DOSES", "3e-5").split(",")]  # per-cell strong dose (OPT 3e-5, BLOOM 7e-6)
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; TOPK = 16; N_FISHER = 200; K_RAND = 3
HEARTBEAT = 40; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753, "BLOOM": 0.83}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
BLOOM_C = ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":   {"model": "facebook/opt-350m",    "ckpt": "opt350m_sst2_wA.pt",   "ck": OPT_C,   "family": "OPT"},
    "bloom560m": {"model": "bigscience/bloom-560m", "ckpt": "bloom560m_sst2_wA.pt", "ck": BLOOM_C, "family": "BLOOM"},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe18_steer_{CELL}.json"


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
def train_phase(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()
def steered_childhood(model, tok, stream, steps, lr, seed, params, carriers, steer=None):
    """Childhood with optional gradient projection: if steer is a dict {n: unit in-V dir}, remove the along-steer component of
    each carrier's grad every step (forbid movement along that direction). steer=None -> baseline."""
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if steer is not None:
            with torch.no_grad():
                for n in carriers:
                    gr = params[n].grad
                    if gr is None: continue
                    d = steer[n]  # unit Frobenius, in V
                    coef = float((gr.double() * d).sum())
                    gr.sub_((coef * d).to(gr.dtype))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()
def labels_of(items): return sorted(set(c for _, c in items))
def me(x): return [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)] if len(x) > 1 else [float(np.mean(x)) if x else 0.0, 0.0]


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P18:{CELL}] {cfg['model']} steer controllability; DOSES={DOSES} N={NSEEDS}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    _env_anchor = os.environ.get("ANCHOR", ""); exp = float(_env_anchor) if _env_anchor else ANCHOR_WA.get(cfg["family"], 0.0)
    print(f"[P18:{CELL}] w_A acc={pre_sst:.3f} (exp {exp}); {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)
    if abs(pre_sst - exp) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f} vs {exp}"}, open(OUT, "w"), indent=2)
        print(f"[P18:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    V16 = {n: V[n][:, :TOPK].contiguous() for n in carriers}
    try:
        ho_tr = load_jsonB("medium_subj")[:-200]; assert len(ho_tr) > 100
        stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
        stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P18:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return
    print(f"[P18:{CELL}] datasets loaded; V ready", flush=True)

    def unit_inV(delta):  # project a per-carrier displacement into V, unit-normalize (Frobenius), for steering
        out = {}
        for n in carriers:
            dv = (delta[n].double() @ V[n]) @ V[n].T; nrm = float(dv.norm())
            out[n] = (dv / (nrm + 1e-30)).contiguous()
        return out
    def realize_dVp(sd):  # NoCH interference direction: phase-2 from w_A, in-V unit per carrier
        model.load_state_dict(ckpt_sd); train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd)
        cur = dict(model.named_parameters()); delta = {n: (cur[n].detach() - ckpt_sd[n].to(dev)) for n in carriers}
        return unit_inV(delta)
    def rand_dir(sd):
        out = {}
        for n in carriers:
            gen = torch.Generator(device="cpu").manual_seed(55000 * sd + hash(n) % 9999)
            raw = torch.randn(dict(model.named_parameters())[n].shape, generator=gen).double().to(dev)
            dv = (raw @ V[n]) @ V[n].T; out[n] = (dv / (float(dv.norm()) + 1e-30)).contiguous()
        return out

    def perturb_eval(anchor_sd, fn):
        model.load_state_dict(anchor_sd)
        with torch.no_grad():
            p = dict(model.named_parameters())
            for n in carriers: p[n].add_(fn(n).to(p[n].dtype))
        return eval_sst(model, tok, sst_eval, posid, negid)
    def measure_dp2(anchor_sd, sd):  # realize phase-2 from anchor, in-V, dp2 = acc0 - acc(anchor+dVp)
        model.load_state_dict(anchor_sd); acc0 = eval_sst(model, tok, sst_eval, posid, negid)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd)
        cur = dict(model.named_parameters()); dVp = {}
        with torch.no_grad():
            for n in carriers:
                d = (cur[n].detach().double() - anchor_sd[n].to(dev).double()); dVp[n] = (d @ V[n]) @ V[n].T
        dp2 = acc0 - perturb_eval(anchor_sd, lambda n: dVp[n]); model.load_state_dict(anchor_sd)
        return float(acc0), float(dp2)

    ARMS = ["baseline", "orth_dVp", "orth_rand"]; D = len(DOSES)
    acc = {a: [{"Np": [], "retA": []} for _ in range(D)] for a in ARMS}
    for si, sd in enumerate(SEEDS):
        _, dp2_n = measure_dp2(ckpt_sd, sd)                       # NoCH, shared across arms/doses
        steer_dVp = realize_dVp(sd); steer_rand = rand_dir(sd)    # steering dirs from w_A
        for di, lr in enumerate(DOSES):
            for a in ARMS:
                steer = None if a == "baseline" else (steer_dVp if a == "orth_dVp" else steer_rand)
                model.load_state_dict(ckpt_sd)
                steered_childhood(model, tok, stream, STEPS_CHILD, lr, sd, params, carriers, steer=steer)
                ch_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                retA = eval_sst(model, tok, sst_eval, posid, negid)
                _, dp2_c = measure_dp2(ch_sd, sd)
                acc[a][di]["Np"].append(dp2_n - dp2_c); acc[a][di]["retA"].append(retA)
        rows = []
        for di, lr in enumerate(DOSES):
            r = {"lr": lr}
            for a in ARMS: r[f"Np_{a}"] = me(acc[a][di]["Np"]); r[f"retA_{a}"] = me(acc[a][di]["retA"])
            rows.append(r)
        json.dump({"cell": CELL, "family": cfg["family"], "VOID": False, "w_A_acc": pre_sst, "n_seeds_done": si+1,
                   "n_target": NSEEDS, "doses": DOSES, "arms": ARMS, "rows": rows}, open(OUT, "w"), indent=2)
        if si % 2 == 0 or si == NSEEDS-1:
            s = " | ".join(f"{r['lr']:.0e} base={r['Np_baseline'][0]:+.3f} oDVp={r['Np_orth_dVp'][0]:+.3f} oRnd={r['Np_orth_rand'][0]:+.3f}" for r in rows)
            print(f"[seed] s{si+1}/{NSEEDS} {s} ({time.time()-_T0:.0f}s)", flush=True)
    print(f"[P18:{CELL}] DONE saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
