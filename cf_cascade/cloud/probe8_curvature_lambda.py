"""PROBE 8 v2 — targeted vs global curvature alignment (+ cross-arch). Pre-reg PREREG_PROBE8_CURVATURE_ALIGNMENT_v2_2026_06_26 (LOCK 93f26963).
(v1 LOCK b33ab105 VOID: R-ratio on CE loss was numerically degenerate. v2 = BOUNDED accuracy-drops, differences not ratios.)
Is the childhood's protection GLOBAL flatness (literature) or TARGETED to the phase-2 interference direction (novel)?
 STAGE 1 (locked, = Probe-7b): lr_child grid [1e-5,1.5e-5,2e-5,3e-5]x12; chosen_lr=max lr with post-childhood task-A >= w_A-0.05.
 STAGE 2 perturb-and-eval, paired, per seed, conditions NoCH (anchor=w_A) and CH (anchor=w_postCH @ chosen_lr):
   - run phase-2(250) from anchor -> realized displacement Delta; project per-carrier into V (top-128 task-A Fisher @ w_A) -> Delta_V.
   - dphase2 = acc_A(anchor) - acc_A(anchor + Delta_V)            [drop along realized phase-2 vector, in-V only]
   - drand   = mean_{K_RAND=5} acc_A(anchor) - acc_A(anchor + R)  [R random in V, per-carrier ||R_n||=||Delta_V,n||]
   - aniso   = drand - dphase2   (>0 => phase-2 dir LESS damaging than random; all quantities bounded in [-1,1])
 Targeting: daniso = mean(aniso_CH - aniso_NoCH) >0 => childhood makes phase-2 dir relatively safer (TARGETED, novel).
 Global:    g      = mean(drand_NoCH - drand_CH) >0 => generic basin widening (literature).
HARD-STOP (VOID) on dataset load fail or w_A anchor off-band.
"""
import os, time, json, math, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100))
LR_GRID = [1e-5, 1.5e-5, 2e-5, 3e-5]; K_CAL = 12; NONDESTRUCT_TOL = 0.05
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; N_FISHER = 200; K_RAND = 5
EPS = 1e-6; HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe8_curvature_{CELL}.json"


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
def eval_loss(model, tok, evalset):
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, len(evalset), 32):
            chunk = evalset[i:i + 32]; inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l]) for s, l in chunk])
            l = model(input_ids=inp, attention_mask=am, labels=lb).loss; tot += float(l) * len(chunk); n += len(chunk)
    return tot / max(n, 1)
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


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P8:{CELL}] {cfg['model']} targeted-vs-global curvature alignment, grid={LR_GRID} N={len(SEEDS)} K_RAND={K_RAND}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P8:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    exp = ANCHOR_WA[cfg["family"]]
    if abs(pre_sst - exp) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off: {pre_sst:.3f} vs {exp:.3f}+-{ANCHOR_TOL}"}, open(OUT, "w"), indent=2)
        print(f"[P8:{CELL}] HARD-STOP: w_A anchor off-band -> VOID.", flush=True); return
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
        print(f"[P8:{CELL}] HARD-STOP: datasets failed to load -> VOID. {e}", flush=True); return
    print(f"[P8:{CELL}] datasets loaded (phase-2 subj {len(ho_tr)}; childhood {len(stream)} streams)", flush=True)

    # ---------- STAGE 1: calibration (locked, = Probe-7b) ----------
    nd_thresh = pre_sst - NONDESTRUCT_TOL; cal = {}
    print(f"[P8:{CELL}] STAGE-1 calibration; non-destructive threshold ret >= {nd_thresh:.3f}", flush=True)
    for lr in LR_GRID:
        accs = []
        for sd in range(K_CAL):
            model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, lr, sd, hb="")
            accs.append(eval_sst(model, tok, sst_eval, posid, negid))
        cal[lr] = float(np.mean(accs))
        print(f"[cal] lr={lr:.1e} r_cal={cal[lr]:.4f} ({'OK' if cal[lr]>=nd_thresh else 'destructive'}) ({time.time()-_T0:.0f}s)", flush=True)
    qualifying = [lr for lr in LR_GRID if cal[lr] >= nd_thresh]; no_nd = len(qualifying) == 0
    chosen_lr = max(qualifying) if qualifying else min(LR_GRID)
    print(f"[P8:{CELL}] STAGE-1 DONE -> chosen_lr={chosen_lr:.1e} no_nondestructive_point={no_nd}", flush=True)
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()

    # ---------- STAGE 2: targeted-vs-global perturb-and-eval ----------
    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def measure(anchor_sd, sd, hb):
        """Returns (acc0, dphase2, drand) = accuracy DROPS (bounded) at anchor. Restores anchor at the end."""
        model.load_state_dict(anchor_sd); acc0 = eval_sst(model, tok, sst_eval, posid, negid)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=hb)  # realized phase-2 from anchor
        cur = dict(model.named_parameters()); dV = {}
        with torch.no_grad():
            for n in carriers:
                d = (cur[n].detach().double() - anchor_sd[n].to(dev).double()); dV[n] = (d @ V[n]) @ V[n].T
        # drop along realized phase-2 direction (in-V)
        model.load_state_dict(anchor_sd)
        with torch.no_grad():
            p = dict(model.named_parameters())
            for n in carriers: p[n].add_(dV[n].to(p[n].dtype))
        dphase2 = acc0 - eval_sst(model, tok, sst_eval, posid, negid)
        # drop along K_RAND random in-V directions, magnitude-matched per carrier
        dr = []
        for k in range(K_RAND):
            model.load_state_dict(anchor_sd); gen = torch.Generator(device="cpu").manual_seed(100000 * sd + 777 * k + 13)
            with torch.no_grad():
                p = dict(model.named_parameters())
                for n in carriers:
                    raw = torch.randn(p[n].shape[0], p[n].shape[1], generator=gen).double().to(dev)
                    rv = (raw @ V[n]) @ V[n].T; nrm = float(rv.norm())
                    rv = rv / (nrm + 1e-30) * float(dV[n].norm()); p[n].add_(rv.to(p[n].dtype))
            dr.append(acc0 - eval_sst(model, tok, sst_eval, posid, negid))
        model.load_state_dict(anchor_sd)
        return float(acc0), float(dphase2), float(np.mean(dr))

    dp2_n = []; dp2_c = []; dr_n = []; dr_c = []; ani_n = []; ani_c = []; retA_ch = []
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"s{si+1}/{ns}"
        # NoCH: anchor = w_A
        _, dp2, dr = measure(ckpt_sd, sd, hb=f"{lab} NoCH_p2"); dp2_n.append(dp2); dr_n.append(dr); ani_n.append(dr - dp2)
        # CH: anchor = w_postCH @ chosen_lr
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, chosen_lr, sd, hb=f"{lab} CH_child")
        ch_sd = snap(); retA_ch.append(eval_sst(model, tok, sst_eval, posid, negid))
        _, dp2, dr = measure(ch_sd, sd, hb=f"{lab} CH_p2"); dp2_c.append(dp2); dr_c.append(dr); ani_c.append(dr - dp2)
        daniso = float(np.mean(np.array(ani_c) - np.array(ani_n))); g = float(np.mean(np.array(dr_n) - np.array(dr_c)))
        print(f"[seed] {lab} DONE  aniso_n={ani_n[-1]:+.3f} aniso_c={ani_c[-1]:+.3f} dr_n={dr_n[-1]:.3f} dr_c={dr_c[-1]:.3f} dp2_n={dp2_n[-1]:.3f} dp2_c={dp2_c[-1]:.3f} retA_ch={retA_ch[-1]:.3f}  run(daniso)={daniso:+.4f} run(g)={g:+.4f}  ({time.time()-_T0:.0f}s)", flush=True)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    daniso = me((np.array(ani_c) - np.array(ani_n)).tolist()); g = me((np.array(dr_n) - np.array(dr_c)).tolist())
    nondestruct = bool(np.mean(retA_ch) >= pre_sst - NONDESTRUCT_TOL); insensitive = bool(np.mean(dr_n) < 0.01)
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst, "VOID": False,
           "calibration_curve": {f"{k:.1e}": v for k, v in cal.items()}, "chosen_lr": chosen_lr, "no_nondestructive_point": no_nd,
           "dphase2_noch": me(dp2_n), "dphase2_ch": me(dp2_c), "drand_noch": me(dr_n), "drand_ch": me(dr_c),
           "aniso_noch": me(ani_n), "aniso_ch": me(ani_c),
           "delta_aniso_targeting": daniso, "global_flatten_g": g, "ret_anchor_ch": me(retA_ch),
           "childhood_nondestructive": nondestruct, "insensitive": insensitive,
           "TARGETED_if": "delta_aniso_targeting>0 (>=2sigma)", "GLOBAL_if": "global_flatten_g>0 (>=2sigma) AND delta_aniso~=0"}
    json.dump(rep, open(OUT, "w"), indent=2)
    dm, de = daniso; gm, ge = g
    print(f"[P8:{CELL}] chosen_lr={chosen_lr:.1e} nondestruct={nondestruct} insensitive={insensitive}  aniso_noch={rep['aniso_noch'][0]:+.3f} aniso_ch={rep['aniso_ch'][0]:+.3f}", flush=True)
    print(f"[P8:{CELL}] delta_aniso (TARGETING) = {dm:+.4f} +- {de:.4f} ({dm/max(de,1e-9):+.1f}sigma)", flush=True)
    print(f"[P8:{CELL}] global_flatten g        = {gm:+.4f} +- {ge:.4f} ({gm/max(ge,1e-9):+.1f}sigma)", flush=True)
    print(f"[P8:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
