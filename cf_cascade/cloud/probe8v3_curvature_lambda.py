"""PROBE 8 v3 — targeted vs global curvature alignment, CLEAN dual-cell. Pre-reg PREREG_PROBE8_CURVATURE_ALIGNMENT_v3_2026_06_27 (LOCK 65ae39d4).
Fixes v2's two flags: (a) calibration MARGIN 0.015 + K_CAL=20 -> reliably non-destructive dose; (b) MULT={1,2,3} magnitude
sweep -> lift random baseline off the floor + test magnitude-robustness. Bounded accuracy-drops (no ratios).
 STAGE 1: lr_child grid x20; chosen_lr = max{lr: r_cal[lr] >= (w_A-0.05)+0.015}; fallback >= w_A-0.05 (tight_dose); else min.
 STAGE 2 per seed, NoCH (anchor=w_A) vs CH (anchor=w_postCH @ chosen_lr); V=top-128 task-A Fisher @ w_A:
   phase-2(250) from anchor -> Delta_V (in-V realized displacement). For m in {1,2,3}:
     dphase2(m)=acc(anchor)-acc(anchor+m*Delta_V);  drand(m)=mean_{K_RAND=5} acc(anchor)-acc(anchor+m*R) [R rand in V, ||R_n||=m*||Delta_V,n||]
     aniso(m)=drand(m)-dphase2(m)
 Targeting daniso(m)=mean(aniso_CH-aniso_NoCH) >0 (TARGETED); global g(m)=mean(drand_NoCH-drand_CH) >0 (literature).
 Sensitivity gate: mean drand(NoCH,m=3) > 0.01. HARD-STOP (VOID) on dataset load fail or w_A anchor off-band.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100))
LR_GRID = [1e-5, 1.5e-5, 2e-5, 3e-5]; K_CAL = 20; NONDESTRUCT_TOL = 0.05; CAL_MARGIN = 0.015
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; N_FISHER = 200; K_RAND = 5; MULT = [1, 2, 3]
EPS = 1e-6; HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe8v3_curvature_{CELL}.json"


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


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P8v3:{CELL}] {cfg['model']} targeted-vs-global curvature alignment, grid={LR_GRID} N={len(SEEDS)} MULT={MULT} K_RAND={K_RAND}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P8v3:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    exp = ANCHOR_WA[cfg["family"]]
    if abs(pre_sst - exp) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off: {pre_sst:.3f} vs {exp:.3f}+-{ANCHOR_TOL}"}, open(OUT, "w"), indent=2)
        print(f"[P8v3:{CELL}] HARD-STOP: w_A anchor off-band -> VOID.", flush=True); return
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
        print(f"[P8v3:{CELL}] HARD-STOP: datasets failed to load -> VOID. {e}", flush=True); return
    print(f"[P8v3:{CELL}] datasets loaded (phase-2 subj {len(ho_tr)}; childhood {len(stream)} streams)", flush=True)

    # ---------- STAGE 1: calibration (margin + K_CAL=20) ----------
    thr = pre_sst - NONDESTRUCT_TOL; cal = {}
    print(f"[P8v3:{CELL}] STAGE-1 calibration; thr={thr:.3f} margin={CAL_MARGIN} (target r_cal>={thr+CAL_MARGIN:.3f})", flush=True)
    for lr in LR_GRID:
        accs = []
        for sd in range(K_CAL):
            model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, lr, sd, hb="")
            accs.append(eval_sst(model, tok, sst_eval, posid, negid))
        cal[lr] = float(np.mean(accs))
        print(f"[cal] lr={lr:.1e} r_cal={cal[lr]:.4f} (K={K_CAL}) ({time.time()-_T0:.0f}s)", flush=True)
    q_margin = [lr for lr in LR_GRID if cal[lr] >= thr + CAL_MARGIN]; q_bare = [lr for lr in LR_GRID if cal[lr] >= thr]
    tight_dose = False; no_nd = False
    if q_margin: chosen_lr = max(q_margin)
    elif q_bare: chosen_lr = max(q_bare); tight_dose = True
    else: chosen_lr = min(LR_GRID); no_nd = True
    print(f"[P8v3:{CELL}] STAGE-1 DONE -> chosen_lr={chosen_lr:.1e} tight_dose={tight_dose} no_nondestructive_point={no_nd}", flush=True)
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()

    # ---------- STAGE 2: targeted-vs-global perturb-and-eval, MULT sweep ----------
    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def measure(anchor_sd, sd, hb):
        """Returns (acc0, {m:(dphase2,drand)}). Restores anchor. Random direction fixed per (sd,k), scaled by m."""
        model.load_state_dict(anchor_sd); acc0 = eval_sst(model, tok, sst_eval, posid, negid)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=hb)
        cur = dict(model.named_parameters()); dV = {}
        with torch.no_grad():
            for n in carriers:
                d = (cur[n].detach().double() - anchor_sd[n].to(dev).double()); dV[n] = (d @ V[n]) @ V[n].T
        out = {}
        for m in MULT:
            model.load_state_dict(anchor_sd)
            with torch.no_grad():
                p = dict(model.named_parameters())
                for n in carriers: p[n].add_((m * dV[n]).to(p[n].dtype))
            dphase2 = acc0 - eval_sst(model, tok, sst_eval, posid, negid)
            dr = []
            for k in range(K_RAND):
                model.load_state_dict(anchor_sd); gen = torch.Generator(device="cpu").manual_seed(100000 * sd + 777 * k + 13)
                with torch.no_grad():
                    p = dict(model.named_parameters())
                    for n in carriers:
                        raw = torch.randn(p[n].shape[0], p[n].shape[1], generator=gen).double().to(dev)
                        rv = (raw @ V[n]) @ V[n].T; rv = rv / (float(rv.norm()) + 1e-30) * (m * float(dV[n].norm()))
                        p[n].add_(rv.to(p[n].dtype))
                dr.append(acc0 - eval_sst(model, tok, sst_eval, posid, negid))
            out[m] = (float(dphase2), float(np.mean(dr)))
        model.load_state_dict(anchor_sd)
        return float(acc0), out

    dp_n = {m: [] for m in MULT}; dr_n = {m: [] for m in MULT}; dp_c = {m: [] for m in MULT}; dr_c = {m: [] for m in MULT}
    ani_n = {m: [] for m in MULT}; ani_c = {m: [] for m in MULT}; retA_ch = []
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"s{si+1}/{ns}"
        _, on = measure(ckpt_sd, sd, hb=f"{lab} NoCH_p2")
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, chosen_lr, sd, hb=f"{lab} CH_child")
        ch_sd = snap(); retA_ch.append(eval_sst(model, tok, sst_eval, posid, negid))
        _, oc = measure(ch_sd, sd, hb=f"{lab} CH_p2")
        for m in MULT:
            dp_n[m].append(on[m][0]); dr_n[m].append(on[m][1]); ani_n[m].append(on[m][1] - on[m][0])
            dp_c[m].append(oc[m][0]); dr_c[m].append(oc[m][1]); ani_c[m].append(oc[m][1] - oc[m][0])
        d1 = float(np.mean(np.array(ani_c[1]) - np.array(ani_n[1]))); g1 = float(np.mean(np.array(dr_n[1]) - np.array(dr_c[1])))
        print(f"[seed] {lab} DONE  m1: aniN={ani_n[1][-1]:+.3f} aniC={ani_c[1][-1]:+.3f} dp_n={dp_n[1][-1]:.3f} dp_c={dp_c[1][-1]:.3f} drN={dr_n[1][-1]:.3f}  drN(m3)={dr_n[3][-1]:.3f} retA_ch={retA_ch[-1]:.3f}  run(daniso1)={d1:+.4f} run(g1)={g1:+.4f}  ({time.time()-_T0:.0f}s)", flush=True)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    nondestruct = bool(np.mean(retA_ch) >= pre_sst - NONDESTRUCT_TOL)
    sens_ok = bool(np.mean(dr_n[3]) > 0.01); insensitive = not sens_ok
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst, "VOID": False,
           "calibration_curve": {f"{k:.1e}": v for k, v in cal.items()}, "chosen_lr": chosen_lr, "tight_dose": tight_dose, "no_nondestructive_point": no_nd,
           "ret_anchor_ch": me(retA_ch), "childhood_nondestructive": nondestruct, "sensitivity_drandN_m3": me(dr_n[3]), "insensitive": insensitive, "MULT": MULT}
    for m in MULT:
        rep[f"dphase2_noch_m{m}"] = me(dp_n[m]); rep[f"dphase2_ch_m{m}"] = me(dp_c[m])
        rep[f"drand_noch_m{m}"] = me(dr_n[m]); rep[f"drand_ch_m{m}"] = me(dr_c[m])
        rep[f"aniso_noch_m{m}"] = me(ani_n[m]); rep[f"aniso_ch_m{m}"] = me(ani_c[m])
        rep[f"delta_aniso_m{m}"] = me((np.array(ani_c[m]) - np.array(ani_n[m])).tolist())
        rep[f"global_g_m{m}"] = me((np.array(dr_n[m]) - np.array(dr_c[m])).tolist())
    rep["TARGETED_if"] = "delta_aniso_m1>0 (>=3sigma) AND childhood_nondestructive AND sensitivity_ok AND sign holds m1,m2,m3"
    rep["GLOBAL_if"] = "global_g_m*>0 (>=2sigma) AND delta_aniso~=0"
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[P8v3:{CELL}] chosen_lr={chosen_lr:.1e} nondestruct={nondestruct} (retA_ch={rep['ret_anchor_ch'][0]:.3f}) sensitivity drandN_m3={rep['sensitivity_drandN_m3'][0]:.3f} (ok={sens_ok})", flush=True)
    for m in MULT:
        dm, de = rep[f"delta_aniso_m{m}"]; gm, ge = rep[f"global_g_m{m}"]
        print(f"[P8v3:{CELL}] m={m}: delta_aniso(TARGETING)={dm:+.4f}+-{de:.4f} ({dm/max(de,1e-9):+.1f}s)  global_g={gm:+.4f}+-{ge:.4f} ({gm/max(ge,1e-9):+.1f}s)  dphase2_noch={rep[f'dphase2_noch_m{m}'][0]:+.3f} dphase2_ch={rep[f'dphase2_ch_m{m}'][0]:+.3f} drand_noch={rep[f'drand_noch_m{m}'][0]:+.3f}", flush=True)
    print(f"[P8v3:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
