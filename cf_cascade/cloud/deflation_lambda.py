"""DEFLATION TEST (Wilson-scoped) — is the fixed contested core just task-A's dominant-curvature subspace?
Null to kill: if the contested core = task-A's top-Fisher(curvature) directions, "every task contests the same core"
is near-tautological (tasks fight the loudest directions) -> deflates to a relabel of a known object
(Gur-Ari tiny gradient subspace / Wu2020 fixed Hessian top-subspace / Papyan class-gradient origin).

PRIMITIVE: PA = task-A top-64 Fisher eigvecs (ascending, so col 63 = highest curvature). For each task-B, PB = its
top-64. contestation_j = mean_B ||PB^T PA[:,j]||^2 = how much (on avg) the task-Bs want PA-axis j. Contested core =
top-16 contestation axes. (cross-B consistency of contestation reproduces the z=+197 universality in axis-aligned view.)

WILSON's 2 REQUIRED MEASUREMENTS (both needed for DISTINCT):
 M1 Grassmann/subspace overlap: contested-core-16 vs task-A top-16-highest-Fisher-eigenvalue (= PA cols 48..63).
    Both are axis-aligned subsets of the orthonormal PA frame -> overlap = |index intersection|/16. chance=16/64=0.25.
    >=0.7 -> DEFLATION (core IS the curvature subspace).
 M2 dissociation: contestation_j vs Fisher-eigenvalue-rank j. Spearman corr + COUNT of high-contested(top-16) dirs
    that are low-curvature (eigrank in bottom-32). High-contested-low-curvature dirs = CF-specific object the
    landscape literature does NOT predict.
PRE-COMMITTED VERDICT: M1<0.4 AND dissociation present (>0 high-contested-low-curv) -> DISTINCT (novel structural
 object). M1>=0.7 -> DEFLATION (pivot headline to diffusion). M1 low BUT no dissociation -> BORDERLINE (don't bank).
RIDER (Laitinen 2601.18699 / LRCP-GPM fold-in): M3 = is the core reducible to the dominant-ACTIVATION subspace
 (input-cov E[x^T x], the basis LRCP/GPM protect)? overlap of core vs top-16 activation dirs + corr. Strong-distinct
 needs core distinct from curvature AND activation. Special attention: mlp.dense_4h_to_h = Wdown = Laitinen peak-drift.
SAVES artifacts (FA eigden, contestation, contested core, activation dirs) so future tests don't retrain.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/deflation.json"; ART = f"{G}/deflation_artifacts.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
TASKS = [("agnews", "low_agnews"), ("yelp", "high_yelp"), ("subj", "medium_subj")]
KTOP = 64; CORE = 16; N_KFAC = 16
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_FISHER = 150; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst_trA():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:max(N_FISHER, 200)]]


def load_taskB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def b_sst(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def b_pair(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def pack(tok, seqs, labs):
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher(model, params, names, tok, items, is_pair):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for it in items[:N_FISHER]:
        inp, lb, am = (b_pair(tok, [it]) if is_pair else b_sst(tok, [it])); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def activation_cov(model, mods, tok, trA):
    cur = {}; H = []
    A = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1], device=dev, dtype=torch.float64) for n, m in mods.items()}; cnt = {n: 0 for n in mods}
    for n, m in mods.items():
        def fwd(mod, i, o, nn=n): cur[nn] = i[0].detach()
        H.append(m.register_forward_hook(fwd))
    gk = torch.Generator().manual_seed(7); model.eval()
    for _ in range(N_KFAC):
        inp, lb, am = b_sst(tok, [trA[k] for k in torch.randperm(len(trA), generator=gk)[:BATCH].tolist()]); model(input_ids=inp, attention_mask=am)
        for n in mods: x = cur[n].reshape(-1, cur[n].shape[-1]).double(); A[n] += x.T @ x; cnt[n] += x.shape[0]
    for h in H: h.remove()
    return {n: A[n] / max(cnt[n], 1) for n in A}


def train_taskB(model, tok, B):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0); run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        for i in range(0, len(B) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); return float(np.mean(run)) if run else float("nan"), step


def main():
    t0 = time.time(); print("[DEF] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    mods = {n: m for n, m in model.named_modules() if any(c in n for c in CARRIERS) and hasattr(m, "weight") and getattr(m.weight, "ndim", 0) == 2}

    # FA (curvature) + PA top-64 + eigenvalues; activation cov (LRCP/GPM basis) top dirs — both at w_A
    FA = fisher(model, params, names, tok, trA, False)
    PA = {}; LAM = {}
    for n in names:
        ev, V = torch.linalg.eigh(FA[n]); PA[n] = V[:, -KTOP:].contiguous(); LAM[n] = ev[-KTOP:]   # ascending -> col 63 highest curv
    Aacc = activation_cov(model, mods, tok, trA)
    ACT16 = {n: torch.linalg.eigh(Aacc[n[:-len(".weight")]])[1][:, -CORE:].contiguous() for n in names}
    print(f"[DEF] FA + activation-cov done ({time.time()-t0:.0f}s)", flush=True)

    # contestation per PA-axis, per task-B
    contest_B = {n: [] for n in names}
    for taskkey, jsonname in TASKS:
        model.load_state_dict(torch.load(CKPT)); B = load_taskB(jsonname)
        loss, steps = train_taskB(model, tok, B)
        FB = fisher(model, params, names, tok, B, True)
        for n in names:
            PB = torch.linalg.eigh(FB[n])[1][:, -KTOP:]
            contest_B[n].append((PB.T @ PA[n]).pow(2).sum(0).cpu().numpy())   # (64,) per task-B
        print(f"[DEF] {taskkey} loss={loss:.3f} steps={steps} ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-carrier deflation measurements ----
    out = {}; arts = {}
    print(f"\n[DEF] === DEFLATION (M1 chance={CORE/KTOP:.3f}; >=0.7 deflate, <0.4+dissoc distinct) ===", flush=True)
    for c in CARRIERS:
        cn = [x for x in names if c in x]
        m1s, sps, dissocs, m3ovl, m3corr, consist = [], [], [], [], [], []
        for n in cn:
            cB = np.stack(contest_B[n])                      # (3, 64)
            contest = cB.mean(0)                             # (64,)
            consist.append(float(np.mean([np.corrcoef(cB[i], cB[j])[0, 1] for i in range(3) for j in range(i + 1, 3)])))
            core_idx = set(np.argsort(contest)[-CORE:].tolist())     # top-16 contested PA-axes
            curv_idx = set(range(KTOP - CORE, KTOP))                 # top-16 highest-eigenvalue PA-axes (48..63)
            m1s.append(len(core_idx & curv_idx) / CORE)             # M1 Grassmann (axis-aligned -> index overlap)
            sps.append(_spearman(contest, np.arange(KTOP)))   # M2 contest vs eig-rank
            dissocs.append(int(sum(1 for j in core_idx if j < KTOP // 2)))       # high-contested & low-curv (bottom half)
            # M3: activation-subspace reducibility
            actload = (ACT16[n].T.double() @ PA[n].double()).pow(2).sum(0).cpu().numpy()  # (64,) PA-axis loading in act-subspace
            m3ovl.append(float(actload[list(core_idx)].mean()))     # mean act-loading of contested core
            m3corr.append(float(np.corrcoef(contest, actload)[0, 1]))
            arts[f"contest__{n}"] = contest; arts[f"lam__{n}"] = LAM[n].cpu().numpy()
        out[c] = {"M1_grassmann_vs_curv": float(np.mean(m1s)), "M2_spearman_contest_vs_eigrank": float(np.mean(sps)),
                  "M2_dissociation_count": float(np.mean(dissocs)), "M3_core_act_loading": float(np.mean(m3ovl)),
                  "M3_corr_contest_act": float(np.mean(m3corr)), "crossB_consistency": float(np.mean(consist))}
        o = out[c]
        print(f"  [{c:26s}] M1(vs curv)={o['M1_grassmann_vs_curv']:.2f}  M2 spearman={o['M2_spearman_contest_vs_eigrank']:+.2f} dissoc={o['M2_dissociation_count']:.1f}/16  M3 act-load={o['M3_core_act_loading']:.2f} corr={o['M3_corr_contest_act']:+.2f}  consist={o['crossB_consistency']:.2f}", flush=True)

    gM1 = float(np.mean([out[c]["M1_grassmann_vs_curv"] for c in CARRIERS]))
    gDis = float(np.mean([out[c]["M2_dissociation_count"] for c in CARRIERS]))
    gAct = float(np.mean([out[c]["M3_core_act_loading"] for c in CARRIERS]))
    if gM1 >= 0.7:
        verdict = f"DEFLATION (M1={gM1:.2f}>=0.7: contested core IS task-A top-curvature subspace -> known-object relabel; pivot headline to diffusion)"
    elif gM1 < 0.4 and gDis > 0:
        verdict = f"DISTINCT (M1={gM1:.2f}<0.4 AND dissociation present {gDis:.1f}/16 high-contested-low-curv dirs -> CF-specific structural object, NOT the curvature subspace). act-reducible? mean core act-loading {gAct:.2f}"
    else:
        verdict = f"BORDERLINE (M1={gM1:.2f}, dissoc={gDis:.1f}) -> not-yet-distinct, needs longer look; do NOT bank distinct"
    print(f"\n[DEF] grand M1={gM1:.2f} (chance {CORE/KTOP:.2f})  dissoc={gDis:.1f}/16  core-act-loading={gAct:.2f}\n[DEF] VERDICT: {verdict}", flush=True)
    np.savez(ART, **arts)
    json.dump({"per_carrier": out, "grand_M1_vs_curv": gM1, "grand_dissociation": gDis, "grand_core_act_loading": gAct,
               "M1_chance": CORE / KTOP, "ktop": KTOP, "core": CORE, "verdict": verdict, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[DEF] saved deflation.json + deflation_artifacts.npz wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
