"""RESIDUAL FRAGILE TEST — does the BEYOND-CURVATURE seed component protect beyond the curvature lever?
The 7-arm couldn't test this: full fragile = ~stiffest-16, so no stiff-matched control existed (draw_lam 1.28e-2 vs
fragile 9.44e-2). Here we test the RESIDUAL (curvature band projected out) -> by construction NOT the stiffest, so a
stiff-matched control (randorthog, also off-curvature) IS constructible. 3-vs-5 finally isolates "the residual specifically".

SUBSPACES (per carrier, at SST-2 w_A): P_band = top-64 FA eigvecs; CURV = top-16 (= the known lever's target);
S = seed-covariance (sum g^T g over a probe panel DISJOINT from test task-Bs); S_perp = (I-P_band)S(I-P_band);
RESIDUAL = top-16 eigvecs of S_perp (curvature-orthogonal by construction); randorthog = random 16-dim in the
band-orthogonal complement (stiffness-matched to RESIDUAL). GUARD: Grassmann(RESIDUAL,CURV)~chance (else projection failed=void).
ARMS (FULL-TIME only -- 7-arm killed timed-early on real-forgetting AG; timing is dead):
  1 naive               nothing                  -> forgetting floor
  2 curv_full           CURV (top-16 curvature)  -> the KNOWN LEVER (guard: must retain ~0.688 on AG)
  3 curvres_full        CURV + RESIDUAL (32)     -> does adding residual beat curv alone? THE HYPOTHESIS
  4 res_full            RESIDUAL only (16)        -> does residual protect AT ALL?
  5 currand_full        CURV + randorthog (32)   -> CONTROL: residual vs ANY orthogonal subspace (curvature-matched)
DECISIVE: 3v2 (residual adds beyond curv?) ; 3v5 (residual SPECIFIC, not just more dims? -- the now-constructible control) ; 4v1.
ROBUST: AG + a SECOND high-forgetting task-B (pre-checked >=0.10; NOT code=dead substrate); 7 seeds; full-time;
two metrics (SST-2 retention + task-B final loss). Don't bank 3>2 unless the gap clears 2*SE. Save before readout.
PRE-COMMITTED: 3>2 AND 3>5 (both clear 2SE) -> residual protects specifically beyond curvature (novel result survives in correct form);
3~2 -> residual adds nothing, curvature lever captures all (clean negative, caps lever); 3>2 BUT 3~5 -> just more dims, residual not special.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/residual_test.json"; NPZ = f"{G}/residual_test.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
JSONK = {"ag": "low_agnews", "yelp": "high_yelp", "subj": "medium_subj"}
KBAND = 64; FRAG = 16; N_FISHER = 150; N_SEED = 8; STEPS = 300; LR = 5e-5; BATCH = 16
LAB = {0: " negative", 1: " positive"}; dev = "cuda"; SEEDS = list(range(7))
torch.manual_seed(0); np.random.seed(0)
ARMS = [("naive", "none"), ("curv_full", "curv"), ("curvres_full", "curvres"), ("res_full", "res"), ("currand_full", "currand")]


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def load_code():
    from datasets import load_dataset
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    rows = [((r.get("instruction") or "").strip() + "\n", " " + (r.get("output") or "").strip()) for r in ds if (r.get("instruction") and r.get("output"))]
    return rows[:1500]


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def taskdata(key):
    if key == "code": return load_code()
    if key == "sst2": return [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", 400, 1)]
    d = load_jsonB(JSONK[key]); assert len(d) > 50, f"{key} {len(d)}"; return d


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


def fisher(model, params, names, tok, items):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for s, l in items[:N_FISHER]:
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


def train_arm(model, ckpt_sd, tok, B, Vmap, seed, names_prot, posid, negid, sst_eval):
    model.load_state_dict(ckpt_sd); model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    params = dict(model.named_parameters()); g = torch.Generator().manual_seed(seed); lastloss = float("nan")
    for step in range(STEPS):
        inp, lb, am = b_pairs(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if Vmap is not None:
            for n in names_prot:
                Vn = Vmap[n]; gd = params[n].grad; gd.sub_((gd @ Vn) @ Vn.T)
        opt.step(); lastloss = float(loss.item())
    return eval_sst(model, tok, sst_eval, posid, negid), lastloss


def grass(A, B):
    M = A.T @ B; s = np.linalg.svd(M, compute_uv=False); return float((s ** 2).sum() / A.shape[1])


def main():
    t0 = time.time(); print("[RES] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); pre_acc = eval_sst(model, tok, sst_eval, posid, negid)

    # curvature basis (panel-independent)
    FA = fisher(model, params, names, tok, sst_rows("train", 200, 0))
    Pband = {}; CURV = {}; LAM = {}
    for n in names:
        ev, V = torch.linalg.eigh(FA[n]); Pband[n] = V[:, -KBAND:].float().contiguous(); CURV[n] = V[:, -FRAG:].float().contiguous(); LAM[n] = ev
    FAf = {n: FA[n].float() for n in names}
    print(f"[RES] curvature basis done, pre SST-2 acc={pre_acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # PRE-CHECK: naive forgetting for ag/yelp/subj (1 seed) -> pick AG + best second (>=0.10), NOT code
    forget = {}
    for tk in ["ag", "yelp", "subj"]:
        r, _ = train_arm(model, ckpt_sd, tok, taskdata(tk), None, 0, names, posid, negid, sst_eval)
        forget[tk] = pre_acc - r; print(f"[RES] precheck {tk}: naive ret={r:.3f} forget={forget[tk]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    second = max(["yelp", "subj"], key=lambda k: forget[k]); TESTBS = ["ag", second]
    print(f"[RES] second task-B = {second} (forget {forget[second]:.3f}{' OK' if forget[second]>=0.10 else ' <0.10 WEAK-FLAG'}); test on {TESTBS}", flush=True)

    # RESIDUAL from panel DISJOINT from test task-Bs (universal, no circularity)
    panel = [p for p in ["yelp", "subj", "code"] if p not in TESTBS]
    S = {n: torch.zeros_like(FA[n]) for n in names}
    for pk in panel:
        Bp = taskdata(pk); model.load_state_dict(ckpt_sd); params = dict(model.named_parameters())
        gg = torch.Generator().manual_seed(0)
        for _ in range(N_SEED):
            model.zero_grad(set_to_none=True)
            inp, lb, am = b_pairs(tok, [Bp[k] for k in torch.randperm(len(Bp), generator=gg)[:BATCH].tolist()])
            model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
            for n in names: g = params[n].grad.detach().double(); S[n] += g.T @ g
    model.load_state_dict(ckpt_sd)
    RES = {}
    for n in names:
        Pb = Pband[n].double(); Sn = S[n]
        Sb = Sn - Pb @ (Pb.T @ Sn); Sp = Sb - (Sb @ Pb) @ Pb.T          # band projected out both sides
        ev, V = torch.linalg.eigh(Sp); RES[n] = V[:, -FRAG:].float().contiguous()
    # GUARDS: residual orthogonality + stiffness-match (Rayleigh lam)
    def ray(Vn, n): return float((Vn.double().T @ FAf[n].double() @ Vn.double()).diagonal().mean())
    g_rc = float(np.mean([grass(RES[n].cpu().numpy(), CURV[n].cpu().numpy()) for n in names]))
    lam_curv = float(np.mean([ray(CURV[n], n) for n in names])); lam_res = float(np.mean([ray(RES[n], n) for n in names]))
    print(f"[RES] panel={panel}; GUARD Grassmann(RES,CURV)={g_rc:.3f} (chance~{FRAG/1024:.3f}); lam CURV={lam_curv:.2e} RES={lam_res:.2e} ({time.time()-t0:.0f}s)", flush=True)

    def make_V(target, seed):
        if target == "none": return None, None
        if target == "curv": return {n: CURV[n] for n in names}, None
        if target == "res": return {n: RES[n] for n in names}, None
        if target == "curvres": return {n: torch.cat([CURV[n], RES[n]], 1).contiguous() for n in names}, None
        if target == "currand":
            rng = np.random.default_rng(3000 + seed); Vm = {}; lams = []
            for n in names:
                inn = Pband[n].shape[0]; M = torch.tensor(rng.standard_normal((inn, FRAG))).to(dev).float()
                Pb = Pband[n]; M = M - Pb @ (Pb.T @ M); Q, _ = torch.linalg.qr(M); ro = Q[:, :FRAG].contiguous()
                Vm[n] = torch.cat([CURV[n], ro], 1).contiguous(); lams.append(ray(ro, n))
            return Vm, float(np.mean(lams))
        return None, None

    results = {}; rand_lam = None
    for tb in TESTBS:
        B = taskdata(tb); results[tb] = {}
        for (aname, target) in ARMS:
            rets, bls = [], []
            for sd in SEEDS:
                Vmap, rl = make_V(target, sd)
                if rl is not None: rand_lam = rl
                ret, bl = train_arm(model, ckpt_sd, tok, B, Vmap, sd, names, posid, negid, sst_eval)
                rets.append(ret); bls.append(bl)
            m, s = float(np.mean(rets)), float(np.std(rets))
            results[tb][aname] = {"ret_mean": m, "ret_std": s, "ret_se": s / len(SEEDS) ** 0.5, "bloss_mean": float(np.mean(bls)), "rets": rets}
            print(f"  [{tb}][{aname:13s}] ret={m:.3f}+-{s:.3f} (se {s/len(SEEDS)**0.5:.3f}) bloss={np.mean(bls):.2f}", flush=True)

    # SAVE before readout
    np.savez(NPZ, **{f"rets__{tb}__{a}": np.array(results[tb][a]["rets"]) for tb in TESTBS for (a, _) in ARMS})
    rep = {"pre_acc": pre_acc, "forget_precheck": forget, "second_taskB": second, "panel": panel,
           "guard_grass_res_curv": g_rc, "lam_curv": lam_curv, "lam_res": lam_res, "lam_randorthog": rand_lam, "results": results, "steps": STEPS}
    try:
        constr_ok = g_rc < 0.10
        verdicts = {}
        for tb in TESTBS:
            R = {a: results[tb][a] for (a, _) in ARMS}
            def gap_clears(a, b):  # a > b clearing 2*combined-SE
                d = R[a]["ret_mean"] - R[b]["ret_mean"]; cse = (R[a]["ret_se"] ** 2 + R[b]["ret_se"] ** 2) ** 0.5
                return d > 2 * cse, d, 2 * cse
            guard = (R["naive"]["ret_mean"] < pre_acc - 0.03) and (R["curv_full"]["ret_mean"] > R["naive"]["ret_mean"] + 0.03)
            adds, d32, t32 = gap_clears("curvres_full", "curv_full")
            spec, d35, t35 = gap_clears("curvres_full", "currand_full")
            alone, d41, t41 = gap_clears("res_full", "naive")
            if not constr_ok:
                v = f"VOID: RESIDUAL not orthogonal to CURV (Grassmann {g_rc:.3f}) -> construction failed"
            elif not guard:
                v = f"VOID: guard failed (naive {R['naive']['ret_mean']:.3f} vs pre {pre_acc:.3f}; curv {R['curv_full']['ret_mean']:.3f})"
            elif adds and spec:
                v = f"RESIDUAL PROTECTS SPECIFICALLY: curvres>curv (+{d32:.3f}>2SE {t32:.3f}) AND curvres>currand (+{d35:.3f}>2SE) -> beyond-curvature fragile structure protects beyond the lever & is not just more dims. NOVEL result survives."
            elif not adds:
                v = f"CLEAN NEGATIVE: curvres~curv (Δ{d32:+.3f}, 2SE {t32:.3f}) -> residual adds NOTHING beyond curvature; curvature lever captures all protectable structure. Caps lever at curvature."
            else:
                v = f"NOT-SPECIFIC: curvres>curv (+{d32:.3f}) BUT curvres~currand (Δ{d35:+.3f}) -> more dims help, residual not special; beyond-curvature story dies for protection."
            verdicts[tb] = {"verdict": v, "construction_ok": constr_ok, "guard": guard, "residual_adds": adds, "residual_specific": spec, "residual_alone": alone,
                            "vals": {a: R[a]["ret_mean"] for a in R}}
            print(f"\n[RES] === {tb} (pre={pre_acc:.3f}, forget {forget[tb]:.3f}) ===\n  {v}\n  res_alone>naive: {alone} (Δ{d41:+.3f}); lam match RES {lam_res:.2e} vs randorthog {rand_lam:.2e} (both off-curv, vs CURV {lam_curv:.2e})", flush=True)
        rep["verdicts"] = verdicts
    except Exception as e:
        import traceback; rep["error"] = traceback.format_exc(); print(f"[RES] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[RES] saved residual_test.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
