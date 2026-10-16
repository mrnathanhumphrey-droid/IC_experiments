"""OPTION 3 — does the universal fragile subspace survive a STRUCTURALLY DIFFERENT task-B?
The universal beyond-curvature aimed subspace was established across 3 task-Bs that are ALL short-text classification
({text}\\nLabel: framing). Live weakness: the shared subspace could be a property of that TASK FAMILY, not of task-A.
Adversarial test: bring in a maximally-different task-B (CODE GENERATION, MBPP) and ask whether it disturbs the SAME
fragile subspace as the 3 classification anchors.

task-B (answer-blind, declared before run): MBPP code generation. Maximizes structural distance on all 3 axes —
objective (multi-token generation vs 1-token softmax label), domain (Python code vs short NL text), framing
(`# {desc}\\n`->function body vs `{text}\\nLabel:`).

EFFICIENCY + RIGOR: the 3 anchors' per-layer seed/contest are LOADED from crosspair_residual.npz (not retrained) ->
anchors are bit-identical to option-1, so their mutual +0.64 reproduces automatically (no pipeline-regression risk) and
only the code task-B is trained. PA is recomputed and verified bit-identical to the anchors' basis (my lam == deflation
lam AND == crosspair stored lam) before any comparison.

MEASUREMENT (per-layer, NEVER pooled): build code task-B's contest + seed on the SAME PA; strip curvature
(partial_spear(.,.|lam)); compare code-vs-each-anchor (the TEST) against anchor-vs-anchor (the REFERENCE, must reproduce
~0.64) for BOTH objects (contest AND seed); dissociation-set overlap vs random null.
PRE-COMMITTED VERDICT (per object): code-vs-anchor ~ anchor-vs-anchor (>=0.30 and >=~0.7x ref) -> UNIVERSAL-STRONG (fragile
subspace is a property of TASK-A, survives structural change); code-vs-anchor ~0 while ref still ~0.64 -> FAMILY-SPECIFIC
(universal collapses to within-classification, the boundary is the finding); intermediate -> PARTIAL, report shared frac.
GUARD: if REFERENCE (anchor-anchor) does NOT reproduce ~0.6, ABORT verdict (pipeline regression). Save before readout.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"
DEFL = f"{G}/deflation_artifacts.npz"; ANCH = f"{G}/crosspair_residual.npz"
OUT = f"{G}/crosspair_opt3.json"; NPZ = f"{G}/crosspair_opt3.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
ANCHORS = ["agnews", "yelp", "subj"]; NEW = "code"
KTOP = 64; CORE = 16; N_FISHER = 150; N_SEED = 8
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst_trA():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:max(N_FISHER, 200)]]


def load_gen():
    """Structurally-different task-B: code-generation (sharpest) with robust fallback. Returns (rows, taskname)."""
    from datasets import load_dataset
    # 1) CODE generation via PLAIN-PARQUET community dataset (sidesteps the script-dataset URI bug that broke humaneval/mbpp).
    #    objective=multi-token gen, DOMAIN=python code (the sharp axis), framing=instruction->code (not {text}\nLabel:)
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
        rows = []
        for r in ds:
            instr = (r.get("instruction") or "").strip(); inp = (r.get("input") or "").strip(); out = (r.get("output") or "").strip()
            if instr and out:
                rows.append((instr + (("\n" + inp) if inp else "") + "\n", " " + out))
        rows = rows[:1500]
        if len(rows) >= 200:
            print(f"[OPT3] code task-B = python_code_instructions: {len(rows)} examples (gen / CODE / instruction->python)", flush=True)
            return rows, "python-code-instructions"
    except Exception as e:
        print(f"[OPT3] python_code dataset failed: {str(e)[:100]}", flush=True)
    # 2) humaneval completion (script-dataset; may hit the URI bug)
    try:
        ds = load_dataset("openai_humaneval")["test"]
        rows = [(r["prompt"], r["canonical_solution"]) for r in ds if r.get("prompt") and r.get("canonical_solution")]
        if len(rows) >= 100:
            print(f"[OPT3] code task-B = humaneval completion: {len(rows)} examples (gen / code / signature->body)", flush=True)
            return rows, "humaneval-code-completion"
    except Exception as e:
        print(f"[OPT3] humaneval failed: {str(e)[:100]}", flush=True)
    # 3) instruction-following generation (fallback: gen + different framing; NL domain = weaker contrast)
    try:
        ds = load_dataset("databricks/databricks-dolly-15k")["train"]
        rows = []
        for r in ds:
            if r.get("context"): continue
            instr = (r.get("instruction") or "").strip(); resp = (r.get("response") or "").strip()
            if instr and resp: rows.append((instr + "\n", " " + resp))
        rows = rows[:1000]
        print(f"[OPT3] code unavailable -> FALLBACK dolly instruction-gen: {len(rows)} examples (gen / NL / instruction->response; weaker contrast)", flush=True)
        return rows, "dolly-instruction-gen-FALLBACK"
    except Exception as e:
        print(f"[OPT3] dolly failed: {str(e)[:100]}", flush=True)
    raise RuntimeError("no structurally-different generation task-B could be loaded")


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
        p, c = p[:256], c[:256]
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


def train_taskB(model, tok, B):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0); run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        for i in range(0, max(1, len(B) - BATCH), BATCH):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); return float(np.mean(run)) if run else float("nan"), step


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def partial_spear(a, b, ctrl):
    ra, rb, rc = [np.argsort(np.argsort(x)).astype(float) for x in (a, b, ctrl)]
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]; beta = np.linalg.lstsq(x1, y, rcond=None)[0]; return y - x1 @ beta
    return float(np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1])


def main():
    t0 = time.time(); print("[OPT3] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    FA = fisher(model, params, names, tok, trA, False)
    PA = {}; LAM = {}
    for n in names:
        ev, V = torch.linalg.eigh(FA[n]); PA[n] = V[:, -KTOP:].contiguous(); LAM[n] = ev[-KTOP:].cpu().numpy()
    print(f"[OPT3] FA + PA done ({time.time()-t0:.0f}s)", flush=True)

    # NEW task-B = structurally-different generation: seed @ w_A + train + contest
    Bc, taskname = load_gen()
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    model.eval(); model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(0)
    for _ in range(N_SEED):
        inp, lb, am = b_pair(tok, [Bc[k] for k in torch.randperm(len(Bc), generator=g)[:BATCH].tolist()])
        (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_SEED).backward()
    seed_code = {n: (params[n].grad.detach().double() @ PA[n].double()).pow(2).sum(0).cpu().numpy() for n in names}
    model.zero_grad(set_to_none=True)
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    loss, steps = train_taskB(model, tok, Bc)
    FB = fisher(model, params, names, tok, Bc, True)
    contest_code = {n: (torch.linalg.eigh(FB[n])[1][:, -KTOP:].T @ PA[n]).pow(2).sum(0).cpu().numpy() for n in names}
    print(f"[OPT3] code: seed+train(loss={loss:.3f},steps={steps})+contest done ({time.time()-t0:.0f}s)", flush=True)

    # ---- SAVE camera frame FIRST ----
    arts = {"code_loss": np.array([loss]), "code_steps": np.array([steps])}
    for c in CARRIERS:
        cn = [n for n in names if c in n]
        arts[f"seed__code__{c}"] = np.stack([seed_code[n] for n in cn])
        arts[f"contest__code__{c}"] = np.stack([contest_code[n] for n in cn])
        arts[f"mylam__{c}"] = np.stack([LAM[n] for n in cn]); arts[f"names__{c}"] = np.array(cn)
    np.savez(NPZ, **arts)
    print(f"[OPT3] saved camera frame crosspair_opt3.npz ({time.time()-t0:.0f}s)", flush=True)

    report = {}
    try:
        defl = np.load(DEFL); anch = np.load(ANCH, allow_pickle=True); rng = np.random.default_rng(0)
        # GUARDS: my PA basis == deflation's == anchors'
        align_d = max(float(np.max(np.abs(LAM[n] - defl[f"lam__{n}"]) / (np.abs(defl[f"lam__{n}"]) + 1e-12))) for n in names)
        align_a = 0.0
        for c in CARRIERS:
            cn = [n for n in names if c in n]; stored = anch[f"lam__{c}"]
            for li, n in enumerate(cn):
                align_a = max(align_a, float(np.max(np.abs(LAM[n] - stored[li]) / (np.abs(stored[li]) + 1e-12))))
        # code M1 (sanity it forms a core like the anchors did)
        m1c = []
        for n in names:
            ci = set(np.argsort(contest_code[n])[-CORE:].tolist()); curv = set(range(KTOP - CORE, KTOP)); m1c.append(len(ci & curv) / CORE)
        print(f"[OPT3] GUARD align(vs deflation)={align_d:.1e} align(vs anchors)={align_a:.1e}  code_M1(vs curv)={np.mean(m1c):.2f}", flush=True)

        def get(obj_anch, obj_code, tk, c, li, n):
            return obj_code[n] if tk == NEW else anch[f"{obj_anch}__{tk}__{c}"][li]

        def analyze(obj_anch, obj_code, label):
            res = {}; lowcurv = list(range(KTOP // 2))
            for c in CARRIERS:
                cn = [n for n in names if c in n]
                ref, test = [], []          # anchor-anchor (reference) ; code-anchor (test), per layer
                jac, jz = [], []
                for li, n in enumerate(cn):
                    lam = defl[f"lam__{n}"].astype(float)
                    prof = {tk: get(obj_anch, obj_code, tk, c, li, n).astype(float) for tk in ANCHORS + [NEW]}
                    for (i, j) in [("agnews", "yelp"), ("agnews", "subj"), ("yelp", "subj")]:
                        ref.append(partial_spear(prof[i], prof[j], lam))
                    for a in ANCHORS:
                        test.append(partial_spear(prof[NEW], prof[a], lam))
                    # dissoc: code's high-value low-curv set vs each anchor's
                    cs = set(np.argsort(prof[NEW])[-CORE:].tolist()) & set(lowcurv)
                    for a in ANCHORS:
                        as_ = set(np.argsort(prof[a])[-CORE:].tolist()) & set(lowcurv)
                        if cs and as_:
                            jc = len(cs & as_) / len(cs | as_); jac.append(jc)
                            nl = np.array([len(set(rng.choice(lowcurv, len(cs), False)) & set(rng.choice(lowcurv, len(as_), False))) /
                                           max(1, len(set(rng.choice(lowcurv, len(cs), False)) | set(rng.choice(lowcurv, len(as_), False)))) for _ in range(400)])
                            jz.append((jc - nl.mean()) / (nl.std() + 1e-12))
                res[c] = {"ref_mean": float(np.mean(ref)), "test_mean": float(np.mean(test)),
                          "test_pos": int((np.array(test) > 0).sum()), "test_n": len(test),
                          "dissoc_jac": float(np.mean(jac)) if jac else float("nan"), "dissoc_z": float(np.mean(jz)) if jz else float("nan"),
                          "test_per": test}
            gref = float(np.mean([res[c]["ref_mean"] for c in CARRIERS]))
            gtest = float(np.mean([res[c]["test_mean"] for c in CARRIERS]))
            allt = np.array([p for c in CARRIERS for p in res[c]["test_per"]]); gpos = int((allt > 0).sum())
            frac = gtest / gref if gref > 0.05 else float("nan")
            if gref < 0.4:
                v = f"ABORT/REGRESSION: anchor-anchor reference {label}={gref:+.2f} did NOT reproduce ~0.64 -> pipeline broke; verdict void."
            elif gtest >= 0.30 and gtest >= 0.7 * gref:
                v = f"UNIVERSAL-STRONG: code-vs-anchor {label}={gtest:+.2f} ~ anchor-anchor {gref:+.2f} ({frac:.0%}), {gpos}/{len(allt)} pos -> fragile subspace is a property of TASK-A, survives a structurally-different task-B. Strongest mechanism claim."
            elif gtest <= 0.12:
                v = f"FAMILY-SPECIFIC: code-vs-anchor {label}={gtest:+.2f} ~0 while anchors agree {gref:+.2f} -> subspace is task-FAMILY-specific; 'universal' = within-classification. The boundary IS the finding."
            else:
                v = f"PARTIAL: code-vs-anchor {label}={gtest:+.2f} vs anchor-anchor {gref:+.2f} (shared frac {frac:.0%}), {gpos}/{len(allt)} pos -> partly task-A-property, partly family-specific."
            res["grand"] = {"ref_mean": gref, "test_mean": gtest, "shared_frac": frac, "test_pos": gpos, "test_n": len(allt), "verdict": v}
            print(f"\n[OPT3] === {label.upper()} ===", flush=True)
            for c in CARRIERS:
                r = res[c]; print(f"  [{c:26s}] anchor-anchor(ref)={r['ref_mean']:+.2f}  code-anchor(test)={r['test_mean']:+.2f}  test pos={r['test_pos']}/{r['test_n']}  dissoc Jac={r['dissoc_jac']:.2f}(z{r['dissoc_z']:+.1f})", flush=True)
            print(f"  GRAND ref={gref:+.2f} test={gtest:+.2f} frac={frac:.0%} pos={gpos}/{len(allt)}\n  VERDICT: {v}", flush=True)
            return res

        report["guards"] = {"align_vs_deflation": align_d, "align_vs_anchors": align_a, "code_M1": float(np.mean(m1c)),
                            "code_loss": loss, "code_steps": steps, "code_n_examples": len(Bc)}
        report["contest"] = analyze("contest", contest_code, "contest")
        report["seed"] = analyze("seed", seed_code, "seed")
        cv = report["contest"]["grand"]["verdict"].split(":")[0]; sv = report["seed"]["grand"]["verdict"].split(":")[0]
        report["agree"] = f"contest:{cv} | seed:{sv}"
        print(f"\n[OPT3] OBJECT AGREEMENT: {report['agree']}", flush=True)
    except Exception as e:
        import traceback; report["error"] = traceback.format_exc(); print(f"[OPT3] analysis error (camera frame saved): {e}", flush=True)
    json.dump({"report": report, "task_b": taskname, "ktop": KTOP, "core": CORE, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[OPT3] saved crosspair_opt3.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
