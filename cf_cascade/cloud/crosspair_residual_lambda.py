"""CROSS-PAIR BEYOND-CURVATURE RESIDUAL — is the aimed-forgetting component UNIVERSAL or PER-PAIR?
The seed<->core flip showed (for AG) the seed tracks the contested core beyond curvature (+0.45 partial, 72/72 layers).
That beyond-curvature structure: is it the SAME across task-Bs (a fixed aimed-forgetting subspace, mechanism) or unique
to each pair (per-task-B aiming, description)?

Per-task-B cores were NOT saved (deflation kept only cB.mean(0)) -> recompute per task-B. Two objects, one run:
  CONTEST_B = (PB^T PA)^2 per layer (PB = task-B top-64 Fisher eigvecs @ w_B; PA = task-A curvature basis @ w_A) -- "where tasks compete"
  SEED_B    = (grad_B @ PA)^2 per layer (task-B step-0 gradient @ w_A)                                       -- "where forgetting aims"
Both in the SHARED PA basis per layer (all task-Bs projected onto task-A's PA) -> cross-pair comparison is well-defined.

MEASUREMENT (per layer, NEVER pooled -- pooling manufactured the false null last time):
  strip curvature via partial_spear(prof_i, prof_j | lam) for each pair (AG-Yelp, AG-Subj, Yelp-Subj); distribution across 24 layers.
  + dissociation-set overlap: each task-B's high-contest-LOW-curv dirs; do pairs contest the SAME ones? (Jaccard vs random-draw null.)
  Grassmann is VACUOUS here (shared PA basis -> axis-aligned -> Grassmann==index overlap); use profile-partial + dissoc-overlap as the
  two non-vacuous corroborators (same substitution as the seed<->core test, flagged).
GUARDS before verdict: (1) align = my PA-eigenvalues vs deflation lam__ (==0); (2) mean-contest reproduce = mean over my 3 contests
  vs deflation's saved contest__ (the mean it kept) -- confirms I recompute the identical object, just per-task-B now.
PRE-COMMITTED VERDICT (per object):
  cross-pair partial HIGH (t>3, consistent across layers, >>chance) -> UNIVERSAL aimed component (fixed beyond-curv subspace all share).
  cross-pair partial ~0 (t<2)                                       -> PER-PAIR (only curvature shared; aiming is pair-specific).
  intermediate                                                       -> partly-shared + per-pair margin; report shared fraction.
Saves all per-task-B profiles + cross-pair stats BEFORE readout.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
DEFL = f"{G}/deflation_artifacts.npz"
OUT = f"{G}/crosspair_residual.json"; NPZ = f"{G}/crosspair_residual.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
TASKS = [("agnews", "low_agnews"), ("yelp", "high_yelp"), ("subj", "medium_subj")]
KTOP = 64; CORE = 16; N_FISHER = 150; N_SEED = 8
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
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


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def partial_spear(a, b, ctrl):
    ra, rb, rc = [np.argsort(np.argsort(x)).astype(float) for x in (a, b, ctrl)]
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]; beta = np.linalg.lstsq(x1, y, rcond=None)[0]; return y - x1 @ beta
    return float(np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1])


def main():
    t0 = time.time(); print("[CPR] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    FA = fisher(model, params, names, tok, trA, False)               # task-A curvature @ w_A
    PA = {}; LAM = {}
    for n in names:
        ev, V = torch.linalg.eigh(FA[n]); PA[n] = V[:, -KTOP:].contiguous(); LAM[n] = ev[-KTOP:].cpu().numpy()
    print(f"[CPR] FA + PA done ({time.time()-t0:.0f}s)", flush=True)

    seed_B = {tk: {} for tk, _ in TASKS}; contest_B = {tk: {} for tk, _ in TASKS}
    for taskkey, jsonname in TASKS:
        B = load_taskB(jsonname)
        # SEED (step 0) @ w_A
        model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
        model.eval(); model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(0)
        for _ in range(N_SEED):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
            (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_SEED).backward()
        for n in names: seed_B[taskkey][n] = (params[n].grad.detach().double() @ PA[n].double()).pow(2).sum(0).cpu().numpy()
        model.zero_grad(set_to_none=True)
        # TRAIN to convergence -> FB -> contest
        model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
        loss, steps = train_taskB(model, tok, B)
        FB = fisher(model, params, names, tok, B, True)
        for n in names:
            PB = torch.linalg.eigh(FB[n])[1][:, -KTOP:]
            contest_B[taskkey][n] = (PB.T @ PA[n]).pow(2).sum(0).cpu().numpy()
        print(f"[CPR] {taskkey}: seed+train(loss={loss:.3f},steps={steps})+contest done ({time.time()-t0:.0f}s)", flush=True)

    # ---- SAVE camera frame FIRST ----
    arts = {}
    for c in CARRIERS:
        cn = [n for n in names if c in n]
        arts[f"lam__{c}"] = np.stack([LAM[n] for n in cn]); arts[f"names__{c}"] = np.array(cn)
        for tk, _ in TASKS:
            arts[f"seed__{tk}__{c}"] = np.stack([seed_B[tk][n] for n in cn])
            arts[f"contest__{tk}__{c}"] = np.stack([contest_B[tk][n] for n in cn])
    np.savez(NPZ, **arts)
    print(f"[CPR] saved camera frame crosspair_residual.npz ({time.time()-t0:.0f}s)", flush=True)

    # ---- analysis (wrapped) ----
    report = {}
    try:
        defl = np.load(DEFL); rng = np.random.default_rng(0); NPERM = 2000
        PAIRS = [("agnews", "yelp"), ("agnews", "subj"), ("yelp", "subj")]
        # GUARDS
        align = {}; meanrep = {}
        for c in CARRIERS:
            cn = [n for n in names if c in n]
            align[c] = max(float(np.max(np.abs(LAM[n] - defl[f"lam__{n}"]) / (np.abs(defl[f"lam__{n}"]) + 1e-12))) for n in cn)
            sp = [spear((contest_B["agnews"][n] + contest_B["yelp"][n] + contest_B["subj"][n]) / 3.0, defl[f"contest__{n}"].astype(float)) for n in cn]
            meanrep[c] = float(np.mean(sp))
        print(f"[CPR] GUARD align(max reldiff)={max(align.values()):.1e}  mean-contest reproduce spearman={np.mean(list(meanrep.values())):+.3f}", flush=True)

        def analyze(OBJ, label):
            res = {}
            for c in CARRIERS:
                cn = [n for n in names if c in n]
                pairstats = {}
                for (i, j) in PAIRS:
                    parts, raws = [], []
                    for n in cn:
                        lam = defl[f"lam__{n}"].astype(float)
                        parts.append(partial_spear(OBJ[i][n], OBJ[j][n], lam)); raws.append(spear(OBJ[i][n], OBJ[j][n]))
                    pa = np.array(parts); t = float(pa.mean() / (pa.std(ddof=1) / np.sqrt(len(pa)) + 1e-12))
                    pairstats[f"{i}-{j}"] = {"partial_mean": float(pa.mean()), "partial_t": t, "n_pos": int((pa > 0).sum()),
                                             "raw_mean": float(np.mean(raws)), "partial_per_layer": parts}
                # dissociation-set cross-pair overlap (residual = high-value & low-curvature dirs)
                lowcurv = list(range(KTOP // 2))
                jac, jz = [], []
                for n in cn:
                    sets = {}
                    for tk, _ in TASKS:
                        top = set(np.argsort(OBJ[tk][n])[-CORE:].tolist()); sets[tk] = top & set(lowcurv)
                    for (i, j) in PAIRS:
                        a, b = sets[i], sets[j]
                        if a and b:
                            jc = len(a & b) / len(a | b); jac.append(jc)
                            null = np.array([len(set(rng.choice(lowcurv, len(a), False)) & set(rng.choice(lowcurv, len(b), False))) /
                                             max(1, len(set(rng.choice(lowcurv, len(a), False)) | set(rng.choice(lowcurv, len(b), False)))) for _ in range(500)])
                            jz.append((jc - null.mean()) / (null.std() + 1e-12))
                res[c] = {"pairs": pairstats, "dissoc_jaccard_mean": float(np.mean(jac)) if jac else float("nan"),
                          "dissoc_jaccard_z": float(np.mean(jz)) if jz else float("nan")}
            allp = np.array([p for c in CARRIERS for (i, j) in PAIRS for p in res[c]["pairs"][f"{i}-{j}"]["partial_per_layer"]])
            gmean = float(allp.mean()); gt = float(allp.mean() / (allp.std(ddof=1) / np.sqrt(len(allp)) + 1e-12)); gpos = int((allp > 0).sum())
            graw = float(np.mean([res[c]["pairs"][f"{i}-{j}"]["raw_mean"] for c in CARRIERS for (i, j) in PAIRS]))
            # EFFECT-SIZE discriminator (t is n-inflated at n=216 -> useless as discriminator; keep only as 'reliably>0').
            # Benchmark: within-pair seed<->core beyond-curv partial was +0.45 = "fully shares the structure". frac vs that.
            REF = 0.45; frac = gmean / REF
            if gmean >= 0.30 and gpos > 0.7 * len(allp):
                v = (f"UNIVERSAL: cross-pair {label} partial mean={gmean:+.2f} (={frac:.0%} of within-pair {REF}), t={gt:+.0f} "
                     f"{gpos}/{len(allp)} pos, raw={graw:.2f} -> fixed beyond-curvature aimed subspace shared across task-Bs. MECHANISM.")
            elif gmean <= 0.12:
                v = (f"PER-PAIR: cross-pair {label} partial mean={gmean:+.2f} (={frac:.0%} of within-pair {REF}) -- NEAR-ZERO effect "
                     f"(t={gt:+.0f} is n-inflated, ignore) -> beyond-curv aiming is pair-specific; only curvature shared. DESCRIPTION.")
            else:
                v = (f"PARTLY-SHARED: cross-pair {label} partial mean={gmean:+.2f} (={frac:.0%} of within-pair {REF}), raw={graw:.2f} "
                     f"-> partial universal aimed component + per-pair margin (shared fraction ~{frac:.0%}).")
            res["grand"] = {"partial_mean": gmean, "partial_t": gt, "n_pos": gpos, "n": len(allp), "raw_mean": graw, "verdict": v}
            print(f"\n[CPR] === {label.upper()} ===", flush=True)
            for c in CARRIERS:
                ps = " ".join(f"{k}={res[c]['pairs'][k]['partial_mean']:+.2f}(t{res[c]['pairs'][k]['partial_t']:+.0f})" for k in res[c]['pairs'])
                print(f"  [{c:26s}] {ps} | dissoc Jac={res[c]['dissoc_jaccard_mean']:.2f}(z{res[c]['dissoc_jaccard_z']:+.1f})", flush=True)
            print(f"  GRAND raw={graw:.2f} partial={gmean:+.2f} t={gt:+.1f} pos={gpos}/{len(allp)}\n  VERDICT: {v}", flush=True)
            return res

        report["guards"] = {"align_max_reldiff": max(align.values()), "mean_contest_reproduce_spear": float(np.mean(list(meanrep.values())))}
        report["contest"] = analyze(contest_B, "contest")
        report["seed"] = analyze(seed_B, "seed")
        report["agree"] = (f"contest:{report['contest']['grand']['verdict'].split(':')[0]} | seed:{report['seed']['grand']['verdict'].split(':')[0]}")
        print(f"\n[CPR] OBJECT AGREEMENT: {report['agree']}", flush=True)
    except Exception as e:
        import traceback; report["error"] = traceback.format_exc(); print(f"[CPR] analysis error (camera frame saved): {e}", flush=True)
    json.dump({"report": report, "ktop": KTOP, "core": CORE, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[CPR] saved crosspair_residual.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
