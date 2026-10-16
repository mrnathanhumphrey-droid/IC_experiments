"""CUT 1 — VARY TASK-A for the first time. Does the attention fragile subspace MOVE with the task or stay BOLTED to
the model (Pythia)? task-A was SST-2 the entire program. If the universal attention fragile subspace moves when task-A
changes -> it's SST-2's property (the lead as named). If it stays put -> it's Pythia attention's INTRINSIC fragility,
forgetting was just the probe (the reorganization).

FIRST CROSS-BASIS comparison in the program -> two methodology flips:
 (1) Grassmann is NON-vacuous now (different curvature bases per task-A -> genuinely different ambient vectors). Index
     overlap is ILL-DEFINED across bases (rank j differs by basis) -> use Grassmann (+ a coarse >0.7-cosine match frac).
 (2) NEW confound: two Pythia-attention FA may share eigenstructure from the ARCHITECTURE alone. Control: compare the
     FRAGILE-subspace overlap vs the generic TOP-CURVATURE-subspace overlap (same task-A pair). random null = floor;
     topcurv-overlap = architectural reference. fragile>>topcurv -> specific; fragile~topcurv -> generic-architectural.

task-As (answer-blind): sst2 (existing ckpt, reference) ; AG-News (diff task, same family, control) ; UCDP-GED
type_of_violence (MAX distance — conflict-event narrative -> 3-class violence typology, the user's substrate, clean basin).
Each new task-A: fine-tune Pythia BASE -> task-A min (report convergence; all 3 classification -> clean basins, freeze gate OK).
Fragile subspace of task-A = top-16 PA-columns by the UNIVERSAL seed (mean over a probe panel), as ambient weight vectors.
Probes (seed @ w_A, step-0, convergence-CLEAN per the lesson): sst2<-{ag,yelp,subj}(reproduces known=guard); ag,code<-{yelp,subj,sst2}.
DECISIVE: per layer per carrier, Grassmann(fragile_sst2, fragile_A) vs Grassmann(topcurv_sst2, topcurv_A) vs random null.
MLP-vs-attention CONTRAST: prediction (from erosion result) = attention stays (architectural), MLP moves (task-set).
PRE-COMMITTED VERDICT: fragile attention Grassmann >> random and high across layers -> STAYS PUT = architectural (lead
 reorganizes to intrinsic attention fragility). ~random -> MOVES = task-A's property (lead holds). intermediate -> fixed
 architectural core + task-A margin (report split). Plus: if fragile~topcurv it's generic curvature; if fragile>topcurv specific.
GUARDS: align(my PA_sst eig vs deflation lam)=0 (no-regression); dataset-identity asserts; report each task-A convergence.
Save fragile/topcurv subspaces + overlaps BEFORE readout.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch, copy
G = "/home/ubuntu/gateA"
BASE = "EleutherAI/pythia-410m"; SST2_CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
DEFL = f"{G}/deflation_artifacts.npz"
OUT = f"{G}/taskA_variation.json"; NPZ = f"{G}/taskA_variation.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
KTOP = 64; FRAG = 16; N_FISHER = 150; N_SEED = 8
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True
# task-A -> probe panel (probes define the universal fragile subspace)
PANEL = {"sst2": ["ag", "yelp", "subj"], "ag": ["yelp", "subj", "sst2"], "ucdp": ["yelp", "subj", "sst2"]}
JSONK = {"ag": "low_agnews", "yelp": "high_yelp", "subj": "medium_subj", "ucdp": "ucdp_violence"}


def sst_trA(seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    assert len(ds) > 60000, f"sst2 identity: got {len(ds)}"
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:400]]


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def taskdata(key):
    """returns (list[(prompt,completion)], is_pair_format) for any task key, with identity assert."""
    if key == "sst2":
        rows = sst_trA(1); return [(f"{s}\nSentiment:", LAB[l]) for s, l in rows], True
    d = load_jsonB(JSONK[key]); assert len(d) > 50, f"{key} identity: got {len(d)}"; return d, True


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
    for it in items[:N_FISHER]:
        inp, lb, am = b_pairs(tok, [it]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def train(model, tok, A):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0); run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        for i in range(0, max(1, len(A) - BATCH), BATCH):
            inp, lb, am = b_pairs(tok, [A[k] for k in torch.randperm(len(A), generator=g)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); return float(np.mean(run)) if run else float("nan"), step


def grass(A, B):  # A (D,m), B (D,k) orthonormal cols -> mean squared canonical cosine in [0,1]
    M = A.T @ B; s = np.linalg.svd(M, compute_uv=False); return float((s ** 2).sum() / A.shape[1])


def match_frac(A, B, thr=0.7):  # frac of A-cols with a >thr cosine to some B-col (index-overlap analog)
    C = np.abs(A.T @ B); return float((C.max(1) > thr).mean())


def main():
    t0 = time.time(); print("[TAV] loading base + tokenizer...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(dev)
    base_sd = copy.deepcopy(model.state_dict())
    params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    print(f"[TAV] base loaded, {len(names)} carriers ({time.time()-t0:.0f}s)", flush=True)

    # cache probe datasets once
    probe_data = {}
    for k in ["ag", "yelp", "subj", "sst2"]:
        probe_data[k], _ = taskdata(k)
        print(f"[TAV] probe '{k}': {len(probe_data[k])} examples", flush=True)

    fragile = {}; topcurv = {}; conv = {}; align_sst = None
    for taskA in ["sst2", "ag", "ucdp"]:
        # --- get w_A ---
        if taskA == "sst2":
            model.load_state_dict(torch.load(SST2_CKPT)); conv[taskA] = ("loaded_ckpt", 0)
            faA = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_trA(0)]   # SST-2 Fisher data (matches prior; b_pairs==b_sst here)
        else:
            model.load_state_dict(base_sd); Adata, _ = taskdata(taskA)
            loss, steps = train(model, tok, Adata); conv[taskA] = (loss, steps); faA = Adata
            print(f"[TAV] task-A={taskA} trained loss={loss:.3f} steps={steps} ({time.time()-t0:.0f}s)", flush=True)
        params = dict(model.named_parameters())
        # --- curvature basis PA at w_A ---
        FA = fisher(model, params, names, tok, faA)
        PA = {}; LAMA = {}
        for n in names:
            ev, V = torch.linalg.eigh(FA[n]); PA[n] = V[:, -KTOP:].contiguous(); LAMA[n] = ev[-KTOP:].cpu().numpy()
        if taskA == "sst2":
            defl = np.load(DEFL); align_sst = max(float(np.max(np.abs(LAMA[n] - defl[f"lam__{n}"]) / (np.abs(defl[f"lam__{n}"]) + 1e-12))) for n in names)
            print(f"[TAV] GUARD align(PA_sst vs deflation)={align_sst:.1e}", flush=True)
        # --- universal seed over probe panel ---
        wA = {n: params[n].detach().clone() for n in names}
        useed = {n: np.zeros(KTOP) for n in names}; npr = 0
        for pk in PANEL[taskA]:
            model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(0); B = probe_data[pk]
            for _ in range(N_SEED):
                inp, lb, am = b_pairs(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
                (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_SEED).backward()
            for n in names: useed[n] += (params[n].grad.detach().double() @ PA[n].double()).pow(2).sum(0).cpu().numpy()
            model.zero_grad(set_to_none=True); npr += 1
        # --- fragile subspace (top-FRAG universal-seed PA cols) + topcurv subspace (top-FRAG curvature) as AMBIENT vecs ---
        fragile[taskA] = {}; topcurv[taskA] = {}
        for n in names:
            u = useed[n] / npr; frag_ranks = np.argsort(u)[-FRAG:]
            fragile[taskA][n] = PA[n][:, frag_ranks].cpu().numpy().astype(np.float64)     # (in, 16)
            topcurv[taskA][n] = PA[n][:, -FRAG:].cpu().numpy().astype(np.float64)          # (in, 16) highest curv
        print(f"[TAV] task-A={taskA} fragile+topcurv subspaces built ({time.time()-t0:.0f}s)", flush=True)

    # ---- SAVE camera frame FIRST ----
    arts = {}
    for taskA in ["sst2", "ag", "ucdp"]:
        for c in CARRIERS:
            cn = [n for n in names if c in n]
            arts[f"fragile__{taskA}__{c}"] = np.stack([fragile[taskA][n] for n in cn])   # (24, in, 16)
            arts[f"topcurv__{taskA}__{c}"] = np.stack([topcurv[taskA][n] for n in cn])
    np.savez(NPZ, **arts)
    print(f"[TAV] saved camera frame taskA_variation.npz ({time.time()-t0:.0f}s)", flush=True)

    # ---- analysis ----
    report = {"convergence": {k: {"loss": conv[k][0], "steps": conv[k][1]} for k in conv}, "align_sst_vs_deflation": align_sst}
    try:
        rng = np.random.default_rng(0); NRAND = 200
        def randsub(D, m):
            Q, _ = np.linalg.qr(rng.standard_normal((D, m))); return Q[:, :m]
        for newA in ["ag", "ucdp"]:
            report[newA] = {}
            for c in CARRIERS:
                cn = [n for n in names if c in n]
                fg, tc, mf, zf = [], [], [], []
                for n in cn:
                    Fs, Fa = fragile["sst2"][n], fragile[newA][n]; Ts, Ta = topcurv["sst2"][n], topcurv[newA][n]
                    gfrag = grass(Fs, Fa); gtop = grass(Ts, Ta)
                    null = np.array([grass(Fs, randsub(Fs.shape[0], FRAG)) for _ in range(NRAND)])
                    fg.append(gfrag); tc.append(gtop); mf.append(match_frac(Fs, Fa))
                    zf.append((gfrag - null.mean()) / (null.std() + 1e-12))
                report[newA][c] = {"grass_fragile_mean": float(np.mean(fg)), "grass_topcurv_mean": float(np.mean(tc)),
                                   "z_fragile_vs_random": float(np.mean(zf)), "match_frac_mean": float(np.mean(mf)),
                                   "grass_fragile_per_layer": fg, "chance_grass": FRAG / cn[0].__len__() if False else FRAG / fragile["sst2"][cn[0]].shape[0]}
                r = report[newA][c]
                print(f"  [sst2 vs {newA}][{c:26s}] fragile G={r['grass_fragile_mean']:.2f}(z{r['z_fragile_vs_random']:+.0f}) topcurv G={r['grass_topcurv_mean']:.2f} match={r['match_frac_mean']:.2f} chance~{r['chance_grass']:.3f}", flush=True)
        # verdict (attention = the lead carrier; MLP = contrast)
        def carrier_g(newA, c): return report[newA][c]["grass_fragile_mean"]
        att_g = float(np.mean([carrier_g(a, "attention.query_key_value") for a in ["ag", "ucdp"]]))
        att_z = float(np.mean([report[a]["attention.query_key_value"]["z_fragile_vs_random"] for a in ["ag", "ucdp"]]))
        att_tc = float(np.mean([report[a]["attention.query_key_value"]["grass_topcurv_mean"] for a in ["ag", "ucdp"]]))
        mlp_g = float(np.mean([carrier_g(a, c) for a in ["ag", "ucdp"] for c in ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]]))
        if att_z < 3:
            stay = f"ATTENTION MOVES (fragile G={att_g:.2f} z{att_z:+.0f}~random) -> TASK-A's property. Lead holds as named (across-task-B invariance within an anchor)."
        elif att_g >= 0.6:
            stay = f"ATTENTION STAYS PUT (fragile G={att_g:.2f} z{att_z:+.0f}>>random, bulk of dirs shared) -> ARCHITECTURAL: Pythia attention has a fixed fragile subspace independent of task-A. Lead REORGANIZES to intrinsic attention fragility."
        else:
            stay = f"ATTENTION PARTIAL (fragile G={att_g:.2f} z{att_z:+.0f}>>random but not full) -> fixed architectural core + task-A margin (shared frac ~{att_g:.0%})."
        spec = "specific(fragile>topcurv)" if att_g > att_tc + 0.1 else ("generic-curvature(fragile~topcurv)" if abs(att_g - att_tc) <= 0.1 else "fragile<topcurv(odd)")
        contrast = f"attention G={att_g:.2f} vs MLP G={mlp_g:.2f} -> " + ("attention MORE architectural than MLP (matches erosion: attention stays / MLP moves)" if att_g > mlp_g + 0.05 else "attention ~ MLP (no carrier split)")
        report["verdict"] = {"attention_fragile_G": att_g, "attention_z": att_z, "attention_topcurv_G": att_tc, "mlp_fragile_G": mlp_g,
                             "stay_or_move": stay, "specific_or_generic": spec, "carrier_contrast": contrast}
        print(f"\n[TAV] VERDICT: {stay}\n  {spec}\n  {contrast}", flush=True)
    except Exception as e:
        import traceback; report["error"] = traceback.format_exc(); print(f"[TAV] analysis error (camera frame saved): {e}", flush=True)
    json.dump({"report": report, "ktop": KTOP, "frag": FRAG, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[TAV] saved taskA_variation.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
