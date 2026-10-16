"""POKE TEST — pre-load the fragile subspace before task-B: SINK (worse, primed channel) or RESOURCE (deflect)?
The program found+characterized the fragile/contested core but never POKED it. This deliberately pre-loads structure
into task-A's fragile directions before task-B, and watches what task-B's surge does.

** CONSTRUCTION CRUX (the naive matched-magnitude poke is CONFOUNDED): fragile is ~10^6x stiffer than the controls
   (lam_fragile 0.105 vs lam_random 1.9e-8 vs lam_complement 6e-8, from residual_test). A matched-MAGNITUDE weight poke
   is non-destructive on random/complement but CATASTROPHIC on fragile -> B's "SINK" would be a trivial damage artifact,
   and matched-magnitude + non-destructive are JOINTLY UNSATISFIABLE (program's known fragile=stiffest obstruction).
   FIX = run BOTH constructible designs (user chose "fire both"): **

DESIGN 1 — FUNCTIONAL OCCUPY (non-destructive, budget-matched): phase-0 = extra task-A (SST) training, gradient
  RESTRICTED to the target subspace (keep only U-component: g <- (g@U)@U^T). Non-destructive by construction (it's
  task-A, loss can't spike); matched on STEPS not magnitude. Deepens task-A's commitment in the target dirs.
DESIGN 2 — DAMAGE-MATCHED PERTURB: phase-0 = inject dW = alpha * (R@U)@U^T (R fixed random), alpha binary-searched
  PER-ARM so each causes the SAME small task-A loss increase. Matched non-destructiveness, NOT magnitude (fragile alpha
  tiny, complement alpha huge -> quantifies the 10^6x curvature gap). Tests "given equal small damage, does WHERE matter."

ARMS (both designs): A no-poke (floor) · B poke-FRAGILE (top-r input-curv of FA@w_A) · C poke-RANDOM (matched-size random
  subspace) · D poke-COMPLEMENT (bottom-r, the flat dirs task-B flees to). Phase-2 = depth-train subj (whole net, 250
  steps), identical across arms. Held-out subj (in-neighborhood, real range, comparable to #2/cascade). 7 paired seeds.
DECISIVE: B vs A (poke change forgetting? more=DEFLECT/resource, less=SINK/primed, ~=inert) · B vs C (fragile-specific?)
  · B vs D (fragile vs escape-route). GUARDS: poke non-destructive (post-poke SST ~ pre, else void) · matched (steps D1 /
  damage D2) · arm A forgets >=0.10 · task-B not starved · reproduce FA (align-style) · per-seed saved.
PRE-REG: sign genuinely unpredictable. Cross-arch note (OPT later): predict RESOURCE/deflect likelier on OPT (orthogonal
  core, deposit survives) than Pythia (antagonistic core, deposit eroded -> SINK/null). This is Pythia (the antagonistic case).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/poke_test.json"; NPZ = f"{G}/poke_test.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7))
R = 64                       # fragile subspace size (= deflation KTOP / contested-core scale)
STEPS_POKE = 200; LR_POKE = 5e-5            # design-1 occupy budget
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; BATCH = 16
N_FISHER = 150; N_EVAL = 200
TARGET_DLOSS = 0.05         # design-2 matched task-A loss increase
torch.backends.cudnn.benchmark = True

def sst_data():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    tr = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)]
    trA = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:max(N_FISHER, 256)]]
    va = [(r["sentence"].strip(), LAB[int(r["label"])]) for r in ds["validation"] if r["label"] in (0, 1)][:N_EVAL]
    return trA, va
def load_taskB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    ex = [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]
    comps = sorted(set(e["completion"] for e in d["examples"]))
    eval_raw = [(e["text"], e["completion"]) for e in d["examples"][-N_EVAL:]]
    return ex[:-N_EVAL], eval_raw, t, comps
def pack(tok, seqs, labs):
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)
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
def fisher(model, params, names, tok, items):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for it in items[:N_FISHER]:
        inp, lb, am = b_sst(tok, [it]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F
@torch.no_grad()
def clf_acc(model, tok, raw_items, template, completions):
    model.eval(); cand_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in completions]; correct = 0
    for text, true in raw_items:
        p_ids = tok(template.format(text=text), add_special_tokens=False)["input_ids"]; best, best_lp = None, -1e30
        for c, c_ids in zip(completions, cand_ids):
            inp = torch.tensor([p_ids + c_ids]).to(dev); logits = model(input_ids=inp).logits[0]
            lp = sum(torch.log_softmax(logits[len(p_ids) + j - 1], -1)[tid].item() for j, tid in enumerate(c_ids))
            if lp > best_lp: best_lp, best = lp, c
        correct += int(best == true)
    return correct / len(raw_items)
@torch.no_grad()
def sst_loss(model, tok, items):
    model.eval(); inp, lb, am = b_sst(tok, items)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return float(model(input_ids=inp, attention_mask=am, labels=lb).loss.item())

def depth_subj(model, tok, B_train, seed):  # phase-2, whole net, identical all arms
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_DEPTH); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_DEPTH):
        inp, lb, am = b_pair(tok, [B_train[k] for k in torch.randperm(len(B_train), generator=g)[:BATCH].tolist()])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()

def poke_occupy(model, tok, trA, names, U, seed):  # DESIGN 1: train task-A keeping ONLY target-subspace grad component
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_POKE); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_POKE):
        inp, lb, am = b_sst(tok, [trA[k] for k in torch.randperm(len(trA), generator=g)[:BATCH].tolist()])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            for n in names:
                gr = dict(model.named_parameters())[n].grad
                if gr is not None: gr.copy_((gr @ U[n]) @ U[n].T)   # KEEP only target-subspace component
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()

@torch.no_grad()
def poke_perturb(model, tok, trA, names, U, base_loss, Rmats):  # DESIGN 2: inject dW into target subspace, match dloss
    # dW(alpha) = alpha * (R @ U) @ U^T per carrier; binary-search alpha so task-A loss rises by ~TARGET_DLOSS
    W0 = {n: dict(model.named_parameters())[n].detach().clone() for n in names}
    pm = dict(model.named_parameters())
    PC = {n: (Rmats[n] @ U[n]) @ U[n].T for n in names}       # (d_out,d_in) supported on target input-subspace
    def apply(alpha):
        for n in names: pm[n].copy_(W0[n] + alpha * PC[n])
    def dloss(alpha): apply(alpha); return sst_loss(model, tok, trA[:64]) - base_loss
    # scale-robust bracketing: expand hi (cap 50 to avoid model-wrecking NaN) until dl crosses target
    lo, hi = 0.0, 1e-4
    while dloss(hi) < TARGET_DLOSS and hi < 50.0: lo = hi; hi *= 4.0
    capped = hi >= 50.0 and dloss(hi) < TARGET_DLOSS         # flat subspace: can't reach target without wrecking model
    for _ in range(24):                                       # bisect the bracket [lo, hi]
        mid = 0.5 * (lo + hi); dl = dloss(mid)
        if dl < TARGET_DLOSS: lo = mid
        else: hi = mid
    mid = 0.5 * (lo + hi); dl = dloss(mid)
    return float(mid if not capped else 50.0), float(dl)

def subspaces(VA, names, seed):
    """fragile=top-R input-eigvecs of FA; complement=bottom-R; random=R-dim random orthonormal (per seed)."""
    out = {"fragile": {}, "random": {}, "complement": {}}
    gcpu = torch.Generator().manual_seed(1234 + seed)
    for n in names:
        d = VA[n].shape[0]
        out["fragile"][n] = VA[n][:, -R:].float().contiguous()
        out["complement"][n] = VA[n][:, :R].float().contiguous()
        Q, _ = torch.linalg.qr(torch.randn(d, R, generator=gcpu))
        out["random"][n] = Q.to(dev).float().contiguous()
    return out

def main():
    t0 = time.time(); print("[POKE] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    sd0 = torch.load(CKPT); model.load_state_dict(sd0); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trA, vaA = sst_data(); B_train, B_eval, B_tmpl, B_comps = load_taskB("medium_subj")
    params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    pre_sst = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])
    base_loss = sst_loss(model, tok, trA[:64])
    print(f"[POKE] pre SST acc={pre_sst:.3f} base_loss={base_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
    # FA @ w_A -> eigvecs (fragile/complement bases). Recompute (align-style guard implicit via downstream reproduction).
    FA = fisher(model, params, names, tok, trA); VA = {n: torch.linalg.eigh(FA[n])[1] for n in names}
    lam_top = float(np.mean([torch.linalg.eigvalsh(FA[n])[-R:].mean().item() for n in names]))
    lam_bot = float(np.mean([torch.linalg.eigvalsh(FA[n])[:R].mean().item() for n in names]))
    print(f"[POKE] FA done. mean lam top-{R}={lam_top:.3e} bot-{R}={lam_bot:.3e} (ratio {lam_top/max(lam_bot,1e-30):.1e}) ({time.time()-t0:.0f}s)", flush=True)
    Rmats = {n: torch.randn(params[n].shape, generator=torch.Generator().manual_seed(7)).to(dev) for n in names}  # fixed random for perturb

    ARMS = ["A", "B", "C", "D"]; SUB = {"B": "fragile", "C": "random", "D": "complement"}
    R1 = {a: {"sstA": [], "taskB": [], "postpoke_sst": []} for a in ARMS}            # design 1 occupy
    R2 = {a: {"sstA": [], "taskB": [], "postpoke_sst": [], "alpha": [], "dloss": []} for a in ARMS}  # design 2 perturb
    for sd in SEEDS:
        US = subspaces(VA, names, sd)
        # ---- Arm A (shared floor: no poke) ----
        model.load_state_dict(ckpt_sd); depth_subj(model, tok, B_train, sd)
        a_sst = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]); a_b = clf_acc(model, tok, B_eval, B_tmpl, B_comps)
        for Rx in (R1, R2):
            Rx["A"]["sstA"].append(a_sst); Rx["A"]["taskB"].append(a_b); Rx["A"]["postpoke_sst"].append(pre_sst)
        # ---- DESIGN 1: occupy ----
        for arm in ("B", "C", "D"):
            model.load_state_dict(ckpt_sd); poke_occupy(model, tok, trA, names, US[SUB[arm]], sd)
            R1[arm]["postpoke_sst"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]))
            depth_subj(model, tok, B_train, sd)
            R1[arm]["sstA"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])); R1[arm]["taskB"].append(clf_acc(model, tok, B_eval, B_tmpl, B_comps))
        # ---- DESIGN 2: damage-matched perturb ----
        for arm in ("B", "C", "D"):
            model.load_state_dict(ckpt_sd); alpha, dl = poke_perturb(model, tok, trA, names, US[SUB[arm]], base_loss, Rmats)
            R2[arm]["postpoke_sst"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])); R2[arm]["alpha"].append(alpha); R2[arm]["dloss"].append(dl)
            depth_subj(model, tok, B_train, sd)
            R2[arm]["sstA"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])); R2[arm]["taskB"].append(clf_acc(model, tok, B_eval, B_tmpl, B_comps))
        print(f"[POKE] seed {sd}: D1 sstA A/B/C/D={R1['A']['sstA'][-1]:.3f}/{R1['B']['sstA'][-1]:.3f}/{R1['C']['sstA'][-1]:.3f}/{R1['D']['sstA'][-1]:.3f}"
              f" | D2={R2['A']['sstA'][-1]:.3f}/{R2['B']['sstA'][-1]:.3f}/{R2['C']['sstA'][-1]:.3f}/{R2['D']['sstA'][-1]:.3f}"
              f" | D2 alpha B/C/D={R2['B']['alpha'][-1]:.2e}/{R2['C']['alpha'][-1]:.2e}/{R2['D']['alpha'][-1]:.2e} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"D1_{a}_{m}": np.array(R1[a][m]) for a in ARMS for m in R1[a]},
             **{f"D2_{a}_{m}": np.array(R2[a][m]) for a in ARMS for m in R2[a]}, pre_sst=pre_sst, base_loss=base_loss)
    def agg(Rx): return {a: {m: [float(np.mean(Rx[a][m])), float(np.std(Rx[a][m]) / max(len(Rx[a][m]), 1) ** 0.5)] for m in Rx[a] if Rx[a][m]} for a in ARMS}
    rep = {"heldout": "subj", "R": R, "pre_sst": pre_sst, "lam_top": lam_top, "lam_bot": lam_bot,
           "config": {"STEPS_POKE": STEPS_POKE, "STEPS_DEPTH": STEPS_DEPTH, "TARGET_DLOSS": TARGET_DLOSS, "seeds": len(SEEDS)},
           "design1_occupy": agg(R1), "design2_perturb": agg(R2)}
    try:
        def verdict(Rx, tag):
            a = np.array(Rx["A"]["sstA"]); b = np.array(Rx["B"]["sstA"]); c = np.array(Rx["C"]["sstA"]); d = np.array(Rx["D"]["sstA"])
            def pr(x, y): m = float((x - y).mean()); se = float((x - y).std() / len(x) ** 0.5); return [m, se, abs(m) > 2 * se]
            BA, BC, BD = pr(b, a), pr(b, c), pr(b, d)
            pp = float(np.mean(Rx["B"]["postpoke_sst"])); nd = pp >= pre_sst - 0.05
            starved = np.array(Rx["B"]["taskB"]).mean() < np.array(Rx["A"]["taskB"]).mean() - 0.05
            if not nd: v = f"[{tag}] VOID: poke-fragile DESTROYED task-A (postpoke {pp:.3f} < pre {pre_sst:.3f}-0.05) -> damage not poke."
            elif not BA[2]: v = f"[{tag}] INERT: B~A (B-A={BA[0]:+.3f}, 2SE {2*BA[1]:.3f}) -> poking fragile doesn't change forgetting."
            elif BA[0] > 0: v = f"[{tag}] RESOURCE/DEFLECT: poke-fragile RETAINS MORE (B-A={BA[0]:+.3f}>2SE){' FRAGILE-SPECIFIC' if BC[2] else ' but ~C (generic)'}{' starved!' if starved else ''}."
            else: v = f"[{tag}] SINK/PRIMED: poke-fragile RETAINS LESS (B-A={BA[0]:+.3f}<0, >2SE){' FRAGILE-SPECIFIC' if BC[2] else ' but ~C (generic)'}."
            return v, {"B_minus_A": BA, "B_minus_C": BC, "B_minus_D": BD, "postpoke_B": pp, "nondestructive": bool(nd), "starved": bool(starved),
                       "retention": {"A": float(a.mean()), "B": float(b.mean()), "C": float(c.mean()), "D": float(d.mean())}}
        v1, p1 = verdict(R1, "D1 occupy"); v2, p2 = verdict(R2, "D2 perturb")
        rep["verdict_design1"], rep["paired_design1"] = v1, p1
        rep["verdict_design2"], rep["paired_design2"] = v2, p2
        rep["alpha_design2"] = {arm: float(np.mean(R2[arm]["alpha"])) for arm in ("B", "C", "D")}
        rep["dloss_design2"] = {arm: float(np.mean(R2[arm]["dloss"])) for arm in ("B", "C", "D")}
        print(f"\n[POKE] === VERDICTS (held-out=subj, pre {pre_sst:.3f}) ===", flush=True)
        print("  " + v1, flush=True); print(f"     retention A/B/C/D = {p1['retention']['A']:.3f}/{p1['retention']['B']:.3f}/{p1['retention']['C']:.3f}/{p1['retention']['D']:.3f}  postpoke-B {p1['postpoke_B']:.3f}", flush=True)
        print("  " + v2, flush=True); print(f"     retention A/B/C/D = {p2['retention']['A']:.3f}/{p2['retention']['B']:.3f}/{p2['retention']['C']:.3f}/{p2['retention']['D']:.3f}  postpoke-B {p2['postpoke_B']:.3f}", flush=True)
        print(f"     D2 alpha B/C/D = {rep['alpha_design2']['B']:.2e}/{rep['alpha_design2']['C']:.2e}/{rep['alpha_design2']['D']:.2e}  (fragile<<complement confirms ~10^6x curvature gap)", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[POKE] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[POKE] saved poke_test.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
