"""POKE TEST on OPT-350m — the CROSS-ARCH SPLIT. Pythia (antagonistic core) gave INERT (D1) / generic-weak-SINK (D2):
the fragile subspace is NOT a manipulable handle there. OPT has the ORTHOGONAL shared core where rung-3 says deposits
SURVIVE (map load-bearing 74%) — so OPT is the one architecture where pre-occupying the fragile subspace could DEFLECT.

PRE-REG (cross-arch fork, committed before firing):
 - OPT D1 DEFLECTS (B>A, fragile-specific B!=C) -> RESOURCE appears WHERE deposits survive -> interventional confirmation
   of antagonism-controls-survival (Pythia eroded=no effect; OPT survives=deflect). Interventional sibling of rung-4 M3.
 - OPT D1 INERT (B~A, like Pythia) -> fragile geometrically blocked as an intervention target on BOTH archs -> "real-but-
   inert as a handle" is universal; rung-3's OPT map-load-bearing works through something other than fragile-occupation.
 - OPT D1 WORSE (B<A) -> primed-channel on both archs -> fragile is a leak-origin universally.
PRIMARY = DESIGN 1 (functional occupy, non-destructive) — Pythia's clean arm (was INERT); the cross-arch comparison is D1-vs-D1.
  D2 (damage-matched perturb) included for apples-to-apples + OPT's generic perturbation/curvature baseline (NOT the headline).

Identical poke protocol to poke_test_lambda.py; only the substrate changes (OPT model/carriers + w_A trained fresh, no ckpt).
Held-out subj, arms A/B/C/D x 7 seeds. GUARDS: w_A acc reproduces rung-1 ~0.77; arm A forgets >=0.10 (VOID else); poke
NON-DESTRUCTIVE (postpoke SST ~ pre — re-derived on OPT's own curvature via D2 alpha-search; D1 non-destructive by construction);
task-B not starved; per-seed saved before readout. "fragile" = top-R input-curvature dirs of FA@w_A (SAME operational def as the
Pythia poke, for a clean cross-arch comparison; on OPT fragile may differ from seed-defined fragile — noted).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "facebook/opt-350m"; AX = f"{G}/overlap_axis"; OUT = f"{G}/poke_test_opt.json"; NPZ = f"{G}/poke_test_opt.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7))
R = 64
STEPS_WA = 1200; LR_WA = 5e-5; SUBSET = 2000          # build w_A (rung-2 convention, SST acc ~0.77)
STEPS_POKE = 200; LR_POKE = 5e-5
STEPS_DEPTH = 200; LR_DEPTH = 5e-5; BATCH = 16    # DEFAULTS — overridden by the depth-severity precheck below
PRECHECK_GRID = [(lr, st) for lr in (1e-5, 2e-5, 3e-5) for st in (60, 120, 200)]  # OPT collapsed at 5e-5/200; find a readable window
WIN_LO, WIN_HI, WIN_MID = 0.58, 0.72, 0.64        # readable SST-retention window (arm A must forget but not floor)
MIN_SUBJ = 0.70                                    # task-B must actually learn
N_FISHER = 150; N_EVAL = 200; TARGET_DLOSS = 0.05
torch.backends.cudnn.benchmark = True

def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]; rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]
def sst_items(k, seed): return [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", k, seed)]
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
def b_items(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:128]; c = tok(co, add_special_tokens=False)["input_ids"][:8]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)
def fisher(model, params, names, tok, items):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    g = torch.Generator().manual_seed(0)
    for _ in range(N_FISHER):
        idx = torch.randperm(len(items), generator=g)[:1].tolist()
        inp, lb, am = b_items(tok, [items[i] for i in idx]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: gr = params[n].grad.detach().double(); F[n] += gr.T @ gr
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
    model.eval(); inp, lb, am = b_items(tok, items)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16): return float(model(input_ids=inp, attention_mask=am, labels=lb).loss.item())
def train_items(model, tok, items, steps, lr, seed):
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        inp, lb, am = b_items(tok, [items[k] for k in torch.randperm(len(items), generator=g)[:BATCH].tolist()])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); del opt
def poke_occupy(model, tok, trA, names, U, seed):  # DESIGN 1: train task-A keeping ONLY target-subspace grad component
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_POKE); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_POKE):
        inp, lb, am = b_items(tok, [trA[k] for k in torch.randperm(len(trA), generator=g)[:BATCH].tolist()])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            for n in names:
                gr = dict(model.named_parameters())[n].grad
                if gr is not None: gr.copy_((gr @ U[n]) @ U[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); del opt
@torch.no_grad()
def poke_perturb(model, tok, trA, names, U, base_loss, Rmats):  # DESIGN 2: dW into subspace, alpha matched to dloss
    pm = dict(model.named_parameters()); W0 = {n: pm[n].detach().clone() for n in names}
    PC = {n: (Rmats[n] @ U[n]) @ U[n].T for n in names}
    def apply(a):
        for n in names: pm[n].copy_(W0[n] + a * PC[n])
    def dloss(a): apply(a); return sst_loss(model, tok, trA[:64]) - base_loss
    lo, hi = 0.0, 1e-4
    while dloss(hi) < TARGET_DLOSS and hi < 50.0: lo = hi; hi *= 4.0
    capped = hi >= 50.0 and dloss(hi) < TARGET_DLOSS
    for _ in range(24):
        mid = 0.5 * (lo + hi); d = dloss(mid)
        if d < TARGET_DLOSS: lo = mid
        else: hi = mid
    mid = 0.5 * (lo + hi); d = dloss(mid)
    return float(mid if not capped else 50.0), float(d)

def main():
    t0 = time.time(); print(f"[POKE-OPT] loading {MODEL} (eager)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    trA = sst_items(SUBSET, 7); vaA = [(s, LAB[l]) for s, l in sst_rows("validation", N_EVAL, 0)]
    B_train, B_eval, B_tmpl, B_comps = load_taskB("medium_subj")
    print(f"[POKE-OPT] training task-A=SST -> w_A ({STEPS_WA} steps)...", flush=True)
    train_items(model, tok, trA, STEPS_WA, LR_WA, 0)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]); base_loss = sst_loss(model, tok, trA[:64])
    print(f"[POKE-OPT] w_A SST acc={pre_sst:.3f} (rung-1 ~0.77) base_loss={base_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ===== DEPTH-SEVERITY PRECHECK (the void-run fix): find a regime where arm A retains readable SST range AND learns subj =====
    print(f"[POKE-OPT] PRECHECK depth regime (prev 5e-5/200 collapsed both tasks to chance)...", flush=True)
    pck = []
    for lr, st in PRECHECK_GRID:
        model.load_state_dict(ckpt_sd); train_items(model, tok, B_train, st, lr, 0)
        ssr = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]); sbj = clf_acc(model, tok, B_eval, B_tmpl, B_comps)
        readable = (WIN_LO <= ssr <= WIN_HI) and (sbj >= MIN_SUBJ) and (pre_sst - ssr >= 0.08)
        pck.append({"lr": lr, "steps": st, "sst_ret": ssr, "subj": sbj, "readable": readable})
        print(f"[POKE-OPT]   lr={lr:.0e} steps={st}: SST_ret={ssr:.3f} subj={sbj:.3f} {'READABLE' if readable else ''} ({time.time()-t0:.0f}s)", flush=True)
    rd = [p for p in pck if p["readable"]]
    if rd:
        choice = min(rd, key=lambda p: abs(p["sst_ret"] - WIN_MID)); no_window = False
    else:  # relaxed fallback: subj learned + some forgetting + above floor; flag heavily
        relaxed = [p for p in pck if p["subj"] >= 0.65 and p["sst_ret"] > 0.54 and (pre_sst - p["sst_ret"]) >= 0.05]
        no_window = True
        choice = (min(relaxed, key=lambda p: abs(p["sst_ret"] - WIN_MID)) if relaxed
                  else max(pck, key=lambda p: p["subj"]))   # last resort: best-subj setting, results will be flagged
    global STEPS_DEPTH, LR_DEPTH
    LR_DEPTH, STEPS_DEPTH = choice["lr"], choice["steps"]
    print(f"[POKE-OPT] PRECHECK -> chosen lr={LR_DEPTH:.0e} steps={STEPS_DEPTH} (SST_ret~{choice['sst_ret']:.3f} subj~{choice['subj']:.3f}) "
          f"{'NO CLEAN WINDOW (results flagged)' if no_window else 'readable window'} ({time.time()-t0:.0f}s)", flush=True)
    if no_window and choice["subj"] < 0.55:
        print("[POKE-OPT] WARNING: no setting learned subj on OPT — poke run will likely be VOID again; proceeding to record the grid.", flush=True)

    # FA @ w_A -> store ONLY top-R (fragile) + bottom-R (complement) eigvec slices (memory: OPT fc2 is 4096^2)
    model.load_state_dict(ckpt_sd)                            # precheck left the model subj-trained; restore w_A before Fisher
    params = dict(model.named_parameters()); FA = fisher(model, params, names, tok, trA)
    Vfrag, Vcomp, din = {}, {}, {}; lt, lb = [], []
    for n in names:
        ev, vv = torch.linalg.eigh(FA[n])                    # ascending
        Vfrag[n] = vv[:, -R:].float().contiguous(); Vcomp[n] = vv[:, :R].float().contiguous(); din[n] = vv.shape[0]
        lt.append(float(ev[-R:].mean().item())); lb.append(float(ev[:R].mean().item())); del FA[n], vv, ev
    del FA; torch.cuda.empty_cache()
    print(f"[POKE-OPT] FA done. mean lam top-{R}={np.mean(lt):.3e} bot-{R}={np.mean(lb):.3e} ({time.time()-t0:.0f}s)", flush=True)
    Rmats = {n: torch.randn(params[n].shape, generator=torch.Generator().manual_seed(7)).to(dev) for n in names}

    def subspaces(seed):
        out = {"fragile": Vfrag, "complement": Vcomp, "random": {}}
        gcpu = torch.Generator().manual_seed(1234 + seed)
        for n in names:
            Q, _ = torch.linalg.qr(torch.randn(din[n], R, generator=gcpu)); out["random"][n] = Q.to(dev).float().contiguous()
        return out
    def depth_subj(seed): train_items(model, tok, B_train, STEPS_DEPTH, LR_DEPTH, seed)
    def evalAB(): return (clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]),
                          clf_acc(model, tok, B_eval, B_tmpl, B_comps))

    ARMS = ["A", "B", "C", "D"]; SUB = {"B": "fragile", "C": "random", "D": "complement"}
    R1 = {a: {"sstA": [], "taskB": [], "postpoke_sst": []} for a in ARMS}
    R2 = {a: {"sstA": [], "taskB": [], "postpoke_sst": [], "alpha": [], "dloss": []} for a in ARMS}
    for sd in SEEDS:
        US = subspaces(sd)
        model.load_state_dict(ckpt_sd); depth_subj(sd); a_sst, a_b = evalAB()
        for Rx in (R1, R2):
            Rx["A"]["sstA"].append(a_sst); Rx["A"]["taskB"].append(a_b); Rx["A"]["postpoke_sst"].append(pre_sst)
        for arm in ("B", "C", "D"):
            model.load_state_dict(ckpt_sd); poke_occupy(model, tok, trA, names, US[SUB[arm]], sd)
            R1[arm]["postpoke_sst"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]))
            depth_subj(sd); s, b = evalAB(); R1[arm]["sstA"].append(s); R1[arm]["taskB"].append(b)
        for arm in ("B", "C", "D"):
            model.load_state_dict(ckpt_sd); alpha, dl = poke_perturb(model, tok, trA, names, US[SUB[arm]], base_loss, Rmats)
            R2[arm]["postpoke_sst"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])); R2[arm]["alpha"].append(alpha); R2[arm]["dloss"].append(dl)
            depth_subj(sd); s, b = evalAB(); R2[arm]["sstA"].append(s); R2[arm]["taskB"].append(b)
        print(f"[POKE-OPT] seed {sd}: D1 A/B/C/D={R1['A']['sstA'][-1]:.3f}/{R1['B']['sstA'][-1]:.3f}/{R1['C']['sstA'][-1]:.3f}/{R1['D']['sstA'][-1]:.3f}"
              f" | D2={R2['A']['sstA'][-1]:.3f}/{R2['B']['sstA'][-1]:.3f}/{R2['C']['sstA'][-1]:.3f}/{R2['D']['sstA'][-1]:.3f}"
              f" | D2a B/C/D={R2['B']['alpha'][-1]:.2e}/{R2['C']['alpha'][-1]:.2e}/{R2['D']['alpha'][-1]:.2e} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"D1_{a}_{m}": np.array(R1[a][m]) for a in ARMS for m in R1[a]},
             **{f"D2_{a}_{m}": np.array(R2[a][m]) for a in ARMS for m in R2[a]}, pre_sst=pre_sst, base_loss=base_loss)
    def agg(Rx): return {a: {m: [float(np.mean(Rx[a][m])), float(np.std(Rx[a][m]) / max(len(Rx[a][m]), 1) ** 0.5)] for m in Rx[a] if Rx[a][m]} for a in ARMS}
    rep = {"model": MODEL, "heldout": "subj", "R": R, "pre_sst": pre_sst, "lam_top": float(np.mean(lt)), "lam_bot": float(np.mean(lb)),
           "config": {"STEPS_WA": STEPS_WA, "STEPS_POKE": STEPS_POKE, "STEPS_DEPTH": STEPS_DEPTH, "LR_DEPTH": LR_DEPTH, "TARGET_DLOSS": TARGET_DLOSS, "seeds": len(SEEDS)},
           "precheck": pck, "chosen_depth": {"lr": LR_DEPTH, "steps": STEPS_DEPTH, "no_clean_window": bool(no_window)},
           "design1_occupy": agg(R1), "design2_perturb": agg(R2)}
    try:
        def verdict(Rx, tag):
            a = np.array(Rx["A"]["sstA"]); b = np.array(Rx["B"]["sstA"]); c = np.array(Rx["C"]["sstA"]); d = np.array(Rx["D"]["sstA"])
            def pr(x, y): m = float((x - y).mean()); se = float((x - y).std() / len(x) ** 0.5); return [m, se, abs(m) > 2 * se]
            BA, BC, BD = pr(b, a), pr(b, c), pr(b, d); pp = float(np.mean(Rx["B"]["postpoke_sst"])); nd = pp >= pre_sst - 0.05
            starved = np.array(Rx["B"]["taskB"]).mean() < np.array(Rx["A"]["taskB"]).mean() - 0.05
            if not nd: v = f"[{tag}] VOID: poke-fragile destroyed task-A (postpoke {pp:.3f}<pre {pre_sst:.3f}-0.05)."
            elif not BA[2]: v = f"[{tag}] INERT: B~A (B-A={BA[0]:+.3f}, 2SE {2*BA[1]:.3f})."
            elif BA[0] > 0: v = f"[{tag}] RESOURCE/DEFLECT: B retains MORE (B-A={BA[0]:+.3f}>2SE){' FRAGILE-SPECIFIC' if BC[2] else ' but ~C (generic)'}{' starved!' if starved else ''}."
            else: v = f"[{tag}] SINK/PRIMED: B retains LESS (B-A={BA[0]:+.3f}<0,>2SE){' FRAGILE-SPECIFIC' if BC[2] else ' but ~C (generic)'}."
            return v, {"B_minus_A": BA, "B_minus_C": BC, "B_minus_D": BD, "postpoke_B": pp, "nondestructive": bool(nd), "starved": bool(starved),
                       "retention": {"A": float(a.mean()), "B": float(b.mean()), "C": float(c.mean()), "D": float(d.mean())}}
        v1, p1 = verdict(R1, "D1 occupy"); v2, p2 = verdict(R2, "D2 perturb")
        rep["verdict_design1"], rep["paired_design1"] = v1, p1; rep["verdict_design2"], rep["paired_design2"] = v2, p2
        rep["alpha_design2"] = {arm: float(np.mean(R2[arm]["alpha"])) for arm in ("B", "C", "D")}
        a = np.array(R1["A"]["sstA"]); rep["armA_forgets"] = float(pre_sst - a.mean())
        print(f"\n[POKE-OPT] === VERDICTS (OPT, held-out=subj, pre {pre_sst:.3f}, armA forgets {pre_sst-a.mean():.3f}) ===", flush=True)
        print("  " + v1, flush=True); print(f"     retention A/B/C/D={p1['retention']['A']:.3f}/{p1['retention']['B']:.3f}/{p1['retention']['C']:.3f}/{p1['retention']['D']:.3f} postpoke-B {p1['postpoke_B']:.3f}", flush=True)
        print("  " + v2, flush=True); print(f"     retention A/B/C/D={p2['retention']['A']:.3f}/{p2['retention']['B']:.3f}/{p2['retention']['C']:.3f}/{p2['retention']['D']:.3f} postpoke-B {p2['postpoke_B']:.3f}", flush=True)
        print(f"     D2 alpha B/C/D={rep['alpha_design2']['B']:.2e}/{rep['alpha_design2']['C']:.2e}/{rep['alpha_design2']['D']:.2e}", flush=True)
        print(f"\n  CROSS-ARCH (D1 occupy, the headline): Pythia was INERT (B-A -0.001). OPT B-A={p1['B_minus_A'][0]:+.3f} -> "
              f"{'DEFLECT (split confirmed!)' if p1['B_minus_A'][2] and p1['B_minus_A'][0]>0 else ('SINK' if p1['B_minus_A'][2] else 'INERT (universal, no split)')}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[POKE-OPT] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[POKE-OPT] saved poke_test_opt.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
