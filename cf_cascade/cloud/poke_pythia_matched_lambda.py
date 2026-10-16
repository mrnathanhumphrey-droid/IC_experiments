"""POKE DISAMBIGUATOR — Pythia D1-occupy at OPT's MATCHED forget severity. Architecture vs regime.
OPT poke (run 2) showed a fragile-specific SINK in D1-occupy (B-C -0.125 SIG, B below random/complement) that Pythia's
NATIVE-regime poke did NOT (Pythia B-C -0.068 ns). BUT OPT was measured at a gentler regime (forgot 0.121 vs Pythia native
0.214) with no clean window -> the OPT sink is regime-confounded. THIS run re-measures Pythia D1 at OPT's forget SEVERITY:
 - Pythia B-C STAYS ns at matched forget -> OPT's fragile-specific sink is ARCHITECTURAL (real cross-arch split, sink not deflect).
 - Pythia B-C ALSO goes SIG-negative -> the sink is a GENERIC GENTLE-FORGETTING regime effect -> OPT "split" DISSOLVES; poke is
   fully real-but-inert-as-a-handle on both archs, no architectural split. Clean kill either way.
SEVERITY MATCH (the lesson): same lr/steps gives different forgetting across archs, so DON'T blind-copy OPT's 2e-5/200 -> a
PRECHECK targets Pythia arm-A forget = 0.121 +/- 0.03 (tries 2e-5/200 first). If no setting matches, the run is flagged
regime-mismatched (not bankable), same discipline as the OPT precheck. Headline statistic = B-C (where the OPT non-null was).
D1 (occupy) ONLY — the disambiguator; D2/perturb dropped (not where the signal was). Held-out subj, arms A/B/C/D x 7 seeds.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/poke_pythia_matched.json"; NPZ = f"{G}/poke_pythia_matched.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7)); R = 64; STEPS_POKE = 200; LR_POKE = 5e-5; BATCH = 16; N_FISHER = 150; N_EVAL = 200
TARGET_FORGET = 0.121; TOL = 0.03                       # match OPT's arm-A forget severity
PRECHECK_GRID = [(2e-5, 200), (2e-5, 150), (3e-5, 150), (2e-5, 250), (3e-5, 200), (3e-5, 100), (5e-5, 100), (5e-5, 150)]
STEPS_DEPTH = 200; LR_DEPTH = 2e-5                      # DEFAULTS — overridden by precheck
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
    comps = sorted(set(e["completion"] for e in d["examples"])); eval_raw = [(e["text"], e["completion"]) for e in d["examples"][-N_EVAL:]]
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
def depth(model, tok, B_train, seed, steps, lr):
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        inp, lb, am = b_pair(tok, [B_train[k] for k in torch.randperm(len(B_train), generator=g)[:BATCH].tolist()])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
def poke_occupy(model, tok, trA, names, U, seed):
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_POKE); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_POKE):
        inp, lb, am = b_sst(tok, [trA[k] for k in torch.randperm(len(trA), generator=g)[:BATCH].tolist()])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            for n in names:
                gr = dict(model.named_parameters())[n].grad
                if gr is not None: gr.copy_((gr @ U[n]) @ U[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
def subspaces(VA, names, seed):
    out = {"fragile": {}, "random": {}, "complement": {}}; gcpu = torch.Generator().manual_seed(1234 + seed)
    for n in names:
        d = VA[n].shape[0]; out["fragile"][n] = VA[n][:, -R:].float().contiguous(); out["complement"][n] = VA[n][:, :R].float().contiguous()
        Q, _ = torch.linalg.qr(torch.randn(d, R, generator=gcpu)); out["random"][n] = Q.to(dev).float().contiguous()
    return out

def main():
    t0 = time.time(); print("[PYM] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trA, vaA = sst_data(); B_train, B_eval, B_tmpl, B_comps = load_taskB("medium_subj")
    params = dict(model.named_parameters()); names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    pre_sst = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])
    print(f"[PYM] pre SST acc={pre_sst:.3f} (target match OPT forget {TARGET_FORGET:.3f}) ({time.time()-t0:.0f}s)", flush=True)

    # ===== FORGET-MATCH PRECHECK: find lr/steps where Pythia arm-A forget ~ OPT's 0.121 (try 2e-5/200 first) =====
    global STEPS_DEPTH, LR_DEPTH
    pck = []
    for lr, st in PRECHECK_GRID:
        model.load_state_dict(ckpt_sd); depth(model, tok, B_train, 0, st, lr)
        ret = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]); forget = pre_sst - ret
        pck.append({"lr": lr, "steps": st, "ret": ret, "forget": forget})
        print(f"[PYM]   lr={lr:.0e} st={st}: ret={ret:.3f} forget={forget:.3f}{'  MATCH' if abs(forget-TARGET_FORGET)<=TOL else ''} ({time.time()-t0:.0f}s)", flush=True)
    choice = min(pck, key=lambda p: abs(p["forget"] - TARGET_FORGET)); matched = abs(choice["forget"] - TARGET_FORGET) <= TOL
    LR_DEPTH, STEPS_DEPTH = choice["lr"], choice["steps"]
    print(f"[PYM] PRECHECK -> lr={LR_DEPTH:.0e} steps={STEPS_DEPTH} forget={choice['forget']:.3f} (OPT 0.121) "
          f"{'MATCHED' if matched else 'NO MATCH -> regime-mismatch flagged'} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(ckpt_sd); params = dict(model.named_parameters())
    FA = fisher(model, params, names, tok, trA); VA = {n: torch.linalg.eigh(FA[n])[1] for n in names}; del FA
    print(f"[PYM] FA basis done ({time.time()-t0:.0f}s)", flush=True)

    ARMS = ["A", "B", "C", "D"]; SUB = {"B": "fragile", "C": "random", "D": "complement"}
    R1 = {a: {"sstA": [], "taskB": [], "postpoke_sst": []} for a in ARMS}
    for sd in SEEDS:
        US = subspaces(VA, names, sd)
        model.load_state_dict(ckpt_sd); depth(model, tok, B_train, sd, STEPS_DEPTH, LR_DEPTH)
        R1["A"]["sstA"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])); R1["A"]["taskB"].append(clf_acc(model, tok, B_eval, B_tmpl, B_comps)); R1["A"]["postpoke_sst"].append(pre_sst)
        for arm in ("B", "C", "D"):
            model.load_state_dict(ckpt_sd); poke_occupy(model, tok, trA, names, US[SUB[arm]], sd)
            R1[arm]["postpoke_sst"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"]))
            depth(model, tok, B_train, sd, STEPS_DEPTH, LR_DEPTH)
            R1[arm]["sstA"].append(clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])); R1[arm]["taskB"].append(clf_acc(model, tok, B_eval, B_tmpl, B_comps))
        print(f"[PYM] seed {sd}: A/B/C/D={R1['A']['sstA'][-1]:.3f}/{R1['B']['sstA'][-1]:.3f}/{R1['C']['sstA'][-1]:.3f}/{R1['D']['sstA'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(NPZ, **{f"D1_{a}_{m}": np.array(R1[a][m]) for a in ARMS for m in R1[a]}, pre_sst=pre_sst)
    def agg(): return {a: {m: [float(np.mean(R1[a][m])), float(np.std(R1[a][m]) / 7 ** 0.5)] for m in R1[a]} for a in ARMS}
    rep = {"model": MODEL, "heldout": "subj", "R": R, "pre_sst": pre_sst, "target_forget": TARGET_FORGET,
           "chosen_regime": {"lr": LR_DEPTH, "steps": STEPS_DEPTH, "forget": choice["forget"], "matched_OPT": bool(matched)},
           "precheck": pck, "design1_occupy": agg()}
    try:
        a, b, c, d = (np.array(R1[x]["sstA"]) for x in "ABCD")
        def pr(x, y): m = float((x - y).mean()); se = float((x - y).std() / 7 ** 0.5); return [m, se, abs(m) > 2 * se]
        BA, BC, BD = pr(b, a), pr(b, c), pr(b, d); pp = float(np.mean(R1["B"]["postpoke_sst"]))
        rep["paired"] = {"B_minus_A": BA, "B_minus_C": BC, "B_minus_D": BD, "armA_forget": float(pre_sst - a.mean()),
                         "retention": {x: float(np.array(R1[x]["sstA"]).mean()) for x in "ABCD"}}
        # headline disambiguation on B-C (where OPT's non-null was: OPT B-C=-0.125 SIG)
        if not matched:
            v = f"REGIME-MISMATCH (not bankable): closest Pythia forget={choice['forget']:.3f} vs OPT 0.121 (>{TOL}). Comparison still confounded."
        elif BC[2] and BC[0] < 0:
            v = (f"REGIME EFFECT — OPT 'split' DISSOLVES: Pythia B-C={BC[0]:+.3f} SIG-negative at matched forget {choice['forget']:.3f} "
                 f"too (OPT was -0.125). The fragile-specific sink is a GENERIC gentle-forgetting effect, not architectural. Poke is "
                 f"real-but-inert-as-a-handle on BOTH archs; NO architectural split. Clean kill.")
        else:
            v = (f"ARCHITECTURAL — real cross-arch split (sink direction): Pythia B-C={BC[0]:+.3f} ns at matched forget {choice['forget']:.3f} "
                 f"(OPT was -0.125 SIG). OPT's fragile-specific SINK is architectural — but it's the OPPOSITE of the pre-registered "
                 f"deflect; antagonism controls deposit-survival WITHOUT giving poke-deflection (the two dissociate).")
        rep["verdict"] = v
        print(f"\n[PYM] === DISAMBIGUATOR VERDICT ===\n  {v}", flush=True)
        print(f"  retention A/B/C/D = {a.mean():.3f}/{b.mean():.3f}/{c.mean():.3f}/{d.mean():.3f} | armA forget {pre_sst-a.mean():.3f} (OPT 0.121, matched={matched})", flush=True)
        print(f"  B-A={BA[0]:+.3f}{'*' if BA[2] else ''} | B-C={BC[0]:+.3f}{'*' if BC[2] else ''} | B-D={BD[0]:+.3f}{'*' if BD[2] else ''} | postpoke-B {pp:.3f} (pre {pre_sst:.3f})", flush=True)
        print(f"  (Pythia NATIVE-regime was: B-A -0.001, B-C -0.068 ns, forget 0.214 | OPT: B-A -0.078 ns, B-C -0.125 SIG, forget 0.121)", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[PYM] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[PYM] saved poke_pythia_matched.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
