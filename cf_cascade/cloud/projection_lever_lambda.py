"""PROJECTION LEVER — the deployable TEST of the hybrid recipe (Wilson-blessed, pre-committed criteria).
Recipe claim: task-A's important subspace is mostly FREE LANE (61-88% task-B-irrelevant) + a small antagonistic
core. So projecting task-B's gradient OFF task-A's top-Fisher input-directions (GPM-style) should PROTECT task-A
at LOW task-B cost — because most of what's projected away is free lane task-B never needed.

Per carrier weight W (d_out,d_in), U = top-r input-eigvecs of FA=E_A[g^T g] (d_in,r):
  during task-B training, after backward:  g <- g - (g @ U) @ U^T   (remove task-A input-dir components)

MEASURE (PERFORMANCE not norm), projected vs unprojected baseline:
  (a) task-A RETENTION  — A_proj vs A_base (the lever's point)
  (b) task-B LEARNING   — B_proj vs B_base (the lever's cost; does projection STARVE task-B?)
PRE-COMMITTED SUCCESS(r): A_proj - A_base >= 0.05 (retention gained) AND B_base - B_proj <= 0.05 (B not starved).
CORROBORATION: don't bank on A-retention alone — A-retained-while-B-craters is just "blocked task-B", NOT a win.
Verdict is on the PAIR. r-sweep {32,64,128} traces the retention-vs-B-cost frontier.
RIDER: FB-at-w_A control (overlap top-k(FA) vs top-k(FB@w_A)) vs the banked FB@w_B (0.184 AG qkv) — isolates
intrinsic task-direction difference from trajectory movement.
Task-A=SST-2 (held-out val), task-B=AG-News (held-out split). Carriers = qkv+mlp_h4h+mlp_4hh (where lane measured).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"; OUT = f"{G}/projection_lever.json"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
RSWEEP = [32, 64, 128]
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_FISHER = 150; N_EVAL = 200; KTOP = 64
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst_data():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    tr = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)]
    trA = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:max(N_FISHER, 200)]]
    va = [(r["sentence"].strip(), LAB[int(r["label"])]) for r in ds["validation"] if r["label"] in (0, 1)][:N_EVAL]
    return trA, va


def load_taskB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    ex = [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]
    comps = sorted(set(e["completion"] for e in d["examples"]))
    eval_raw = [(e["text"], e["completion"]) for e in d["examples"][-N_EVAL:]]
    return ex[:-N_EVAL], eval_raw, t, comps


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


@torch.no_grad()
def clf_acc(model, tok, raw_items, template, completions):
    """raw_items: list of (text, true_completion). argmax sum-logprob over candidate completions."""
    model.eval(); cand_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in completions]
    correct = 0
    for text, true in raw_items:
        p_ids = tok(template.format(text=text), add_special_tokens=False)["input_ids"]
        best, best_lp = None, -1e30
        for c, c_ids in zip(completions, cand_ids):
            inp = torch.tensor([p_ids + c_ids]).to(dev)
            logits = model(input_ids=inp).logits[0]
            lp = sum(torch.log_softmax(logits[len(p_ids) + j - 1], -1)[tid].item() for j, tid in enumerate(c_ids))
            if lp > best_lp: best_lp, best = lp, c
        correct += int(best == true)
    return correct / len(raw_items)


def train_taskB(model, tok, B, U=None, carrier_names=None):
    """U: dict name->(d_in,r) float32 projection basis; if given, project carrier grads off it each step."""
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        for i in range(0, len(B) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            if U is not None:
                with torch.no_grad():
                    for n in carrier_names:
                        gr = dict(model.named_parameters())[n].grad
                        if gr is not None: gr.sub_((gr @ U[n]) @ U[n].T)   # remove task-A input-dir components
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); return float(np.mean(run)) if run else float("nan"), step


def main():
    t0 = time.time(); print("[PL] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA, vaA = sst_data(); B_train, B_eval, B_tmpl, B_comps = load_taskB("low_agnews")
    params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    # ---- FA at w_A + projection bases for each r ----
    FA = fisher(model, params, names, tok, trA, False)
    VA = {n: torch.linalg.eigh(FA[n])[1] for n in names}                       # ascending eigvecs
    Ur = {r: {n: VA[n][:, -r:].float().contiguous() for n in names} for r in RSWEEP}
    A0 = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])
    print(f"[PL] FA done; task-A ceiling A0={A0:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- RIDER: FB at w_A (intrinsic vs trajectory) ----
    FBwA = fisher(model, params, names, tok, B_train, True)
    fbwa = {}
    for c in CARRIERS:
        ovs, chs = [], []
        for n in [x for x in names if c in x]:
            _, VB = torch.linalg.eigh(FBwA[n]); PA = VA[n][:, -KTOP:]; PB = VB[:, -KTOP:]
            ovs.append(float((PA.T @ PB).pow(2).sum().item() / KTOP)); chs.append(KTOP / FA[n].shape[0])
        fbwa[c] = {"overlap_FB_at_wA": float(np.mean(ovs)), "chance": float(np.mean(chs))}
    print(f"[PL] FB@w_A rider: " + "  ".join(f"{c.split('.')[-1]}={fbwa[c]['overlap_FB_at_wA']:.3f}" for c in CARRIERS) + f"  (vs FB@w_B banked qkv=0.184) ({time.time()-t0:.0f}s)", flush=True)

    def runB(U):
        model.load_state_dict(torch.load(CKPT))                                 # reset to w_A
        loss, steps = train_taskB(model, tok, B_train, U=U, carrier_names=names if U else None)
        A = clf_acc(model, tok, vaA, "{text}\nSentiment:", [" negative", " positive"])
        Bb = clf_acc(model, tok, B_eval, B_tmpl, B_comps)
        return {"A": A, "B": Bb, "trainloss": loss, "steps": steps}

    # ---- baseline (unprojected) ----
    base = runB(None)
    print(f"[PL] BASELINE: A={base['A']:.3f} (from A0={A0:.3f}, forgot {A0-base['A']:+.3f})  B={base['B']:.3f}  loss={base['trainloss']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- projected r-sweep ----
    proj = {}
    for r in RSWEEP:
        res = runB(Ur[r])
        Again = res["A"] - base["A"]; Bcost = base["B"] - res["B"]
        ok = (Again >= 0.05) and (Bcost <= 0.05)
        res["A_retention_gain"] = Again; res["B_cost"] = Bcost; res["success"] = bool(ok)
        proj[f"r{r}"] = res
        print(f"[PL] PROJ r={r}: A={res['A']:.3f} (gain {Again:+.3f})  B={res['B']:.3f} (cost {Bcost:+.3f})  loss={res['trainloss']:.3f}  -> {'SUCCESS' if ok else 'fail'} ({time.time()-t0:.0f}s)", flush=True)

    any_ok = any(proj[k]["success"] for k in proj)
    verdict = ("LEVER WORKS (>=1 r: A retained AND B not starved -> free lane is usable, forgetting preventable)" if any_ok
               else "LEVER FAILS pre-committed bar (no r retains A without starving B -> free lane less usable than geometry implied)")
    print(f"\n[PL] VERDICT: {verdict}", flush=True)
    json.dump({"A0_ceiling": A0, "baseline": base, "projected": proj, "fb_at_wA_rider": fbwa,
               "rsweep": RSWEEP, "success_criterion": "A_gain>=0.05 AND B_cost<=0.05", "any_success": any_ok,
               "verdict": verdict, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[PL] saved projection_lever.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
