"""Cheap-test batch, FIRST HALF (decides whether ANY traffic-sign works, or reverse-after is the only lever):
TEST #1 basis-eigenspace overlap (firm-or-scoped adjudication-negative): task-A Fisher F_A=E_A[g^T g] and
  task-B Fisher F_B=E_B[g^T g] per carrier (input-dir grad covariance, TASK-NEUTRAL: each task its OWN
  eigenbasis, NOT pre-binned into shared curvature shells). Principal-angle subspace overlap between
  top-k(F_A) and top-k(F_B) vs chance k/d. >>chance => tasks GENUINELY share directions (adjudication FIRM,
  stop/yield dead = no free lane). ~chance => curvature-binning manufactured the alignment (separable,
  adjudication back on table).
TEST #2a antagonism along-path (stop-sign vs roundabout): cosine(grad_A, grad_B) at w_A+t*(w_B-w_A),
  t in {.25,.5,.75} (OFF both minima so grad_A != 0), full + restricted to shared top-curvature dirs.
  Anti-aligned => no joint descent => antagonistic => STOP/YIELD. Positive => joint descent exists =>
  ROUNDABOUT geometrically possible. (Global half (b) = test #4 B-retention, second half.)
Carriers = qkv + mlp_h4h + mlp_4hh. Pythia/AG. DISCIPLINE: "compatible/roundabout" needs BOTH #2a positive
AND #4 retention (6th corroboration) - not banked here off #2a alone.
LAMBDA COPY: paths repointed to /home/ubuntu (Linux VM). Otherwise identical to firsthalf.py.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"; OUT = f"{G}/firsthalf_agnews.json"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_FISHER = 150; KTOP = 64; N_KFAC = 12
TS = [0.25, 0.5, 0.75]; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def data():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    trA = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:max(N_FISHER, 200)]]
    d = json.load(open(f"{AX}/low_agnews.json")); t = d["prompt_template"]
    B = [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]
    return trA, B


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


def main():
    t0 = time.time(); print("[FH] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA, B = data(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    # ---- K-FAC input-cov A per carrier at w_A (the shared top-curvature dirs for #2a restriction) ----
    model.eval(); cur = {}; H = []
    mods = {n: m for n, m in model.named_modules() if any(c in n for c in CARRIERS) and hasattr(m, "weight") and getattr(m.weight, "ndim", 0) == 2}
    Aacc = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1], device=dev) for n, m in mods.items()}; cnt = {n: 0 for n in mods}
    for n, m in mods.items():
        def fwd(mod, i, o, nn=n): cur[nn] = i[0].detach()
        H.append(m.register_forward_hook(fwd))
    gk = torch.Generator().manual_seed(7)
    for _ in range(N_KFAC):
        inp, lb, am = b_sst(tok, [trA[k] for k in torch.randperm(len(trA), generator=gk)[:BATCH].tolist()]); model(input_ids=inp, attention_mask=am)
        for n in mods: x = cur[n].reshape(-1, cur[n].shape[-1]); Aacc[n] += x.T @ x; cnt[n] += x.shape[0]
    for h in H: h.remove()
    Utop = {}                                                   # top-KTOP curvature input-dirs per module
    for n in names:
        mn = n[:-len(".weight")]; A = (Aacc[mn] / max(cnt[mn], 1)).double(); ev, U = torch.linalg.eigh(A); Utop[n] = U[:, -KTOP:].float()

    # ---- task-A Fisher at w_A ----
    wA = {n: params[n].detach().clone() for n in names}; FA = fisher(model, params, names, tok, trA, False)
    print(f"[FH] K-FAC + task-A Fisher done; training task-B... ({time.time()-t0:.0f}s)", flush=True)

    # ---- train task-B FULL ----
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
    model.eval(); wB = {n: params[n].detach().clone() for n in names}; dW = {n: wB[n] - wA[n] for n in names}
    FB = fisher(model, params, names, tok, B, True)
    print(f"[FH] task-B trained + task-B Fisher done ({time.time()-t0:.0f}s)", flush=True)

    # ---- TEST #1: per-carrier subspace overlap top-k(F_A) vs top-k(F_B) ----
    print("\n[FH] === TEST #1 BASIS-EIGENSPACE OVERLAP (task-A vs task-B top-Fisher subspaces, task-neutral) ===", flush=True)
    t1 = {}
    for c in CARRIERS:
        ovs = []; chs = []
        for n in [x for x in names if c in x]:
            _, VA = torch.linalg.eigh(FA[n]); _, VB = torch.linalg.eigh(FB[n])
            PA = VA[:, -KTOP:]; PB = VB[:, -KTOP:]; ov = float((PA.T @ PB).pow(2).sum().item() / KTOP); ovs.append(ov); chs.append(KTOP / FA[n].shape[0])
        t1[c] = {"overlap": float(np.mean(ovs)), "chance": float(np.mean(chs)), "ratio": float(np.mean(ovs) / np.mean(chs))}
        print(f"  [{c:26s}] top{KTOP} overlap={t1[c]['overlap']:.3f}  chance={t1[c]['chance']:.3f}  ratio={t1[c]['ratio']:.1f}x", flush=True)
    firm = all(t1[c]["ratio"] > 3 for c in CARRIERS)
    print(f"  -> adjudication-negative {'FIRM (tasks genuinely share dirs, >>chance -> stop/yield dead, no free lane)' if firm else 'SCOPED/REOPENED (near chance -> curvature-binning manufactured alignment, free lane exists)'}", flush=True)

    # ---- TEST #2a: along-path antagonism cosine(grad_A, grad_B) ----
    print("\n[FH] === TEST #2a ANTAGONISM along-path (cosine grad_A,grad_B in shared dirs) ===", flush=True)
    t2 = {}
    def grad_at(items, is_pair):
        model.zero_grad(set_to_none=True)
        if is_pair:
            inp, lb, am = b_pair(tok, items)
        else:
            inp, lb, am = b_sst(tok, items)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        return {n: params[n].grad.detach().double().clone() for n in names}
    Abatch = [trA[k] for k in range(BATCH)]; Bbatch = B[:BATCH]
    for t in TS:
        for n in names: params[n].data = (wA[n].double() + t * dW[n].double()).float().data   # carriers along path; non-carriers stay at w_B (context)
        gA = grad_at(Abatch, False); gB = grad_at(Bbatch, True)
        for n in names: params[n].data = wB[n].data
        per_c = {}
        for c in CARRIERS:
            cf = []; cs = []
            for n in [x for x in names if c in x]:
                a = gA[n].flatten(); b = gB[n].flatten(); cf.append(float((a @ b) / (a.norm() * b.norm() + 1e-30)))
                ap = (gA[n] @ Utop[n].double()).flatten(); bp = (gB[n] @ Utop[n].double()).flatten()
                cs.append(float((ap @ bp) / (ap.norm() * bp.norm() + 1e-30)))
            per_c[c] = {"cos_full": float(np.mean(cf)), "cos_shared": float(np.mean(cs))}
        t2[f"t={t}"] = per_c
        print(f"  t={t}: " + "  ".join(f"{c.split('.')[-1]}: full={per_c[c]['cos_full']:+.3f} shared={per_c[c]['cos_shared']:+.3f}" for c in CARRIERS), flush=True)
    shared_cos = np.mean([t2[f"t={t}"][c]["cos_shared"] for t in TS for c in CARRIERS])
    verdict2 = ("ANTAGONISTIC (anti-aligned in shared dirs -> no joint descent -> STOP/YIELD only)" if shared_cos < -0.05 else
                "COMPATIBLE-LEANING (positive in shared dirs -> joint descent exists -> ROUNDABOUT possible; CONFIRM with #4 retention before banking)" if shared_cos > 0.05 else
                "NEAR-ORTHOGONAL (cos~0 -> neither clearly antagonistic nor jointly-descending; #4 retention decides)")
    print(f"  mean shared-dir cosine = {shared_cos:+.3f} -> {verdict2}", flush=True)
    json.dump({"test1_overlap": t1, "test1_firm": firm, "test2a_cosines": t2, "test2a_mean_shared_cos": shared_cos,
               "test2a_verdict": verdict2, "ktop": KTOP, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"\n[FH] saved firsthalf_agnews.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
