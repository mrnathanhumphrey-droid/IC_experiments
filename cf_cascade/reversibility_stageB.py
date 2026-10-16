"""Stage B — deployable encapsulation recipe: minimal targeted reversal that recovers task-A AND
retains task-B. Redundancy is the lever (reverse a thin hi-A slice of ONE pathway, let the other carry B).

4-quantity map per curvature shell S, on CARRIERS = qkv + mlp (NOT attention.dense = minority):
  1 task-A Fisher in S        2 task-B Fisher in S  (targeting criterion)
  3 task-A recovery (acc restored by reversing S)   4 task-B retention (loss change) + MAGNITUDE CONTROL
    (vs random same-size slice: "B retained" only if a random slice that size WOULD hurt B)
Predictive RULE: hi-A-importance & lo-B-importance (pre-reversal) -> predicts hi-recovery & hi-retention.
Match -> a RULE not a cell. ADJUDICATION read: does the hi-A-lo-B cell EXIST? -> live protect-one-pathway
routing buildable; else reverse-after-only.

Pythia (all Linear weight (out,in)); shells = input-axis K-FAC input-cov U_A per carrier type, 16 shells.
argv[1]=taskB key (low_agnews|high_yelp). Validity: val_A 872; flag if forgetting<0.03.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
G = "D:/IC_experiments/cf_cascade/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
TASKB = sys.argv[1] if len(sys.argv) > 1 else "low_agnews"; OUT = f"{G}/stageB_{TASKB}.json"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_SHELLS = 16; N_FISHER = 150; N_KFAC = 12
VAL_A = 872; VAL_B = 400; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def data():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    vaA = [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"] if r["label"] in (0, 1)][:VAL_A]
    trA = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)]
    trA = [trA[i] for i in np.random.default_rng(0).permutation(len(trA))[:N_FISHER]]
    d = json.load(open(f"{AX}/{TASKB}.json")); t = d["prompt_template"]
    B = [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]
    return trA, vaA, B


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


@torch.no_grad()
def accA(model, tok, va):
    Lc = {}
    for cl in (0, 1):
        per = []; items = [(s, cl) for s, _ in va]
        for i in range(0, len(items), 64):
            inp, lb, am = b_sst(tok, items[i:i + 64]); lg = model(input_ids=inp, attention_mask=am).logits
            ll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), lb[:, 1:].reshape(-1), ignore_index=-100, reduction="none").reshape(lb.shape[0], -1)
            per.extend((ll.sum(1) / (lb[:, 1:] != -100).sum(1).clamp(min=1)).tolist())
        Lc[cl] = per
    return float(np.mean([int((0 if Lc[0][i] < Lc[1][i] else 1) == va[i][1]) for i in range(len(va))]))


@torch.no_grad()
def lossB(model, tok, items):
    tot = 0.0; nb = 0
    for i in range(0, len(items), 32):
        inp, lb, am = b_pair(tok, items[i:i + 32]); tot += float(model(input_ids=inp, attention_mask=am, labels=lb).loss); nb += 1
    return tot / nb


def main():
    t0 = time.time(); print(f"[B] taskB={TASKB} loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); trA, vaA, Bitems = data(); Bval = Bitems[:VAL_B]
    params = dict(model.named_parameters())
    names = {c: [n for n in params if c in n and n.endswith("weight")] for c in CARRIERS}
    din = {c: params[names[c][0]].shape[1] for c in CARRIERS}

    # ---- K-FAC input-cov U_A per carrier (at w_A) + shells ----
    model.eval(); modules = {n: m for n, m in model.named_modules() if any(c in n for c in CARRIERS) and hasattr(m, "weight") and m.weight.ndim == 2}
    Aacc = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1], device=dev) for n, m in modules.items()}; cnt = {n: 0 for n in modules}; cur = {}; H = []
    for n, m in modules.items():
        def fwd(mod, i, o, nn=n): cur[nn] = i[0].detach()
        H.append(m.register_forward_hook(fwd))
    gk = torch.Generator().manual_seed(7)
    for _ in range(N_KFAC):
        inp, lb, am = b_sst(tok, [vaA[k] for k in torch.randperm(len(vaA), generator=gk)[:BATCH].tolist()])
        model(input_ids=inp, attention_mask=am)
        for n in modules:
            x = cur[n].reshape(-1, cur[n].shape[-1]); Aacc[n] += x.T @ x; cnt[n] += x.shape[0]
    for h in H: h.remove()
    UA = {}; shell_of = {}
    for c in CARRIERS:
        eigs = []; vecs = {}
        for n in names[c]:
            mn = n[:-len(".weight")]; A = (Aacc[mn] / max(cnt[mn], 1)).double(); ev, U = torch.linalg.eigh(A)
            UA[n] = U.float(); eigs.append(ev.cpu().numpy())
        pooled = np.concatenate(eigs); order = np.argsort(pooled); sid = np.empty(len(pooled), int)
        edg = np.linspace(0, len(pooled), N_SHELLS + 1).astype(int)
        for s in range(N_SHELLS): sid[order[edg[s]:edg[s + 1]]] = s
        offs = np.cumsum([0] + [din[c]] * len(names[c]))
        shell_of[c] = {n: sid[offs[i]:offs[i + 1]] for i, n in enumerate(names[c])}
    print(f"[B] K-FAC shells built ({time.time()-t0:.0f}s)", flush=True)

    # ---- baselines + task-A Fisher at w_A ----
    acc_A = accA(model, tok, vaA); wA = {n: params[n].detach().clone() for n in sum(names.values(), [])}
    fishA = {n: torch.zeros(params[n].shape[1], device=dev, dtype=torch.float64) for n in sum(names.values(), [])}
    for s, l in trA:
        inp, lb, am = b_sst(tok, [(s, l)]); model.zero_grad(set_to_none=True); model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in fishA: g = params[n].grad.detach().double(); fishA[n] += ((g @ UA[n].double()) ** 2).sum(0)
    print(f"[B] acc_A={acc_A:.3f}, task-A Fisher done; training task-B... ({time.time()-t0:.0f}s)", flush=True)

    # ---- train task-B FULL ----
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0); run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        for i in range(0, len(Bitems) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [Bitems[k] for k in torch.randperm(len(Bitems), generator=g)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); acc_B = accA(model, tok, vaA); lossB_base = lossB(model, tok, Bval); forgetting = acc_A - acc_B
    wB = {n: params[n].detach().clone() for n in wA}; dW = {n: wB[n] - wA[n] for n in wA}
    Ccoord = {n: (dW[n].double() @ UA[n].double()) for n in wA}                    # (out, in_comp)
    valid = bool(forgetting > 0.03)
    print(f"[B] acc_B={acc_B:.3f} forget={forgetting:+.3f} taskB_loss_base={lossB_base:.3f} valid={valid} ({time.time()-t0:.0f}s)", flush=True)

    # ---- task-B Fisher at w_B ----
    fishB = {n: torch.zeros(params[n].shape[1], device=dev, dtype=torch.float64) for n in wA}
    for it in Bitems[:N_FISHER]:
        inp, lb, am = b_pair(tok, [it]); model.zero_grad(set_to_none=True); model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in fishB: g = params[n].grad.detach().double(); fishB[n] += ((g @ UA[n].double()) ** 2).sum(0)
    for n in wA: params[n].data = wB[n].data                                       # back to w_B
    print(f"[B] task-B Fisher done ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0); res = {}
    for c in CARRIERS:
        impA = np.zeros(N_SHELLS); impB = np.zeros(N_SHELLS); szs = np.zeros(N_SHELLS, int)
        for n in names[c]:
            fa = fishA[n].cpu().numpy(); fb = fishB[n].cpu().numpy(); sm = shell_of[c][n]
            for s in range(N_SHELLS): impA[s] += fa[sm == s].sum(); impB[s] += fb[sm == s].sum(); szs[s] += int((sm == s).sum())
        recov = np.zeros(N_SHELLS); reten = np.zeros(N_SHELLS); reten_rand = np.zeros(N_SHELLS)
        for s in range(N_SHELLS):
            for n in names[c]:
                mask = torch.tensor(shell_of[c][n] == s, device=dev); Cs = Ccoord[n].clone(); Cs[:, mask] = 0
                params[n].data = (wA[n].double() + (Cs @ UA[n].double().T)).float().data
            recov[s] = (accA(model, tok, vaA) - acc_B) / max(forgetting, 1e-9); reten[s] = lossB(model, tok, Bval) - lossB_base
            for n in names[c]: params[n].data = wB[n].data
            # magnitude control: random same-size slice per matrix
            for n in names[c]:
                k = int((shell_of[c][n] == s).sum()); idx = torch.tensor(rng.choice(params[n].shape[1], k, replace=False), device=dev)
                Cs = Ccoord[n].clone(); Cs[:, idx] = 0; params[n].data = (wA[n].double() + (Cs @ UA[n].double().T)).float().data
            reten_rand[s] = lossB(model, tok, Bval) - lossB_base
            for n in names[c]: params[n].data = wB[n].data
        res[c] = {"impA": (impA / impA.max()).tolist(), "impB": (impB / impB.max()).tolist(), "sizes": szs.tolist(),
                  "recovery": recov.tolist(), "retention_lossinc": reten.tolist(), "retention_rand_lossinc": reten_rand.tolist()}
        # adjudication: hi-A-lo-B shells -> do they recover AND retain (loss-inc < random)?
        Ahi = impA >= np.median(impA); Blo = impB <= np.median(impB); hiAloB = Ahi & Blo
        rec_hi = recov >= 0.3; ret_ok = reten < reten_rand                          # reversing S hurts B LESS than random same-size
        deployable = hiAloB & rec_hi & ret_ok
        res[c]["hiAloB_shells"] = [int(s) for s in np.where(hiAloB)[0]]
        res[c]["deployable_shells"] = [int(s) for s in np.where(deployable)[0]]
        res[c]["rule_match"] = float(np.mean(rec_hi[hiAloB]) if hiAloB.any() else 0.0)
        print(f"\n[B] [{c}] hi-A-lo-B shells={res[c]['hiAloB_shells']} | DEPLOYABLE (hiAloB & recovers & B-spared)={res[c]['deployable_shells']}", flush=True)
        print(f"     recovery: {np.round(recov,2).tolist()}", flush=True)
        print(f"     retention loss-inc real: {np.round(reten,3).tolist()}", flush=True)
        print(f"     retention loss-inc rand: {np.round(reten_rand,3).tolist()}", flush=True)
        print(f"     impA: {np.round(impA/impA.max(),2).tolist()}  impB: {np.round(impB/impB.max(),2).tolist()}", flush=True)

    json.dump({"taskB": TASKB, "acc_A": acc_A, "acc_B": acc_B, "forgetting": forgetting, "valid": valid,
               "taskB_loss_base": lossB_base, "carriers": res, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"\n[B] saved stageB_{TASKB}.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
