"""UNIVERSAL-CORE — EMPIRICAL RANDOM-PB NULL (Wilson-mandated corroboration, the 8th catch one level deeper).
The contested-core mutual overlap (0.64, 2.5x ANALYTIC chance 0.25) is suspect because the core is a TWO-STAGE
construction (u_B = top-M_CORE left-svecs of PA^T PB) whose extraction could MANUFACTURE mutual overlap from
task-A's spectrum regardless of task-B. Analytic chance assumes the construction is unbiased; it may not be.

DECISIVE NULL: hold task-A fixed (same PA). Replace real task-B projectors PB with RANDOM orthonormal 64-frames
of the SAME RANK, run the IDENTICAL construction, measure mutual overlap of the random-derived cores. The null
has the construction bias baked in.
  random-PB null ~ 0.25  -> construction unbiased -> real 0.64 is REAL shared structure -> PARTIALLY-UNIVERSAL core.
  random-PB null ~ 0.64  -> construction manufactures it from task-A spectrum -> 0.64 is ARTIFACT -> task-specific.
Report the null DISTRIBUTION (mean + spread + percentiles), z-score the observed real, don't bank off z alone.
Computes REAL (retrain 3 task-Bs, same PA) + NULL (N_NULL random draws) in ONE run for apples-to-apples.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"; OUT = f"{G}/ucore_null.json"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
TASKS = [("agnews", "low_agnews"), ("yelp", "high_yelp"), ("subj", "medium_subj")]
KTOP = 64; M_CORE = 16; N_NULL = 200
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_FISHER = 150; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
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


def core_coords(PA_n, PB_n):
    """u = top-M_CORE left-singular vecs of PA^T PB  (task-A-basis coords of the contested core)."""
    M = PA_n.T @ PB_n
    U, S, _ = torch.linalg.svd(M)
    return U[:, :M_CORE].contiguous()


def mutual(u_list):
    """mean pairwise ||u_i^T u_j||_F^2 / M_CORE over the 3 cores."""
    o = []
    for a in range(len(u_list)):
        for b in range(a + 1, len(u_list)):
            o.append(float((u_list[a].T @ u_list[b]).pow(2).sum().item() / M_CORE))
    return float(np.mean(o))


def main():
    t0 = time.time(); print("[NULL] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    FA = fisher(model, params, names, tok, trA, False)
    PA = {n: torch.linalg.eigh(FA[n])[1][:, -KTOP:].contiguous() for n in names}   # float64, shared frame
    chance = M_CORE / KTOP
    print(f"[NULL] FA done; analytic chance={chance:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- REAL: train 3 task-Bs (same PA), real contested cores ----
    realU = {n: [] for n in names}
    for taskkey, jsonname in TASKS:
        model.load_state_dict(torch.load(CKPT)); B = load_taskB(jsonname)
        loss, steps = train_taskB(model, tok, B)
        FB = fisher(model, params, names, tok, B, True)
        for n in names:
            PB = torch.linalg.eigh(FB[n])[1][:, -KTOP:]
            realU[n].append(core_coords(PA[n], PB))
        print(f"[NULL] real {taskkey} loss={loss:.3f} steps={steps} ({time.time()-t0:.0f}s)", flush=True)
    real_by_name = {n: mutual(realU[n]) for n in names}

    # ---- NULL: random-PB of same rank, identical construction, N_NULL draws ----
    null_by_name = {n: [] for n in names}
    for d in range(N_NULL):
        gen = torch.Generator(device=dev).manual_seed(1000 + d)
        for n in names:
            din = PA[n].shape[0]
            us = []
            for _ in range(len(TASKS)):
                rg = torch.randn(din, KTOP, generator=gen, device=dev, dtype=torch.float64)
                Q, _ = torch.linalg.qr(rg)                                 # random orthonormal 64-frame
                us.append(core_coords(PA[n], Q[:, :KTOP]))
            null_by_name[n].append(mutual(us))
        if (d + 1) % 50 == 0: print(f"[NULL] null draw {d+1}/{N_NULL} ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate per carrier ----
    out = {}; print(f"\n[NULL] === REAL vs RANDOM-PB NULL (analytic chance={chance:.3f}) ===", flush=True)
    for c in CARRIERS:
        cn = [x for x in names if c in x]
        real_c = float(np.mean([real_by_name[n] for n in cn]))
        null_draws = np.array([np.mean([null_by_name[n][d] for n in cn]) for d in range(N_NULL)])
        nm, ns = float(null_draws.mean()), float(null_draws.std())
        z = (real_c - nm) / (ns + 1e-12)
        pctile = float((null_draws < real_c).mean() * 100)
        out[c] = {"real": real_c, "null_mean": nm, "null_std": ns, "null_p5": float(np.percentile(null_draws, 5)),
                  "null_p95": float(np.percentile(null_draws, 95)), "z": z, "real_pctile_vs_null": pctile}
        print(f"  [{c:26s}] real={real_c:.3f}  null={nm:.3f}±{ns:.3f} (p5={out[c]['null_p5']:.3f},p95={out[c]['null_p95']:.3f})  z={z:+.1f}  real>{pctile:.0f}% of null", flush=True)

    grand_real = float(np.mean([out[c]["real"] for c in CARRIERS]))
    grand_nullmean = float(np.mean([out[c]["null_mean"] for c in CARRIERS]))
    grand_z = float(np.mean([out[c]["z"] for c in CARRIERS]))
    real_is_signal = grand_z > 3 and grand_real > grand_nullmean + 0.10
    verdict = (f"REAL SHARED STRUCTURE — partially-universal contested core (real {grand_real:.2f} >> random-PB null "
               f"{grand_nullmean:.2f}, z={grand_z:+.1f}) -> construction is NOT manufacturing it; the cores genuinely point the same way"
               if real_is_signal else
               f"CONSTRUCTION ARTIFACT / task-specific (real {grand_real:.2f} ~ random-PB null {grand_nullmean:.2f}, z={grand_z:+.1f}) "
               f"-> the two-stage extraction makes ~this overlap from random task-B too; no genuine shared core")
    print(f"\n[NULL] grand: real={grand_real:.3f}  random-PB null={grand_nullmean:.3f}  z={grand_z:+.1f}\n[NULL] VERDICT: {verdict}", flush=True)
    json.dump({"per_carrier": out, "grand_real": grand_real, "grand_null_mean": grand_nullmean, "grand_z": grand_z,
               "analytic_chance": chance, "ktop": KTOP, "m_core": M_CORE, "n_null": N_NULL,
               "real_is_signal": real_is_signal, "verdict": verdict, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[NULL] saved ucore_null.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
