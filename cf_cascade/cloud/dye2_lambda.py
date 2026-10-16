"""DYE 2 — active injection tracer (dye-in-the-water, Wilson-scoped). The PREDICTIVE mechanism test.
Inject a controlled perturbation at the task-A minimum and clock its spread:
  run STIFF = inject along a high-curvature input direction; run FLAT = inject along a low-curvature one. SAME norm.
Isolate the injected dye from natural forgetting via PAIRED runs (same data seed): Delta_inject(t) = w_pert(t) -
w_baseline(t); project onto the curvature basis, track spread (PR/wetted/front) over steps.
PRE-REGISTERED: stiff-injected dye spreads FASTER (PR grows faster) than flat -> curvature-graded TRANSPORT.
FALSIFIER: same spread rate -> grading is seeding-driven not transport-driven (grading lives in where damage seeds).
Either answer informative. Substrate Pythia/AG. Injection norm = c*||w_A_layer|| (same c for stiff & flat = fair).
SAVES damage series BEFORE readouts (lesson). Output dye2.json + dye2_series.npz.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/dye2.json"; NPZ = f"{G}/dye2_series.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
SNAP = [1, 2, 3, 5, 10, 25, 50, 100, 200, 400, 800]; MAXSTEP = SNAP[-1]; SNAP0 = [0] + SNAP
KEEP = 1024; N_FISHER = 150; INJ_C = 0.02      # injection norm = 2% of layer weight norm
BATCH = 16; LR = 5e-5; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
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


def fisher(model, params, names, tok, items):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for it in items[:N_FISHER]:
        inp, lb, am = b_sst(tok, [it]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def pr(x):
    x = np.asarray(x, dtype=np.float64); s1 = x.sum(); s2 = (x * x).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def train_snapshot(model, params, names, tok, B, base_snaps, Vk, KEEPn, record_base, dmg_out, wA):
    """Train task-B; if record_base, save per-layer weight snaps (fp16 CPU) into base_snaps;
    else compute Delta=w-base_snaps[step], project squared per curvature-rank into dmg_out[carrier][step]."""
    g = torch.Generator().manual_seed(0); step = 0; snapset = set(SNAP)
    def snap(s):
        if record_base:
            base_snaps[s] = {n: params[n].detach().half().cpu() for n in names}
        else:
            for c in CARRIERS:
                cn = [x for x in names if c in x]; k = KEEPn[cn[0]]; acc = torch.zeros(k, device=dev)
                for n in cn:
                    delta = params[n].detach() - base_snaps[s][n].float().to(dev)
                    acc += (delta @ Vk[n]).pow(2).sum(0)
                dmg_out[c][s] = (acc / len(cn)).cpu().numpy()
    snap(0)                                                   # step 0 = w_A (base) or w_A+inj (pert)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    while step < MAXSTEP:
        for i in range(0, len(B) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            if step in snapset: snap(step)
            if step >= MAXSTEP: break
    model.eval()


def main():
    t0 = time.time(); print("[DYE2] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); B = load_taskB("low_agnews"); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    FA = fisher(model, params, names, tok, trA)
    Vk = {}; KEEPn = {}; vstiff = {}; vflat = {}; inj_stiff = {}; inj_flat = {}
    for n in names:
        ev, V = torch.linalg.eigh(FA[n]); k = min(KEEP, V.shape[1]); KEEPn[n] = k
        Vk[n] = torch.flip(V[:, -k:], dims=[1]).float().contiguous()   # rank 0 = highest curvature
        vstiff[n] = V[:, -1].float()                                   # highest-curvature dir (tracked rank 0)
        vflat[n] = V[:, -k].float()                                    # lowest-curv WITHIN top-k (tracked rank k-1, in-basis)
        dout = params[n].shape[0]; u = torch.ones(dout, device=dev) / (dout ** 0.5)
        eps = INJ_C * params[n].detach().norm()
        inj_stiff[n] = eps * torch.outer(u, vstiff[n])                 # rank-1: u outer v, ||.||_F = eps
        inj_flat[n] = eps * torch.outer(u, vflat[n])
    print(f"[DYE2] curvature basis + injections built ({time.time()-t0:.0f}s)", flush=True)

    # BASELINE run (record weight snaps)
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    base_snaps = {}
    train_snapshot(model, params, names, tok, B, base_snaps, Vk, KEEPn, True, None, None)
    print(f"[DYE2] baseline done, {len(base_snaps)} snaps ({time.time()-t0:.0f}s)", flush=True)

    # STIFF run
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    for n in names: params[n].data += inj_stiff[n]
    dmg_stiff = {c: {} for c in CARRIERS}
    train_snapshot(model, params, names, tok, B, base_snaps, Vk, KEEPn, False, dmg_stiff, None)
    print(f"[DYE2] stiff-injection run done ({time.time()-t0:.0f}s)", flush=True)

    # FLAT run
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    for n in names: params[n].data += inj_flat[n]
    dmg_flat = {c: {} for c in CARRIERS}
    train_snapshot(model, params, names, tok, B, base_snaps, Vk, KEEPn, False, dmg_flat, None)
    print(f"[DYE2] flat-injection run done ({time.time()-t0:.0f}s)", flush=True)

    # SAVE series first (lesson)
    sd = {}
    for c in CARRIERS:
        sd[f"stiff__{c}"] = np.stack([dmg_stiff[c][s] for s in SNAP0])
        sd[f"flat__{c}"] = np.stack([dmg_flat[c][s] for s in SNAP0])
    sd["snap0"] = np.array(SNAP0)
    np.savez(NPZ, **sd)
    print(f"[DYE2] saved dye2_series.npz ({time.time()-t0:.0f}s)", flush=True)

    # readouts: spread (PR over curvature ranks) of injected dye, stiff vs flat, over steps
    report = {}
    try:
        for c in CARRIERS:
            pr_s = [pr(dmg_stiff[c][s]) for s in SNAP0]; pr_f = [pr(dmg_flat[c][s]) for s in SNAP0]
            report[c] = {"steps": SNAP0, "PR_stiff": pr_s, "PR_flat": pr_f,
                         "PR_stiff_final": pr_s[-1], "PR_flat_final": pr_f[-1]}
            print(f"  [{c:26s}] PR stiff {pr_s[0]:.1f}->{pr_s[-1]:.1f}  flat {pr_f[0]:.1f}->{pr_f[-1]:.1f}", flush=True)
        # verdict: stiff spreads faster (final PR) across carriers?
        sf = float(np.mean([report[c]["PR_stiff_final"] for c in CARRIERS]))
        ff = float(np.mean([report[c]["PR_flat_final"] for c in CARRIERS]))
        ratio = sf / (ff + 1e-9)
        verdict = (f"TRANSPORT GRADED (stiff dye spreads faster: final PR stiff {sf:.0f} vs flat {ff:.0f}, {ratio:.1f}x) "
                   f"-> curvature-graded TRANSPORT confirmed (prediction held)" if ratio > 1.3 else
                   f"SAME RATE (stiff {sf:.0f} ~ flat {ff:.0f}, {ratio:.1f}x) -> grading is SEEDING-driven not transport "
                   f"(falsifier: graded rate lives in where damage seeds, not how perturbations transport)")
        report["verdict"] = verdict; print(f"\n[DYE2] {verdict}", flush=True)
    except Exception as e:
        report["error"] = str(e); print(f"[DYE2] readout error (series saved): {e}", flush=True)
    json.dump({"report": report, "snap0": SNAP0, "inj_c": INJ_C, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[DYE2] saved dye2.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
