"""DYE 1 — passive fine-grained tracer (dye-in-the-water program, Wilson-scoped).
Watch the natural forgetting-diffusion at high resolution: no injection, just a better camera.
Dense early snapshots (0,1,2,3,5,10,25,50,100,200,400,800); step 0 = the initial task-B pull at the task-A
minimum (the seed BEFORE movement). Per-direction in the CURVATURE basis (task-A Fisher eigvecs), so Dye 3 can
build the flow field off this exact data. Substrate: Pythia / AG-News (where C1-C5 are strongest).

READOUTS:
 (a) SEED PR (step 0) = effective # of initial droplets = how localized the seed is (per-layer PR, mean per carrier).
 (b) FRONT ADVANCE = wetted-rank count (dirs with damage > thresh) vs step; + front position (deepest wetted rank).
 (c) CHANNELED vs ISOTROPIC = damage center-of-mass over curvature-RANK vs step (low com = stays at high curvature
     = channeled; growing com = spreads to low-curvature = isotropic-ward) + PR-over-rank vs step.
SAVES per-carrier damage(step, rank) + seed(rank) to dye1_series.npz (the camera frame for Dye 3 flow analysis).
Directions indexed by CURVATURE RANK within each layer's Fisher eigenbasis (rank 0 = highest curvature), averaged
across layers — comparable across layers (gauge-correct).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/dye1.json"; NPZ = f"{G}/dye1_series.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
SNAP = [1, 2, 3, 5, 10, 25, 50, 100, 200, 400, 800]; MAXSTEP = SNAP[-1]
KEEP = 1024; N_FISHER = 150; N_SEED = 8
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


def pr(x):  # participation ratio of a nonneg vector (numpy or tensor)
    x = np.asarray(x, dtype=np.float64); s1 = x.sum(); s2 = (x * x).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def main():
    t0 = time.time(); print("[DYE1] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); B = load_taskB("low_agnews"); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    # curvature basis at w_A: top-KEEP Fisher eigvecs per layer, rank 0 = highest curvature
    FA = fisher(model, params, names, tok, trA)
    Vk = {}; KEEPn = {}
    for n in names:
        k = min(KEEP, FA[n].shape[0]); KEEPn[n] = k
        V = torch.linalg.eigh(FA[n])[1][:, -k:]            # ascending -> last = highest
        Vk[n] = torch.flip(V, dims=[1]).float().contiguous()   # rank 0 = highest curvature
    wA = {n: params[n].detach().clone() for n in names}
    print(f"[DYE1] curvature basis done ({time.time()-t0:.0f}s)", flush=True)

    def project_damage(delta):  # per carrier: mean over layers of per-direction squared projection, ranked
        out = {}
        for c in CARRIERS:
            cn = [x for x in names if c in x]; k = KEEPn[cn[0]]
            acc = torch.zeros(k, device=dev)
            for n in cn:
                d = (delta[n].float() @ Vk[n]).pow(2).sum(0)   # (k,) per curvature-rank damage
                acc += d
            out[c] = (acc / len(cn)).cpu().numpy()
        return out

    # ---- SEED (step 0): task-B gradient at w_A ----
    model.eval(); model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(0)
    for _ in range(N_SEED):
        inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
        (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_SEED).backward()
    seed = {c: project_damage({n: params[n].grad.detach() for n in names})[c] for c in CARRIERS}
    model.zero_grad(set_to_none=True)
    print(f"[DYE1] seed captured ({time.time()-t0:.0f}s)", flush=True)

    # ---- train task-B, snapshot Delta_w at dense steps ----
    series = {c: {} for c in CARRIERS}                      # carrier -> step -> damage(rank)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); gt = torch.Generator().manual_seed(0)
    step = 0; snapset = set(SNAP)
    while step < MAXSTEP:
        for i in range(0, len(B) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=gt)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            if step in snapset:
                dmg = project_damage({n: (params[n].detach() - wA[n]) for n in names})
                for c in CARRIERS: series[c][step] = dmg[c]
                print(f"[DYE1] snap step={step} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step >= MAXSTEP: break

    # ---- SAVE CAMERA FRAME FIRST (never lose expensive training data to a readout bug) ----
    sd = {f"seed__{c}": np.asarray(seed[c]) for c in CARRIERS}
    for c in CARRIERS:
        sd[f"series__{c}"] = np.stack([series[c][s] for s in SNAP])     # (n_snap, KEEP)
    sd["snap_steps"] = np.array(SNAP)
    np.savez(NPZ, **sd)
    print(f"[DYE1] saved camera frame dye1_series.npz ({time.time()-t0:.0f}s)", flush=True)

    # ---- readouts (pure numpy) ----
    steps_all = [0] + SNAP
    def dmg_at(c, s): return np.asarray(seed[c] if s == 0 else series[c][s], dtype=np.float64)
    report = {}
    try:
        for c in CARRIERS:
            seedpr = pr(seed[c]); wetted, front, com, prank = [], [], [], []
            for s in steps_all:
                d = dmg_at(c, s); mask = d > d.max() * 0.01
                wetted.append(int(mask.sum()))
                front.append(int(np.nonzero(mask)[0].max()) if mask.any() else 0)
                com.append(float((np.arange(len(d)) * d).sum() / (d.sum() + 1e-30)))
                prank.append(pr(d))
            report[c] = {"seed_PR": seedpr, "steps": steps_all, "wetted_count": wetted, "front_rank": front,
                         "com_rank": com, "PR_over_rank": prank}
            print(f"  [{c:26s}] seedPR={seedpr:.1f}  wetted {wetted[0]}->{wetted[-1]}  com_rank {com[0]:.1f}->{com[-1]:.1f}  PR_rank {prank[0]:.0f}->{prank[-1]:.0f}", flush=True)
    except Exception as e:
        report["error"] = str(e); print(f"[DYE1] readout error (camera frame already saved): {e}", flush=True)
    json.dump({"report": report, "snap": SNAP, "keep": KEEP, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[DYE1] saved dye1.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
