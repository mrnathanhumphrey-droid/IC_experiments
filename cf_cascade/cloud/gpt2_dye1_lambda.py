"""GPT-2 DYE 1 (Stage 1, HOLD test) — passive fine tracer on GPT-2/AG. Reuses gpt2_dye_basis.npz (saved Fisher
curvature basis from Stage 0, rank0=highest curv) -> NO Fisher recompute. Conv1D orientation: weight/dw (in,out),
Vk (in,KEEP), damage = (Vk^T @ dw)^2 summed over out -> (KEEP,). Dense early snaps; step0 = seed (task-B grad at w_A).
Readouts (same as Pythia dye1): seed PR, wetted-count, com-rank, PR-over-rank. Saves gpt2_dye1_series.npz for Dye3.
Tests the COMMITTED prediction: HOLD (laminar/ordering) ; seed LESS localized than Pythia (higher PR) per continuum.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "gpt2-medium"; CKPT = f"{G}/gpt2med_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"; BASIS = f"{G}/gpt2_dye_basis.npz"
OUT = f"{G}/gpt2_dye1.json"; NPZ = f"{G}/gpt2_dye1_series.npz"
CARRIERS = ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"]
SNAP = [1, 2, 3, 5, 10, 25, 50, 100, 200, 400, 800]; MAXSTEP = SNAP[-1]
N_SEED = 8; BATCH = 16; LR = 5e-5; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def load_taskB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def b_pair(tok, items):
    seqs, labs = [], []
    for pr_, co in items:
        p = tok(pr_, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def pack(tok, seqs, labs):
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def pr(x):
    x = np.asarray(x, dtype=np.float64); s1 = x.sum(); s2 = (x * x).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def main():
    t0 = time.time(); print("[G2D1] loading gpt2-medium + basis...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    B = load_taskB("low_agnews"); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    bz = np.load(BASIS); Vk = {n: torch.tensor(bz[f"V__{n}"]).float().to(dev) for n in names}   # (in, KEEP), rank0=hi curv
    KEEPn = {n: Vk[n].shape[1] for n in names}
    wA = {n: params[n].detach().clone() for n in names}
    print(f"[G2D1] {len(names)} carriers, basis loaded ({time.time()-t0:.0f}s)", flush=True)

    def project_damage(delta):   # Conv1D: dw (in,out), Vk (in,KEEP) -> (Vk^T @ dw)^2 sum over out
        out = {}
        for c in CARRIERS:
            cn = [x for x in names if c in x]; k = KEEPn[cn[0]]; acc = torch.zeros(k, device=dev)
            for n in cn:
                acc += (Vk[n].T @ delta[n].float()).pow(2).sum(1)
            out[c] = (acc / len(cn)).cpu().numpy()
        return out

    # SEED (step 0): task-B grad at w_A
    model.eval(); model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(0)
    for _ in range(N_SEED):
        inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
        (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_SEED).backward()
    seed = project_damage({n: params[n].grad.detach() for n in names}); model.zero_grad(set_to_none=True)
    print(f"[G2D1] seed captured ({time.time()-t0:.0f}s)", flush=True)

    # train task-B, snapshot Delta_w
    series = {c: {} for c in CARRIERS}
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); gt = torch.Generator().manual_seed(0); step = 0; snapset = set(SNAP)
    while step < MAXSTEP:
        for i in range(0, len(B) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=gt)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            if step in snapset:
                dmg = project_damage({n: (params[n].detach() - wA[n]) for n in names})
                for c in CARRIERS: series[c][step] = dmg[c]
                print(f"[G2D1] snap step={step} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step >= MAXSTEP: break

    # SAVE camera frame FIRST
    sd = {f"seed__{c}": np.asarray(seed[c]) for c in CARRIERS}
    for c in CARRIERS: sd[f"series__{c}"] = np.stack([series[c][s] for s in SNAP])
    sd["snap_steps"] = np.array(SNAP); np.savez(NPZ, **sd)
    print(f"[G2D1] saved camera frame ({time.time()-t0:.0f}s)", flush=True)

    # readouts (pure numpy)
    steps_all = [0] + SNAP
    def dmg_at(c, s): return np.asarray(seed[c] if s == 0 else series[c][s], dtype=np.float64)
    report = {}
    try:
        for c in CARRIERS:
            seedpr = pr(seed[c]); wetted, com, prank = [], [], []
            for s in steps_all:
                d = dmg_at(c, s); mask = d > d.max() * 0.01
                wetted.append(int(mask.sum())); com.append(float((np.arange(len(d)) * d).sum() / (d.sum() + 1e-30))); prank.append(pr(d))
            report[c] = {"seed_PR": seedpr, "steps": steps_all, "wetted_count": wetted, "com_rank": com, "PR_over_rank": prank}
            print(f"  [{c:12s}] seedPR={seedpr:.1f}  wetted {wetted[0]}->{wetted[-1]}  com {com[0]:.0f}->{com[-1]:.0f}  PR_rank {prank[0]:.0f}->{prank[-1]:.0f}", flush=True)
    except Exception as e:
        report["error"] = str(e); print(f"[G2D1] readout error (frame saved): {e}", flush=True)
    json.dump({"report": report, "snap": SNAP, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[G2D1] saved gpt2_dye1.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
