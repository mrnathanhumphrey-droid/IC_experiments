"""GPT-2 DYE 2 (Stage 1, MOVE test) — active injection on GPT-2/AG. Tests the COMMITTED prediction: WEAKER grading
than Pythia's 1.4× transient peak (GPT-2 continuum spectrum -> gentler stiff/flat curvature contrast).
Conv1D orientation: weight/dw/inj (in,out); inject along input dir v: inj = eps*outer(v_in, u_out); damage =
(Vk^T @ dw)^2 summed over out. Reuses gpt2_dye_basis.npz (vstiff=Vk[:,0]=highest curv, vflat=Vk[:,KEEP-1]=lowest-in-top-K).
Paired runs (baseline + stiff-inj + flat-inj, same seed); Delta_inject = w_pert - w_base; track spread-PR vs steps.
Verdict on TRANSIENT trajectory (rise->peak->converge), not endpoint. Saves series BEFORE readouts.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "gpt2-medium"; CKPT = f"{G}/gpt2med_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"; BASIS = f"{G}/gpt2_dye_basis.npz"
OUT = f"{G}/gpt2_dye2.json"; NPZ = f"{G}/gpt2_dye2_series.npz"
CARRIERS = ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"]
SNAP = [1, 2, 3, 5, 10, 25, 50, 100, 200, 400, 800]; MAXSTEP = SNAP[-1]; SNAP0 = [0] + SNAP
INJ_C = 0.02; BATCH = 16; LR = 5e-5; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
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


def train_snapshot(model, params, names, tok, B, base_snaps, Vk, record_base, dmg_out):
    g = torch.Generator().manual_seed(0); step = 0; snapset = set(SNAP)
    def snap(s):
        if record_base:
            base_snaps[s] = {n: params[n].detach().half().cpu() for n in names}
        else:
            for c in CARRIERS:
                cn = [x for x in names if c in x]; k = Vk[cn[0]].shape[1]; acc = torch.zeros(k, device=dev)
                for n in cn:
                    delta = params[n].detach() - base_snaps[s][n].float().to(dev)   # (in,out)
                    acc += (Vk[n].T @ delta).pow(2).sum(1)                          # Conv1D: (KEEP,out)->sum over out
                dmg_out[c][s] = (acc / len(cn)).cpu().numpy()
    snap(0)
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
    t0 = time.time(); print("[G2D2] loading gpt2-medium + basis...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    B = load_taskB("low_agnews"); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    bz = np.load(BASIS); Vk = {n: torch.tensor(bz[f"V__{n}"]).float().to(dev) for n in names}   # (in,KEEP) rank0=hi curv
    inj_stiff = {}; inj_flat = {}
    for n in names:
        k = Vk[n].shape[1]; vstiff = Vk[n][:, 0]; vflat = Vk[n][:, k - 1]            # in-basis dirs
        out = params[n].shape[1]; u = torch.ones(out, device=dev) / (out ** 0.5)
        eps = INJ_C * params[n].detach().norm()
        inj_stiff[n] = eps * torch.outer(vstiff, u)                                  # Conv1D (in,out), ||.||_F=eps
        inj_flat[n] = eps * torch.outer(vflat, u)
    print(f"[G2D2] basis + injections built ({time.time()-t0:.0f}s)", flush=True)

    # BASELINE (record snaps)
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    base_snaps = {}; train_snapshot(model, params, names, tok, B, base_snaps, Vk, True, None)
    print(f"[G2D2] baseline done ({time.time()-t0:.0f}s)", flush=True)
    # STIFF
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    for n in names: params[n].data += inj_stiff[n]
    dmg_stiff = {c: {} for c in CARRIERS}; train_snapshot(model, params, names, tok, B, base_snaps, Vk, False, dmg_stiff)
    print(f"[G2D2] stiff run done ({time.time()-t0:.0f}s)", flush=True)
    # FLAT
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    for n in names: params[n].data += inj_flat[n]
    dmg_flat = {c: {} for c in CARRIERS}; train_snapshot(model, params, names, tok, B, base_snaps, Vk, False, dmg_flat)
    print(f"[G2D2] flat run done ({time.time()-t0:.0f}s)", flush=True)

    sd = {}
    for c in CARRIERS:
        sd[f"stiff__{c}"] = np.stack([dmg_stiff[c][s] for s in SNAP0]); sd[f"flat__{c}"] = np.stack([dmg_flat[c][s] for s in SNAP0])
    sd["snap0"] = np.array(SNAP0); np.savez(NPZ, **sd)
    print(f"[G2D2] saved series ({time.time()-t0:.0f}s)", flush=True)

    report = {}
    try:
        for c in CARRIERS:
            pr_s = [pr(dmg_stiff[c][s]) for s in SNAP0]; pr_f = [pr(dmg_flat[c][s]) for s in SNAP0]
            ratio = [pr_s[i] / (pr_f[i] + 1e-9) for i in range(len(SNAP0))]
            peak = max(ratio[1:]); peakstep = SNAP0[1:][int(np.argmax(ratio[1:]))]
            report[c] = {"steps": SNAP0, "PR_stiff": pr_s, "PR_flat": pr_f, "ratio": ratio,
                         "peak_ratio": peak, "peak_step": peakstep, "final_ratio": ratio[-1]}
            print(f"  [{c:12s}] PR stiff {pr_s[0]:.1f}->{pr_s[-1]:.1f} flat {pr_f[0]:.1f}->{pr_f[-1]:.1f} | peak ratio {peak:.2f}@{peakstep} final {ratio[-1]:.2f}", flush=True)
        meanpeak = float(np.mean([report[c]["peak_ratio"] for c in CARRIERS]))
        verdict = (f"WEAKER grading (mean peak {meanpeak:.2f}x < Pythia 1.4x) -> continuum gives gentler transport grading "
                   f"as predicted" if meanpeak < 1.3 else
                   f"grading peak {meanpeak:.2f}x ~ Pythia 1.4x -> NOT weaker (prediction missed; grading not spectrum-set)")
        report["verdict"] = verdict; print(f"\n[G2D2] {verdict}", flush=True)
    except Exception as e:
        report["error"] = str(e); print(f"[G2D2] readout error (series saved): {e}", flush=True)
    json.dump({"report": report, "snap0": SNAP0, "inj_c": INJ_C, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[G2D2] saved gpt2_dye2.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
