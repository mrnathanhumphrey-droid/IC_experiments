"""#2 DECISIVE: per-shell MSD (mean squared displacement) generator law, the DISTRIBUTION-ROBUST
observable (variance-of-squared-displacements ~ t^2 even for NORMAL diffusion -> var-beta confounded
by the damage tail; MSD = mean(dmg) per shell ~ t^beta_MSD: beta_MSD=1 normal/additive, >1 super/
multiplicative, <1 sub). DENSE clean-regime snapshots. Runs GPT-2/AG (question) + Pythia/AG (additive
reference, calibrate beta_MSD~1). Saves per-shell MSD + var + PR/com/ent at every snapshot.

argv[1] = substrate key (pythia_agnews | gpt2_agnews).
PRE-COMMITTED (Wilson): beta_MSD>1.2 + corroboration (PR climbing, var 2nd-deriv>0, com/ent expanding)
-> temporal-mult REAL (path-dependent reversibility); beta_MSD~1 or corroboration fails -> additive
(uniform reversibility #1); mixed -> artifact-leaning, flag.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch
torch.set_num_threads(4)
G = "D:/IC_experiments/cf_cascade/gateA"
CFG = {
  "pythia_agnews": dict(model="EleutherAI/pythia-410m", ckpt=f"{G}/pythia410m_sst2_subsetmin.pt",
                        kfac=f"{G}/kfac_binding_basis.npz", square="attention.dense", conv1d=False, taskB="low_agnews"),
  "gpt2_agnews":   dict(model="gpt2-medium", ckpt=f"{G}/gpt2med_sst2_subsetmin.pt",
                        kfac=f"{G}/gpt2_kfac_binding_basis.npz", square="attn.c_proj", conv1d=True, taskB="low_agnews"),
  "pythia_yelp":   dict(model="EleutherAI/pythia-410m", ckpt=f"{G}/pythia410m_sst2_subsetmin.pt",
                        kfac=f"{G}/kfac_binding_basis.npz", square="attention.dense", conv1d=False, taskB="high_yelp"),
}
KEY = sys.argv[1] if len(sys.argv) > 1 else "gpt2_agnews"; C = CFG[KEY]
AX = f"{G}/overlap_axis"; OUT = f"{G}/msd_{KEY}.json"; SERIES = f"{G}/msd_series_{KEY}.npz"
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_SHELLS = 32; dev = "cuda"
SNAP = sorted(set([10, 25, 50, 75]) | set(range(100, 760, 25)))   # dense clean regime
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def items_taskb():
    d = json.load(open(f"{AX}/{C['taskB']}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def batch(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def main():
    t0 = time.time(); print(f"[MSD] substrate={KEY}; loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C["model"]); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(C["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(C["ckpt"])); B_items = items_taskb()
    names = [n for n, p in model.state_dict().items() if C["square"] in n and n.endswith("weight") and tuple(p.shape) == (1024, 1024)]
    params = dict(model.named_parameters()); wA = {n: params[n].detach().float().clone() for n in names}
    kf = np.load(C["kfac"]); UA = {n: torch.tensor(kf[f"{n[:-len('.weight')]}__UA"]).double().to(dev) for n in names}
    eigA = np.concatenate([kf[f"{n[:-len('.weight')]}__eA"] for n in names])
    order = np.argsort(eigA); shell_id = np.empty(len(eigA), int); edges = np.linspace(0, len(eigA), N_SHELLS + 1).astype(int)
    for s in range(N_SHELLS): shell_id[order[edges[s]:edges[s + 1]]] = s
    shell_eig = np.array([eigA[shell_id == s].mean() for s in range(N_SHELLS)])
    print(f"[MSD] {len(names)} matrices, conv1d={C['conv1d']} ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def snap():
        dmg = []
        for n in names:
            dw = (params[n].detach().float() - wA[n]).double()
            proj = (UA[n].T @ dw) if C["conv1d"] else (dw @ UA[n])
            dmg.append((proj * proj).sum(1 if C["conv1d"] else 0).cpu().numpy())
        dmg = np.concatenate(dmg)
        msd = np.array([dmg[shell_id == s].mean() for s in range(N_SHELLS)])     # MEAN sq displacement
        var = np.array([dmg[shell_id == s].var() for s in range(N_SHELLS)])
        tot = dmg.sum(); PR = float((tot ** 2) / np.sum(dmg ** 2)) if tot > 0 else 0.0
        w = np.array([dmg[shell_id == s].sum() for s in range(N_SHELLS)]); w = w / max(w.sum(), 1e-30)
        return msd, var, PR, float(np.sum(w * shell_eig)), float(-np.sum(w[w > 0] * np.log(w[w > 0])))

    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0; done = False; rec = {k: [] for k in ("steps", "msd", "var", "PR", "com", "ent")}
    while step < MAX_STEPS and not done:
        ordr = torch.randperm(len(B_items), generator=g).tolist()
        for i in range(0, len(ordr) - BATCH, BATCH):
            inp, lb, am = batch(tok, [B_items[k] for k in ordr[i:i + BATCH]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step in SNAP:
                m, v, pr, com, ent = snap()
                for k, val in zip(("steps", "msd", "var", "PR", "com", "ent"), (step, m, v, pr, com, ent)): rec[k].append(val)
                print(f"  [MSD] step {step} tw {np.mean(run):.4f} MSD_med {np.median(m):.3e} PR {pr:.0f} com {com:.2e} ent {ent:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    steps = np.array(rec["steps"]); MSD = np.array(rec["msd"]); V = np.array(rec["var"]); PR = np.array(rec["PR"]); COM = np.array(rec["com"]); ENT = np.array(rec["ent"])
    np.savez(SERIES, steps=steps, per_shell_msd=MSD, per_shell_var=V, PR=PR, com=COM, ent=ENT, shell_eig=shell_eig)

    # clean-regime (step>=100) beta_MSD per shell + corroboration
    m = steps >= 100; cs = steps[m]; ls = np.log(cs)
    bmsd = [np.polyfit(ls, np.log(MSD[m, j]), 1)[0] for j in range(N_SHELLS) if (MSD[m, j] > 0).all()]
    bvar = [np.polyfit(ls, np.log(V[m, j]), 1)[0] for j in range(N_SHELLS) if (V[m, j] > 0).all()]
    med_bmsd = float(np.median(bmsd)); med_bvar = float(np.median(bvar))
    mmsd = np.median(MSD[m], axis=1); q = np.polyfit(cs, mmsd, 2)[0]
    lin = np.polyval(np.polyfit(cs, mmsd, 1), cs); quad = np.polyval(np.polyfit(cs, mmsd, 2), cs); ss = np.sum((mmsd - mmsd.mean()) ** 2)
    msd_convex = bool(q > 0 and (1 - np.sum((mmsd - quad) ** 2) / ss) - (1 - np.sum((mmsd - lin) ** 2) / ss) > 0.01)
    pr_climb = bool(PR[m][-1] / PR[m].max() > 0.98); pr_accel = bool(np.polyfit(cs, PR[m], 2)[0] > 0)
    ent_expand = bool(ENT[m][-1] > ENT[m][0] + 0.02 and np.polyfit(cs, ENT[m], 1)[0] > 0)
    corrob = bool(msd_convex and pr_climb and ent_expand)
    if med_bmsd > 1.2 and corrob: verdict = "TEMPORAL-MULTIPLICATIVITY REAL (beta_MSD>1.2 + corroboration) -> path-dependent reversibility"
    elif med_bmsd <= 1.2: verdict = "ADDITIVE/NORMAL (beta_MSD~1) -> uniform reversibility #1"
    else: verdict = "MIXED/artifact-leaning (beta_MSD>1.2 but corroboration fails) -> do NOT bank multiplicativity"
    print(f"\n[MSD] === {KEY}: beta_MSD(clean)={med_bmsd:.3f}  [beta_var(clean)={med_bvar:.3f}] ===", flush=True)
    print(f"   corrob: MSD-convex={msd_convex} PR-climb={pr_climb} PR-accel={pr_accel} ent-expand={ent_expand} -> corroborated={corrob}", flush=True)
    print(f"   per-shell beta_MSD: {np.round(bmsd,2).tolist()}", flush=True)
    print(f"   VERDICT: {verdict}", flush=True)
    json.dump({"substrate": KEY, "beta_MSD_clean": med_bmsd, "beta_var_clean": med_bvar, "msd_convex": msd_convex,
               "PR_climb": pr_climb, "PR_accel": pr_accel, "ent_expand": ent_expand, "corroborated": corrob,
               "per_shell_beta_MSD": [float(x) for x in bmsd], "verdict": verdict, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[MSD] saved msd_{KEY}.json + series  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
