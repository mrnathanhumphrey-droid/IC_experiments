"""REPLICATION on task-B = Yelp (high overlap, same-skill sentiment) — far end of the overlap axis
from AG-News (low overlap, different-skill topic). Reuses frozen task-A SST-2 min + K-FAC curvature
basis; swaps task-B. Matched step-snapshots to checkpoint_diffusion.json for clean comparison.

Saves diffusion_series_yelp.npz (per-shell variance vs step) + deltaw_yelp.npz (final Delta_w, for
C1/C2). Computes C3 (linear-in-step, rate~curvature) + C4 (early->late) here; C1/C2/tail-index +
the AG-News-vs-Yelp comparison run in yelp_clauses.py (CPU).

SOFT-SPOT WATCH (pre-registered): AG-News had beta grading with curvature (sloppy~0.67, stiff~1.1,
median 0.88). Report the FULL per-shell beta array + Spearman(shell_eig, beta). Fork:
  beta-grades-with-curvature reproduces -> graded ANOMALOUS diffusion is the phenomenon (sharper
    than free diffusion: sub-diffusive in flat dirs, ~ballistic in stiff)
  clean beta~1 on Yelp -> AG-News softness was substrate-specific
  new soft-spot elsewhere -> diagnose, don't paper
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch
torch.set_num_threads(4)

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"; KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/yelp_diffusion.json"
SERIES_NPZ = "D:/IC_experiments/cf_cascade/gateA/diffusion_series_yelp.npz"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_yelp.npz"
TASKB = "high_yelp"; BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; SQUARE = "attention.dense"; N_SHELLS = 32; dev = "cuda"
SNAP = sorted({10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 750, 900, 1100})   # matched to AG-News
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def taskb_items():
    d = json.load(open(f"{AX}/{TASKB}.json")); t = d["prompt_template"]
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
    t0 = time.time(); print(f"[YELP] start; task-B={TASKB}; loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); B_items = taskb_items()
    names = [n for n, p in model.state_dict().items() if SQUARE in n and tuple(p.shape) == (1024, 1024)]
    params = dict(model.named_parameters())
    wA = {n: params[n].detach().float().clone() for n in names}
    kf = np.load(KFAC_NPZ)
    UA = {n: torch.tensor(kf[f"{n[:-len('.weight')]}__UA"]).double().to(dev) for n in names}
    eigA = np.concatenate([kf[f"{n[:-len('.weight')]}__eA"] for n in names])
    order = np.argsort(eigA); shell_id = np.empty(len(eigA), int)
    edges = np.linspace(0, len(eigA), N_SHELLS + 1).astype(int)
    for s in range(N_SHELLS): shell_id[order[edges[s]:edges[s + 1]]] = s
    shell_eig = np.array([eigA[shell_id == s].mean() for s in range(N_SHELLS)])
    print(f"[YELP] loaded; reused K-FAC basis; {len(names)} matrices, {N_SHELLS} shells, {len(B_items)} Yelp items ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def snapshot():
        dmg = []
        for n in names:
            dw = (params[n].detach().float() - wA[n]).double()
            proj = dw @ UA[n]; dmg.append((proj * proj).sum(0).cpu().numpy())
        dmg = np.concatenate(dmg)
        sh_var = np.array([dmg[shell_id == s].var() for s in range(N_SHELLS)])
        sh_tot = np.array([dmg[shell_id == s].sum() for s in range(N_SHELLS)])
        tot = dmg.sum(); PR = float((tot ** 2) / np.sum(dmg ** 2)) if tot > 0 else 0.0
        w = sh_tot / max(sh_tot.sum(), 1e-30)
        com = float(np.sum(w * shell_eig)); ent = float(-np.sum(w[w > 0] * np.log(w[w > 0])))
        return sh_var, PR, com, ent

    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0; steps_rec = []; var_rec = []; pr_rec = []; com_rec = []; ent_rec = []; done = False
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
                sv, pr, com, ent = snapshot(); steps_rec.append(step); var_rec.append(sv); pr_rec.append(pr); com_rec.append(com); ent_rec.append(ent)
                print(f"  [YELP] step {step} train_win {np.mean(run):.4f} PR={pr:.0f} com={com:.3e} ent={ent:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS:
            if step not in steps_rec:
                sv, pr, com, ent = snapshot(); steps_rec.append(step); var_rec.append(sv); pr_rec.append(pr); com_rec.append(com); ent_rec.append(ent)
                print(f"  [YELP] FINAL step {step} PR={pr:.0f} com={com:.3e} ent={ent:.3f}", flush=True)
            done = True
    steps = np.array(steps_rec); V = np.array(var_rec); PR = np.array(pr_rec); COM = np.array(com_rec); ENT = np.array(ent_rec)
    np.savez(SERIES_NPZ, steps=steps, per_shell_var=V, PR=PR, com=COM, ent=ENT, shell_eig=shell_eig)
    # save final Delta_w for C1/C2
    sd = model.state_dict(); np.savez(DW_NPZ, **{n: (sd[n].detach().float() - wA[n]).cpu().numpy() for n in names})

    # ---- C3 + soft-spot beta gradient ----
    betas = []; r2lin = []; rates = []; half = []; ls = np.log(steps)
    for s in range(N_SHELLS):
        y = V[:, s]; m = y > 0
        if m.sum() < 5: betas.append(np.nan); r2lin.append(np.nan); rates.append(np.nan); half.append(np.nan); continue
        betas.append(np.polyfit(ls[m], np.log(y[m]), 1)[0])
        A = np.polyfit(steps, y, 1); pred = np.polyval(A, steps); ss = np.sum((y - y.mean()) ** 2)
        r2lin.append(1 - np.sum((y - pred) ** 2) / ss if ss > 0 else 0); rates.append(A[0])
        h = len(steps) // 2; s1 = np.polyfit(steps[:h], y[:h], 1)[0]; s2 = np.polyfit(steps[h:], y[h:], 1)[0]
        half.append(s2 / s1 if abs(s1) > 1e-30 else np.nan)
    betas = np.array(betas); r2lin = np.array(r2lin); rates = np.array(rates); half = np.array(half)
    from scipy.stats import spearmanr
    rate_eig = float(spearmanr(shell_eig, rates, nan_policy="omit").correlation)
    beta_eig = float(spearmanr(shell_eig, betas, nan_policy="omit").correlation)   # soft-spot: does beta grade with curvature?
    med_beta = float(np.nanmedian(betas)); med_r2 = float(np.nanmedian(r2lin)); med_half = float(np.nanmedian(half))
    sloppy_beta = float(np.nanmedian(betas[:8])); stiff_beta = float(np.nanmedian(betas[-8:]))   # shells ordered ascending eigval

    if med_beta > 1.2: c3 = "SUPER-LINEAR -> temporal multiplicativity (reopens C5)"
    elif med_beta < 0.8 or med_half < 0.5: c3 = "SATURATING -> bounded relaxation"
    elif 0.8 <= med_beta <= 1.2 and med_r2 > 0.9 and rate_eig > 0.5: c3 = "LINEAR + curvature-graded -> FREE GRADED DIFFUSION"
    else: c3 = f"AMBIGUOUS (beta={med_beta:.2f} r2lin={med_r2:.2f} rate-eig={rate_eig:.2f})"
    c4 = {"PR_early": float(PR[0]), "PR_late": float(PR[-1]), "com_early": float(COM[0]), "com_late": float(COM[-1]),
          "ent_early": float(ENT[0]), "ent_late": float(ENT[-1]), "PR_grows": bool(PR[-1] > PR[0]),
          "com_drops": bool(COM[-1] < COM[0]), "ent_grows": bool(ENT[-1] > ENT[0])}

    print(f"\n[YELP] === C3: {c3} ===", flush=True)
    print(f"   median beta={med_beta:.3f}  R2_lin={med_r2:.3f}  half-ratio={med_half:.3f}  rate~curv rho={rate_eig:+.3f}", flush=True)
    print(f"   SOFT-SPOT: beta~curvature rho={beta_eig:+.3f}  sloppy(8 flat shells) beta={sloppy_beta:.3f}  stiff(8) beta={stiff_beta:.3f}", flush=True)
    print(f"   per-shell beta: {np.round(betas,2).tolist()}", flush=True)
    print(f"[YELP] === C4: PR {PR[0]:.0f}->{PR[-1]:.0f}  com {COM[0]:.2e}->{COM[-1]:.2e}  ent {ENT[0]:.3f}->{ENT[-1]:.3f} ===", flush=True)
    json.dump({"task_B": TASKB, "steps": steps.tolist(), "med_beta": med_beta, "med_r2_linear": med_r2,
               "med_half_slope_ratio": med_half, "rate_curvature_rho": rate_eig, "beta_curvature_rho": beta_eig,
               "sloppy_beta": sloppy_beta, "stiff_beta": stiff_beta, "per_shell_beta": betas.tolist(),
               "per_shell_r2lin": r2lin.tolist(), "C3_verdict": c3, "C4": c4, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[YELP] saved yelp_diffusion.json + diffusion_series_yelp.npz + deltaw_yelp.npz  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
