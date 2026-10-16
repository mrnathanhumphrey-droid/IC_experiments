"""Clauses 3 (GENERATOR/diffusion) + 4 (early-vs-late) — the dynamical half. The aging series was
never saved, so re-fit AG-News from w_A WITH step-snapshots, projecting Delta_w_t onto the SAVED
K-FAC curvature basis (U_A) -> per-curvature-shell damage variance vs task-B step.

PRE-REGISTERED C3 readings (LOCKED before data):
  linear (median beta in [0.8,1.2], clean per-shell R2>0.9, no late-plateau) AND rate~curvature
        -> FREE GRADED DIFFUSION (C3 pass; C5 additive confirmed)
  saturating (2nd-half slope / 1st-half slope < 0.5, or beta<0.8)
        -> BOUNDED RELAXATION (endpoints right, generator bounded not free)
  super-linear (beta>1.2)
        -> TEMPORAL MULTIPLICATIVITY -> reopens C5
C4: early-snapshot PR < late (~1850); early damage center-of-mass at HIGH curvature, spreads to low
    (damage-weighted mean shell-eigval decreases; shell-distribution entropy increases).

CPU thread-capped; GPU training. Reuses kfac_binding_basis.npz (no K-FAC recollection).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch
torch.set_num_threads(4)

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"; KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/checkpoint_diffusion.json"
SERIES_NPZ = "D:/IC_experiments/cf_cascade/gateA/diffusion_series.npz"
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; SQUARE = "attention.dense"; N_SHELLS = 32; dev = "cuda"
SNAP = sorted({10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 750, 900, 1100})
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def agnews_items():
    d = json.load(open(f"{AX}/low_agnews.json")); t = d["prompt_template"]
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
    t0 = time.time(); print("[CKPT] start; loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); B_items = agnews_items()
    names = [n for n, p in model.state_dict().items() if SQUARE in n and tuple(p.shape) == (1024, 1024)]
    params = dict(model.named_parameters())
    wA = {n: params[n].detach().float().clone() for n in names}
    kf = np.load(KFAC_NPZ)
    UA = {n: torch.tensor(kf[f"{n[:-len('.weight')]}__UA"]).double().to(dev) for n in names}
    eigA = np.concatenate([kf[f"{n[:-len('.weight')]}__eA"] for n in names])    # pooled curvature eigvals (fixed)
    order = np.argsort(eigA); shell_id = np.empty(len(eigA), int)
    edges = np.linspace(0, len(eigA), N_SHELLS + 1).astype(int)
    for s in range(N_SHELLS): shell_id[order[edges[s]:edges[s + 1]]] = s
    shell_eig = np.array([eigA[shell_id == s].mean() for s in range(N_SHELLS)])
    print(f"[CKPT] loaded; reused K-FAC basis; {len(names)} matrices, {N_SHELLS} shells ({time.time()-t0:.0f}s)", flush=True)

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
    run = []; step = 0; steps_rec = []; var_rec = []; pr_rec = []; com_rec = []; ent_rec = []
    done = False
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
                sv, pr, com, ent = snapshot(); steps_rec.append(step); var_rec.append(sv)
                pr_rec.append(pr); com_rec.append(com); ent_rec.append(ent)
                print(f"  [CKPT] step {step} train_win {np.mean(run):.4f} PR={pr:.0f} com={com:.3e} ent={ent:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS:
            if step not in steps_rec:                      # always snapshot the final state
                sv, pr, com, ent = snapshot(); steps_rec.append(step); var_rec.append(sv)
                pr_rec.append(pr); com_rec.append(com); ent_rec.append(ent)
                print(f"  [CKPT] FINAL step {step} PR={pr:.0f} com={com:.3e} ent={ent:.3f}", flush=True)
            done = True
    steps = np.array(steps_rec); V = np.array(var_rec); PR = np.array(pr_rec); COM = np.array(com_rec); ENT = np.array(ent_rec)
    np.savez(SERIES_NPZ, steps=steps, per_shell_var=V, PR=PR, com=COM, ent=ENT, shell_eig=shell_eig)

    # ---- C3: per-shell variance vs step ----
    betas = []; r2lin = []; rates = []; half_ratio = []
    ls = np.log(steps)
    for s in range(N_SHELLS):
        y = V[:, s]; m = y > 0
        if m.sum() < 5: betas.append(np.nan); r2lin.append(np.nan); rates.append(np.nan); half_ratio.append(np.nan); continue
        b = np.polyfit(ls[m], np.log(y[m]), 1)[0]; betas.append(b)              # log-log exponent
        A = np.polyfit(steps, y, 1); pred = np.polyval(A, steps); ss = np.sum((y - y.mean()) ** 2)
        r2lin.append(1 - np.sum((y - pred) ** 2) / ss if ss > 0 else 0); rates.append(A[0])
        h = len(steps) // 2
        s1 = np.polyfit(steps[:h], y[:h], 1)[0]; s2 = np.polyfit(steps[h:], y[h:], 1)[0]
        half_ratio.append(s2 / s1 if abs(s1) > 1e-30 else np.nan)
    betas = np.array(betas); r2lin = np.array(r2lin); rates = np.array(rates); half_ratio = np.array(half_ratio)
    from scipy.stats import spearmanr
    rate_eig_rho = float(spearmanr(shell_eig, rates, nan_policy="omit").correlation)
    med_beta = float(np.nanmedian(betas)); med_r2 = float(np.nanmedian(r2lin)); med_half = float(np.nanmedian(half_ratio))

    if med_beta > 1.2:
        c3 = "SUPER-LINEAR -> TEMPORAL MULTIPLICATIVITY, reopens C5"
    elif med_beta < 0.8 or med_half < 0.5:
        c3 = "SATURATING -> BOUNDED RELAXATION (generator bounded, not free diffusion)"
    elif 0.8 <= med_beta <= 1.2 and med_r2 > 0.9 and rate_eig_rho > 0.5:
        c3 = "LINEAR + curvature-graded -> FREE GRADED DIFFUSION (C3 pass, C5 additive confirmed)"
    else:
        c3 = f"AMBIGUOUS (beta={med_beta:.2f} r2lin={med_r2:.2f} rate-eig-rho={rate_eig_rho:.2f})"

    # ---- C4: early vs late ----
    c4 = {"PR_early": float(PR[0]), "PR_late": float(PR[-1]), "com_early": float(COM[0]), "com_late": float(COM[-1]),
          "ent_early": float(ENT[0]), "ent_late": float(ENT[-1]),
          "PR_grows": bool(PR[-1] > PR[0]), "com_drops": bool(COM[-1] < COM[0]), "ent_grows": bool(ENT[-1] > ENT[0])}

    print(f"\n[CKPT] === CLAUSE 3 (generator): {c3} ===", flush=True)
    print(f"   median beta(log-log)={med_beta:.3f}  median R2_linear={med_r2:.3f}  median half-slope-ratio={med_half:.3f}  rate~curvature rho={rate_eig_rho:+.3f}", flush=True)
    print(f"[CKPT] === CLAUSE 4 (early->late): PR {PR[0]:.0f}->{PR[-1]:.0f} ({'spreads' if c4['PR_grows'] else 'NO'})  com {COM[0]:.2e}->{COM[-1]:.2e} ({'drops' if c4['com_drops'] else 'NO'})  ent {ENT[0]:.3f}->{ENT[-1]:.3f} ({'grows' if c4['ent_grows'] else 'NO'}) ===", flush=True)
    json.dump({"steps": steps.tolist(), "med_beta": med_beta, "med_r2_linear": med_r2, "med_half_slope_ratio": med_half,
               "rate_curvature_rho": rate_eig_rho, "per_shell_beta": betas.tolist(), "per_shell_r2lin": r2lin.tolist(),
               "C3_verdict": c3, "C4": c4, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[CKPT] saved checkpoint_diffusion.json + diffusion_series.npz  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
