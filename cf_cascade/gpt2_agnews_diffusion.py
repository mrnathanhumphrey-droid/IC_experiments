"""JOB 2 Stage 3: GPT-2-medium task-B = AG-News diffusion battery (C3/C4 + per-shell beta).
Mirrors checkpoint_diffusion.py but on GPT-2: square matrices = attn.c_proj (Conv1D, weight (in,out)).
Curvature damage with the Conv1D orientation: U_A is in-space (K-FAC on input activations, storage-
orientation-independent); damage_j = ((U_A^T @ dw_stored)^2).sum(over out). Matched snapshots to
the anchor. AG-News (same task as anchor) -> PRE-REGISTERED: expect FREE generator + alpha~2.6 if
the generator is TASK-controlled; BOUNDED/heavy -> generator depends on NET too (new finding).
Judge the generator on PR/com/entropy plateau, NOT borderline median-beta/half-ratio (false-PASS trap).
Saves diffusion_series_gpt2_agnews.npz + deltaw_gpt2_agnews.npz (for C1/C2).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch
torch.set_num_threads(4)

MODEL = "gpt2-medium"; CKPT = "D:/IC_experiments/cf_cascade/gateA/gpt2med_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"; KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/gpt2_kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/gpt2_agnews_diffusion.json"
SERIES_NPZ = "D:/IC_experiments/cf_cascade/gateA/diffusion_series_gpt2_agnews.npz"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_gpt2_agnews.npz"
TASKB = "low_agnews"; BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; SQUARE = "attn.c_proj"; N_SHELLS = 32; dev = "cuda"
SNAP = sorted({10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 750, 900, 1100})
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
    t0 = time.time(); print(f"[G2D] start; task-B={TASKB}; loading gpt2-medium...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); B_items = taskb_items()
    names = [n for n, p in model.state_dict().items() if SQUARE in n and n.endswith("weight") and tuple(p.shape) == (1024, 1024)]
    params = dict(model.named_parameters()); wA = {n: params[n].detach().float().clone() for n in names}
    kf = np.load(KFAC_NPZ)
    UA = {n: torch.tensor(kf[f"{n[:-len('.weight')]}__UA"]).double().to(dev) for n in names}
    eigA = np.concatenate([kf[f"{n[:-len('.weight')]}__eA"] for n in names])
    order = np.argsort(eigA); shell_id = np.empty(len(eigA), int); edges = np.linspace(0, len(eigA), N_SHELLS + 1).astype(int)
    for s in range(N_SHELLS): shell_id[order[edges[s]:edges[s + 1]]] = s
    shell_eig = np.array([eigA[shell_id == s].mean() for s in range(N_SHELLS)])
    print(f"[G2D] loaded; {len(names)} c_proj matrices, {N_SHELLS} shells, {len(B_items)} AG-News items ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def snapshot():
        dmg = []
        for n in names:
            dw = (params[n].detach().float() - wA[n]).double()      # Conv1D (in,out)
            proj = UA[n].T @ dw                                      # (in_comp, out)
            dmg.append((proj * proj).sum(1).cpu().numpy())          # L2 over out -> per in-component
        dmg = np.concatenate(dmg)
        sh_var = np.array([dmg[shell_id == s].var() for s in range(N_SHELLS)])
        sh_tot = np.array([dmg[shell_id == s].sum() for s in range(N_SHELLS)])
        tot = dmg.sum(); PR = float((tot ** 2) / np.sum(dmg ** 2)) if tot > 0 else 0.0
        w = sh_tot / max(sh_tot.sum(), 1e-30); com = float(np.sum(w * shell_eig)); ent = float(-np.sum(w[w > 0] * np.log(w[w > 0])))
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
                print(f"  [G2D] step {step} train_win {np.mean(run):.4f} PR={pr:.0f} com={com:.3e} ent={ent:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS:
            if step not in steps_rec:
                sv, pr, com, ent = snapshot(); steps_rec.append(step); var_rec.append(sv); pr_rec.append(pr); com_rec.append(com); ent_rec.append(ent)
                print(f"  [G2D] FINAL step {step} PR={pr:.0f} com={com:.3e} ent={ent:.3f}", flush=True)
            done = True
    steps = np.array(steps_rec); V = np.array(var_rec); PR = np.array(pr_rec); COM = np.array(com_rec); ENT = np.array(ent_rec)
    np.savez(SERIES_NPZ, steps=steps, per_shell_var=V, PR=PR, com=COM, ent=ENT, shell_eig=shell_eig)
    sd = model.state_dict(); np.savez(DW_NPZ, **{n: (sd[n].detach().float() - wA[n]).cpu().numpy() for n in names})

    betas = []; r2lin = []; rates = []; half = []; ls = np.log(steps)
    for s in range(N_SHELLS):
        y = V[:, s]; m = y > 0
        if m.sum() < 5: betas.append(np.nan); r2lin.append(np.nan); rates.append(np.nan); half.append(np.nan); continue
        betas.append(np.polyfit(ls[m], np.log(y[m]), 1)[0]); A = np.polyfit(steps, y, 1); pred = np.polyval(A, steps)
        ss = np.sum((y - y.mean()) ** 2); r2lin.append(1 - np.sum((y - pred) ** 2) / ss if ss > 0 else 0); rates.append(A[0])
        h = len(steps) // 2; s1 = np.polyfit(steps[:h], y[:h], 1)[0]; s2 = np.polyfit(steps[h:], y[h:], 1)[0]
        half.append(s2 / s1 if abs(s1) > 1e-30 else np.nan)
    betas = np.array(betas); r2lin = np.array(r2lin); rates = np.array(rates); half = np.array(half)
    from scipy.stats import spearmanr
    rate_eig = float(spearmanr(shell_eig, rates, nan_policy="omit").correlation)
    beta_eig = float(spearmanr(shell_eig, betas, nan_policy="omit").correlation)
    med_beta = float(np.nanmedian(betas)); med_r2 = float(np.nanmedian(r2lin)); med_half = float(np.nanmedian(half))
    pr_lastmax = float(PR[-1] / PR.max())                              # <1 => plateaued/declined (BOUNDED)
    generator = "FREE (PR climbing)" if pr_lastmax > 0.98 and med_half > 0.7 else "BOUNDED/SATURATING (PR plateau)" if pr_lastmax < 0.97 or med_half < 0.6 else "AMBIGUOUS"
    print(f"\n[G2D] === C3 (judge on plateau, not scalars): generator={generator} ===", flush=True)
    print(f"   median beta={med_beta:.3f} R2_lin={med_r2:.3f} half-ratio={med_half:.3f} rate~curv rho={rate_eig:+.3f}", flush=True)
    print(f"   SOFT-SPOT beta~curv rho={beta_eig:+.3f}  sloppy(8)={np.nanmedian(betas[:8]):.3f} stiff(8)={np.nanmedian(betas[-8:]):.3f}", flush=True)
    print(f"   per-shell beta: {np.round(betas,2).tolist()}", flush=True)
    print(f"[G2D] === C4: PR {PR[0]:.0f}->{PR[-1]:.0f} (last/max={pr_lastmax:.2f})  com {COM[0]:.2e}->{COM[-1]:.2e}  ent {ENT[0]:.3f}->{ENT[-1]:.3f} ===", flush=True)
    json.dump({"task_B": TASKB, "net": "gpt2-medium", "steps": steps.tolist(), "med_beta": med_beta, "med_r2_linear": med_r2,
               "med_half_slope_ratio": med_half, "rate_curvature_rho": rate_eig, "beta_curvature_rho": beta_eig,
               "PR_last_over_max": pr_lastmax, "generator": generator, "per_shell_beta": betas.tolist(),
               "C4": {"PR_early": float(PR[0]), "PR_late": float(PR[-1]), "com_early": float(COM[0]), "com_late": float(COM[-1]),
                      "ent_early": float(ENT[0]), "ent_late": float(ENT[-1])}, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[G2D] saved gpt2_agnews_diffusion.json + series + deltaw  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
