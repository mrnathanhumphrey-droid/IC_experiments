"""JOB 2 Stage 2: GPT-2-medium freeze gate + curvature read at the SST-2 task-A min.
(1) K-FAC binding basis on the 24 attn.c_proj (Conv1D; K-FAC operates on activations -> in-space A,
    out-space G regardless of weight storage orientation). Save gpt2 kfac basis.
(2) Curvature read: binding dynamic range (decades of outer(eG,eA)) -> CURVED (>=2 decades) or flat.
(3) Robust lambda_min cert (the -49 phantom lesson; K-FAC is PSD-blind): HVP power-iteration for
    lambda_max + shifted lambda_min, 2 restarts, plateau early-stop, consistency checks.
Eager attention for HVP double-backward. Gates: lambda_min>=0 (PSD min) AND CURVED.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch
torch.set_num_threads(4)

MODEL = "gpt2-medium"; CKPT = "D:/IC_experiments/cf_cascade/gateA/gpt2med_sst2_subsetmin.pt"
KFAC_OUT = "D:/IC_experiments/cf_cascade/gateA/gpt2_kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/gpt2_freeze_curvature.json"
BATCH = 16; SUBSET = 2000; N_KFAC = 16; N_HESS = 4; PI_ITERS = 100; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0)


def sst():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:SUBSET]]


def batch(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def main():
    t0 = time.time(); print("[F] loading gpt2-medium (eager) + min...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    model.load_state_dict(torch.load(CKPT)); model.eval()
    items = sst(); g = torch.Generator().manual_seed(7)
    fixed = [[items[k] for k in torch.randperm(len(items), generator=g)[:BATCH].tolist()] for _ in range(N_HESS)]

    # ---- K-FAC binding basis on attn.c_proj ----
    dense = {n: m for n, m in model.named_modules() if n.endswith("attn.c_proj")}
    A_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}; G_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}
    cnt = {n: 0 for n in dense}; cur = {}; H = []
    for n, m in dense.items():
        def fwd(mod, i, o, nn=n): cur[nn] = i[0].detach()
        def bwd(mod, gi, go, nn=n):
            x = cur[nn].reshape(-1, 1024); gg = go[0].detach().reshape(-1, 1024); k = gg.abs().sum(1) > 0
            if k.any(): A_acc[nn] += x[k].T @ x[k]; G_acc[nn] += gg[k].T @ gg[k]; cnt[nn] += int(k.sum())
        H += [m.register_forward_hook(fwd), m.register_full_backward_hook(bwd)]
    gk = torch.Generator().manual_seed(7)
    for _ in range(N_KFAC):
        inp, lb, am = batch(tok, [items[k] for k in torch.randperm(len(items), generator=gk)[:BATCH].tolist()])
        model.zero_grad(set_to_none=True); model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
    for h in H: h.remove()
    UA = {}; UG = {}; eA = {}; eG = {}; bind_decades = []
    for n in dense:
        A = (A_acc[n] / max(cnt[n], 1)).double(); G = (G_acc[n] / max(cnt[n], 1)).double()
        evA, vA = torch.linalg.eigh(A); evG, vG = torch.linalg.eigh(G)
        eA[n] = evA.clamp(min=0); eG[n] = evG.clamp(min=0); UA[n] = vA.float(); UG[n] = vG.float()
        bind = torch.outer(eG[n], eA[n]).flatten(); bind = bind[bind > 0]
        bind_decades.append(float(torch.log10(bind.max() / bind.min())))
    np.savez(KFAC_OUT, **{f"{n}__UA": UA[n].cpu().numpy() for n in dense}, **{f"{n}__UG": UG[n].cpu().numpy() for n in dense},
             **{f"{n}__eA": eA[n].float().cpu().numpy() for n in dense}, **{f"{n}__eG": eG[n].float().cpu().numpy() for n in dense})
    curv_decades = float(np.median(bind_decades))
    print(f"[F] K-FAC built; curvature binding dynamic range median = {curv_decades:.2f} decades ({time.time()-t0:.0f}s)", flush=True)

    # ---- robust lambda_min cert via HVP power-iteration ----
    params = [p for p in model.parameters() if p.requires_grad]
    def hvp(vec):
        acc = [torch.zeros_like(p) for p in params]
        for fb in fixed:
            inp, lb, am = batch(tok, fb); model.zero_grad(set_to_none=True)
            loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            grads = torch.autograd.grad(loss, params, create_graph=True)
            gv = sum((g * v).sum() for g, v in zip(grads, vec))
            hv = torch.autograd.grad(gv, params, retain_graph=False)
            for a, h in zip(acc, hv): a += h.detach() / len(fixed)
        return acc
    def normalize(v):
        nrm = torch.sqrt(sum((x * x).sum() for x in v)) + 1e-30; return [x / nrm for x in v], float(nrm)
    def rayleigh(v): Hv = hvp(v); return float(sum((a * b).sum() for a, b in zip(Hv, v))), Hv
    def power(shifted_c=None, seed=0):
        torch.manual_seed(seed); v = [torch.randn_like(p) for p in params]; v, _ = normalize(v)
        lam_prev = 0.0
        for it in range(PI_ITERS):
            Hv = hvp(v)
            if shifted_c is not None: Hv = [shifted_c * vi - hi for vi, hi in zip(v, Hv)]  # (cI - H)
            v, _ = normalize(Hv); lam = sum((a * b).sum() for a, b in zip(hvp(v), v)).item()
            # lam = v^T H v under the ORIGINAL H. For the shifted iter, v converged to the
            # min-eigenvector of H, so lam IS lambda_min directly (no further c - lam needed).
            if it > 8 and abs(lam - lam_prev) < 1e-3 * (abs(lam) + 1e-6): break
            lam_prev = lam
        return lam, it + 1
    print("[F] lambda_max power-iter (restart 1)...", flush=True)
    lmax1, i1 = power(seed=0); print(f"   lam_max={lmax1:+.4f} ({i1} it, {time.time()-t0:.0f}s)", flush=True)
    lmax2, i2 = power(seed=1); print(f"   lam_max(restart2)={lmax2:+.4f}", flush=True)
    lmax = max(lmax1, lmax2); c = abs(lmax) * 1.5 + 10.0
    print(f"[F] lambda_min via shift c={c:.1f}...", flush=True)
    lmin, i3 = power(shifted_c=c, seed=2)            # power() returns v^T H v = lambda_min directly
    lmin2, _ = power(shifted_c=c, seed=3)
    print(f"   lam_min={lmin:+.6f} (restart2 {lmin2:+.6f})", flush=True)
    lmax_agree = abs(lmax1 - lmax2) < 0.15 * (abs(lmax) + 1e-6)
    lmin_agree = abs(lmin - lmin2) < 0.15 * (abs(lmax) + 1e-6)
    psd = bool(lmin > -1e-3 * abs(lmax))                      # numerically non-negative
    dyn = float(lmax / max(abs(lmin), 1e-9))
    verdict = (f"PSD MINIMUM (lam_min={lmin:.2e} ~ 0, lam_max={lmax:.2f}, dyn={dyn:.1e}) + {'CURVED' if curv_decades>=2 else 'FLAT'} ({curv_decades:.1f} binding decades)"
               if psd and lmax_agree and lmin_agree else
               f"INCONSISTENT/NON-PSD (lam_min={lmin:.3f}, agree max={lmax_agree} min={lmin_agree}) -> investigate, do NOT bank")
    print(f"\n[F] === FREEZE GATE + CURVATURE: {verdict} ===", flush=True)
    json.dump({"lam_max": lmax, "lam_max_r2": lmax2, "lam_min": lmin, "lam_min_r2": lmin2, "lmax_agree": lmax_agree,
               "lmin_agree": lmin_agree, "psd": psd, "dyn_range": dyn, "curv_binding_decades": curv_decades,
               "CURVED": bool(curv_decades >= 2), "verdict": verdict, "kfac": KFAC_OUT, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[F] saved gpt2_freeze_curvature.json + gpt2_kfac_binding_basis.npz  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
