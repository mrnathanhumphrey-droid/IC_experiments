"""Reversibility test #1 (the encapsulation fork). UNIFORM ADDITIVE (routing settled by #2/MSD).
2D map per curvature shell S: (importance = task-A empirical Fisher in S) x (recovery = task-A acc
restored by zeroing task-B displacement in S). The 2D guards the trap: high flat-bulk recovery is
NOT "reversible" unless S also mattered to task-A.

4 cells (pre-committed): hi-imp+recovers=reversible-load-bearing(prize); hi-imp+no-recover=permanent
-load-bearing(protect-during-training); lo-imp+recovers=harmless(the (b) confound); lo-imp+no-recover
=permanent-harmless. PREDICTION: stiff-seed=hi-imp AND permanent; flat-bulk=lo-imp AND recovers.
Corroboration (5th time): recovery = task-A ACC restored, NOT norm removed; report both.

argv[1]=substrate (pythia_agnews|pythia_yelp|gpt2_agnews). Rides frozen curvature basis + saved Delta_w.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
G = "D:/IC_experiments/cf_cascade/gateA"
CFG = {
  "pythia_agnews": dict(model="EleutherAI/pythia-410m", ckpt=f"{G}/pythia410m_sst2_subsetmin.pt",
                        kfac=f"{G}/kfac_binding_basis.npz", dw=f"{G}/deltaw_agnews.npz", square="attention.dense", conv1d=False),
  "pythia_yelp":   dict(model="EleutherAI/pythia-410m", ckpt=f"{G}/pythia410m_sst2_subsetmin.pt",
                        kfac=f"{G}/kfac_binding_basis.npz", dw=f"{G}/deltaw_yelp.npz", square="attention.dense", conv1d=False),
  "gpt2_agnews":   dict(model="gpt2-medium", ckpt=f"{G}/gpt2med_sst2_subsetmin.pt",
                        kfac=f"{G}/gpt2_kfac_binding_basis.npz", dw=f"{G}/deltaw_gpt2_agnews.npz", square="attn.c_proj", conv1d=True),
}
KEY = sys.argv[1] if len(sys.argv) > 1 else "pythia_agnews"; C = CFG[KEY]
OUT = f"{G}/reversibility_{KEY}.json"; N_SHELLS = 32; N_FISHER = 200; VAL_N = 300; dev = "cuda"
LAB = {0: " negative", 1: " positive"}; torch.manual_seed(0); np.random.seed(0)


def sst():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    va = [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"] if r["label"] in (0, 1)][:VAL_N]
    tr = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)]
    tr = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:N_FISHER]]
    return tr, va


def batch(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


@torch.no_grad()
def acc(model, tok, va):
    Lc = {}
    for cl in (0, 1):
        per = []
        items = [(s, cl) for s, _ in va]
        for i in range(0, len(items), 64):
            inp, lb, am = batch(tok, items[i:i + 64])
            lg = model(input_ids=inp, attention_mask=am).logits
            sl = lg[:, :-1].reshape(-1, lg.shape[-1]); st = lb[:, 1:].reshape(-1)
            ll = F.cross_entropy(sl, st, ignore_index=-100, reduction="none").reshape(lb.shape[0], -1)
            mask = (lb[:, 1:] != -100)
            per.extend((ll.sum(1) / mask.sum(1).clamp(min=1)).tolist())
        Lc[cl] = per
    pred = [0 if Lc[0][i] < Lc[1][i] else 1 for i in range(len(va))]
    return float(np.mean([int(pred[i] == va[i][1]) for i in range(len(va))]))


def main():
    t0 = time.time(); print(f"[REV] {KEY} loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C["model"]); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(C["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(C["ckpt"])); model.eval()
    tr, va = sst()
    names = [n for n, p in model.state_dict().items() if C["square"] in n and n.endswith("weight") and tuple(p.shape) == (1024, 1024)]
    params = dict(model.named_parameters())
    wA = {n: params[n].detach().float().clone() for n in names}
    dwz = np.load(C["dw"]); dW = {n: torch.tensor(dwz[n]).float().to(dev) for n in names}
    kf = np.load(C["kfac"]); UA = {n: torch.tensor(kf[f"{n[:-len('.weight')]}__UA"]).float().to(dev) for n in names}
    eigA = np.concatenate([kf[f"{n[:-len('.weight')]}__eA"] for n in names])
    order = np.argsort(eigA); shell_id = np.empty(len(eigA), int); edges = np.linspace(0, len(eigA), N_SHELLS + 1).astype(int)
    for s in range(N_SHELLS): shell_id[order[edges[s]:edges[s + 1]]] = s
    shell_eig = np.array([eigA[shell_id == s].mean() for s in range(N_SHELLS)])
    # per-matrix slice of the global shell_id (input-direction index within each matrix)
    sizes = [UA[n].shape[1] for n in names]; offs = np.cumsum([0] + sizes)
    sh_by_mat = {n: shell_id[offs[i]:offs[i + 1]] for i, n in enumerate(names)}

    acc_A = acc(model, tok, va)
    # projection coords C_n = displacement in U_A basis (input axis)
    Ccoord = {}
    for n in names:
        d = dW[n].double()
        Ccoord[n] = (UA[n].double().T @ d) if C["conv1d"] else (d @ UA[n].double())   # conv1d:(in_comp,out) else:(out,in_comp)
    # w_B (full displacement) acc
    for n in names: params[n].data = (wA[n] + dW[n]).data
    acc_B = acc(model, tok, va); forgetting = acc_A - acc_B
    print(f"[REV] task-A acc_A={acc_A:.3f} acc_B(forgotten)={acc_B:.3f} forgetting={forgetting:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- importance: task-A empirical Fisher per shell (per-example grads at w_A) ----
    for n in names: params[n].data = wA[n].data
    fisher_dir = {n: torch.zeros(UA[n].shape[1], device=dev) for n in names}
    for s, l in tr:
        inp, lb, am = batch(tok, [(s, l)]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names:
            g = params[n].grad.detach().double()
            gp = (UA[n].double().T @ g) if C["conv1d"] else (g @ UA[n].double())     # same orient as Ccoord
            fisher_dir[n] += (gp * gp).sum(1 if C["conv1d"] else 0)
    imp_shell = np.zeros(N_SHELLS)
    for n in names:
        fd = fisher_dir[n].cpu().numpy(); sm = sh_by_mat[n]
        for s in range(N_SHELLS): imp_shell[s] += fd[sm == s].sum()
    print(f"[REV] Fisher computed ({time.time()-t0:.0f}s)", flush=True)

    # ---- recovery per shell: zero shell-S displacement, eval task-A acc ----
    rec_acc = np.zeros(N_SHELLS); norm_removed = np.zeros(N_SHELLS)
    for s in range(N_SHELLS):
        for n in names:
            sm = sh_by_mat[n]; mask = torch.tensor(sm == s, device=dev)
            Cs = Ccoord[n].clone()
            if C["conv1d"]: Cs[mask, :] = 0; dkeep = (UA[n].double() @ Cs)            # (in,out)
            else: Cs[:, mask] = 0; dkeep = (Cs @ UA[n].double().T)                    # (out,in)
            removed = (dW[n].double() - dkeep)
            norm_removed[s] += float((removed * removed).sum().item())
            params[n].data = (wA[n].double() + dkeep).float().data
        rec_acc[s] = acc(model, tok, va)
    for n in names: params[n].data = wA[n].data
    recovery = (rec_acc - acc_B) / max(acc_A - acc_B, 1e-9)                            # fraction of forgetting recovered
    print(f"[REV] recovery evals done ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2D cell map (median splits on importance and recovery) ----
    imp_hi = imp_shell >= np.median(imp_shell); rec_hi = recovery >= 0.5               # recovers if >=50% of forgetting back
    cells = {"hi_imp_recovers": [], "hi_imp_permanent": [], "lo_imp_recovers": [], "lo_imp_permanent": []}
    for s in range(N_SHELLS):
        key = ("hi_imp" if imp_hi[s] else "lo_imp") + ("_recovers" if rec_hi[s] else ("_permanent"))
        cells[key].append(s)
    # are stiff shells (high eig, high index) hi-imp + permanent? flat (low) lo-imp + recovers?
    stiff = list(range(N_SHELLS - 8, N_SHELLS)); flat = list(range(8))
    stiff_imp_hi = float(imp_hi[stiff].mean()); stiff_recovers = float(rec_hi[stiff].mean())
    flat_imp_hi = float(imp_hi[flat].mean()); flat_recovers = float(rec_hi[flat].mean())
    pred_holds = bool(stiff_imp_hi > 0.5 and stiff_recovers < 0.5 and flat_imp_hi < 0.5 and flat_recovers > 0.5)

    print(f"\n[REV] === {KEY}: 2D REVERSIBILITY MAP (forgetting {forgetting:+.3f}) ===", flush=True)
    print(f"  cell occupancy (shell counts): hi-imp+recovers={len(cells['hi_imp_recovers'])} hi-imp+PERMANENT={len(cells['hi_imp_permanent'])} lo-imp+recovers={len(cells['lo_imp_recovers'])} lo-imp+perm={len(cells['lo_imp_permanent'])}", flush=True)
    print(f"  STIFF(8 hi-eig): imp-hi frac={stiff_imp_hi:.2f} recovers frac={stiff_recovers:.2f}  | FLAT(8 lo-eig): imp-hi={flat_imp_hi:.2f} recovers={flat_recovers:.2f}", flush=True)
    print(f"  recovery by shell: {np.round(recovery,2).tolist()}", flush=True)
    print(f"  importance(norm) by shell: {np.round(imp_shell/imp_shell.max(),2).tolist()}", flush=True)
    print(f"  PREDICTION (stiff=hi-imp+permanent, flat=lo-imp+recovers) HOLDS = {pred_holds}", flush=True)
    json.dump({"substrate": KEY, "acc_A": acc_A, "acc_B": acc_B, "forgetting": forgetting,
               "importance_shell": imp_shell.tolist(), "recovery_shell": recovery.tolist(),
               "rec_acc_shell": rec_acc.tolist(), "norm_removed_shell": norm_removed.tolist(),
               "shell_eig": shell_eig.tolist(), "cells": cells, "stiff_imp_hi": stiff_imp_hi,
               "stiff_recovers": stiff_recovers, "flat_imp_hi": flat_imp_hi, "flat_recovers": flat_recovers,
               "prediction_holds": pred_holds, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[REV] saved reversibility_{KEY}.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
