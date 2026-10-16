"""Reversibility test #1 — CORRECTED. The v1 bug: w_B was scoped to attention.dense-only, so it
barely forgot (acc ~unchanged) and the recovery denominator was ~0 (garbage). FIX: re-train task-B
FULL model (all params), measure REAL forgetting, then remove attention.dense displacement from the
FULL forgotten model.

LOAD-BEARING PRE-CHECK (now the load-bearing result): restore ALL attention.dense in full w_B ->
how much forgetting recovers? recovers-most => attention.dense damage IS the forgetting (per-shell
map meaningful); recovers-little => forgetting lives elsewhere, the characterized diffusion subspace
is task-A-NEUTRAL (major scope caveat).

Then 2D per curvature shell: importance (task-A Fisher at w_A) x recovery (task-A acc restored by
removing shell-S displacement from full w_B). Corroboration (5th): recovery=acc restored, report
norm-removed alongside. argv[1]=substrate.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
G = "D:/IC_experiments/cf_cascade/gateA"
CFG = {
  "pythia_agnews": dict(model="EleutherAI/pythia-410m", ckpt=f"{G}/pythia410m_sst2_subsetmin.pt",
                        kfac=f"{G}/kfac_binding_basis.npz", square="attention.dense", conv1d=False, taskB="low_agnews"),
  "pythia_yelp":   dict(model="EleutherAI/pythia-410m", ckpt=f"{G}/pythia410m_sst2_subsetmin.pt",
                        kfac=f"{G}/kfac_binding_basis.npz", square="attention.dense", conv1d=False, taskB="high_yelp"),
  "gpt2_agnews":   dict(model="gpt2-medium", ckpt=f"{G}/gpt2med_sst2_subsetmin.pt",
                        kfac=f"{G}/gpt2_kfac_binding_basis.npz", square="attn.c_proj", conv1d=True, taskB="low_agnews"),
}
KEY = sys.argv[1] if len(sys.argv) > 1 else "pythia_agnews"; C = CFG[KEY]
AX = f"{G}/overlap_axis"; OUT = f"{G}/reversibility_v2_{KEY}.json"
N_SHELLS = 32; N_FISHER = 200; VAL_N = 300; BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02
LAB = {0: " negative", 1: " positive"}; dev = "cuda"; torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    va = [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"] if r["label"] in (0, 1)][:VAL_N]
    tr = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)]
    tr = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:N_FISHER]]
    return tr, va


def taskb():
    d = json.load(open(f"{AX}/{C['taskB']}.json")); t = d["prompt_template"]
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


@torch.no_grad()
def acc(model, tok, va):
    Lc = {}
    for cl in (0, 1):
        per = []; items = [(s, cl) for s, _ in va]
        for i in range(0, len(items), 64):
            inp, lb, am = b_sst(tok, items[i:i + 64]); lg = model(input_ids=inp, attention_mask=am).logits
            sl = lg[:, :-1].reshape(-1, lg.shape[-1]); st = lb[:, 1:].reshape(-1)
            ll = F.cross_entropy(sl, st, ignore_index=-100, reduction="none").reshape(lb.shape[0], -1)
            per.extend((ll.sum(1) / (lb[:, 1:] != -100).sum(1).clamp(min=1)).tolist())
        Lc[cl] = per
    return float(np.mean([int((0 if Lc[0][i] < Lc[1][i] else 1) == va[i][1]) for i in range(len(va))]))


def main():
    t0 = time.time(); print(f"[REV2] {KEY} loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C["model"]); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(C["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(C["ckpt"]))
    tr, va = sst(); B_items = taskb()
    names = [n for n, p in model.state_dict().items() if C["square"] in n and n.endswith("weight") and tuple(p.shape) == (1024, 1024)]
    params = dict(model.named_parameters())
    kf = np.load(C["kfac"]); UA = {n: torch.tensor(kf[f"{n[:-len('.weight')]}__UA"]).double().to(dev) for n in names}
    eigA = np.concatenate([kf[f"{n[:-len('.weight')]}__eA"] for n in names])
    order = np.argsort(eigA); shell_id = np.empty(len(eigA), int); edges = np.linspace(0, len(eigA), N_SHELLS + 1).astype(int)
    for s in range(N_SHELLS): shell_id[order[edges[s]:edges[s + 1]]] = s
    sizes = [UA[n].shape[1] for n in names]; offs = np.cumsum([0] + sizes); sh_by_mat = {n: shell_id[offs[i]:offs[i + 1]] for i, n in enumerate(names)}

    model.eval(); acc_A = acc(model, tok, va); wA = {n: params[n].detach().clone() for n in names}
    # Fisher (task-A empirical) at w_A BEFORE training
    fisher = {n: torch.zeros(UA[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for s, l in tr:
        inp, lb, am = b_sst(tok, [(s, l)]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names:
            g = params[n].grad.detach().double(); gp = (UA[n].T @ g) if C["conv1d"] else (g @ UA[n])
            fisher[n] += (gp * gp).sum(1 if C["conv1d"] else 0)
    imp = np.zeros(N_SHELLS)
    for n in names:
        fd = fisher[n].cpu().numpy(); sm = sh_by_mat[n]
        for s in range(N_SHELLS): imp[s] += fd[sm == s].sum()
    print(f"[REV2] acc_A={acc_A:.3f}, Fisher done; training task-B FULL model... ({time.time()-t0:.0f}s)", flush=True)

    # ---- train task-B FULL model ----
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        ordr = torch.randperm(len(B_items), generator=g).tolist()
        for i in range(0, len(ordr) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B_items[k] for k in ordr[i:i + BATCH]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); acc_B = acc(model, tok, va); forgetting = acc_A - acc_B
    wB = {n: params[n].detach().clone() for n in names}                     # full-w_B attention.dense
    dW = {n: (wB[n] - wA[n]) for n in names}
    Ccoord = {n: ((UA[n].T @ dW[n].double()) if C["conv1d"] else (dW[n].double() @ UA[n])) for n in names}
    print(f"[REV2] task-B trained; acc_B(FULL forgotten)={acc_B:.3f} forgetting={forgetting:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    def set_attn(state):  # state[n] -> tensor for attention.dense; rest stays at full w_B
        for n in names: params[n].data = state[n].data

    # ---- LOAD-BEARING pre-check: restore ALL attention.dense to w_A in full w_B ----
    set_attn(wA); acc_restore_all = acc(model, tok, va); set_attn(wB)
    rec_all = (acc_restore_all - acc_B) / max(forgetting, 1e-9)
    print(f"[REV2] LOAD-BEARING: restore ALL attention.dense -> acc {acc_restore_all:.3f} (recovers {rec_all:+.2f} of forgetting)", flush=True)

    # ---- per-shell recovery from full w_B ----
    rec_acc = np.zeros(N_SHELLS); norm_removed = np.zeros(N_SHELLS)
    for s in range(N_SHELLS):
        state = {}
        for n in names:
            mask = torch.tensor(sh_by_mat[n] == s, device=dev); Cs = Ccoord[n].clone()
            if C["conv1d"]: Cs[mask, :] = 0; dkeep = (UA[n] @ Cs)
            else: Cs[:, mask] = 0; dkeep = (Cs @ UA[n].T)
            state[n] = (wA[n].double() + dkeep).float()
            norm_removed[s] += float(((dW[n].double() - dkeep) ** 2).sum().item())
        set_attn(state); rec_acc[s] = acc(model, tok, va)
    set_attn(wB)
    recovery = (rec_acc - acc_B) / max(forgetting, 1e-9)

    imp_hi = imp >= np.median(imp); rec_hi = recovery >= 0.5
    cells = {k: [] for k in ("hi_imp_recovers", "hi_imp_permanent", "lo_imp_recovers", "lo_imp_permanent")}
    for s in range(N_SHELLS): cells[("hi_imp" if imp_hi[s] else "lo_imp") + ("_recovers" if rec_hi[s] else "_permanent")].append(s)
    stiff = list(range(N_SHELLS - 8, N_SHELLS)); flat = list(range(8))
    si_hi = float(imp_hi[stiff].mean()); s_rec = float(rec_hi[stiff].mean()); fi_hi = float(imp_hi[flat].mean()); f_rec = float(rec_hi[flat].mean())
    pred_holds = bool(si_hi > 0.5 and s_rec < 0.5 and fi_hi < 0.5 and f_rec > 0.5)
    valid = bool(abs(forgetting) > 0.03)                                     # test only meaningful if real forgetting

    print(f"\n[REV2] === {KEY}: 2D MAP (forgetting {forgetting:+.3f}, test-valid={valid}) ===", flush=True)
    print(f"  LOAD-BEARING (restore all attn.dense recovers): {rec_all:+.2f} of forgetting", flush=True)
    print(f"  cells: hi-imp+recovers={len(cells['hi_imp_recovers'])} hi-imp+PERM={len(cells['hi_imp_permanent'])} lo-imp+recovers={len(cells['lo_imp_recovers'])} lo-imp+perm={len(cells['lo_imp_permanent'])}", flush=True)
    print(f"  STIFF: imp-hi={si_hi:.2f} recovers={s_rec:.2f} | FLAT: imp-hi={fi_hi:.2f} recovers={f_rec:.2f}", flush=True)
    print(f"  recovery by shell: {np.round(recovery,2).tolist()}", flush=True)
    print(f"  PREDICTION (stiff=hi-imp+perm, flat=lo-imp+recovers) HOLDS={pred_holds}", flush=True)
    json.dump({"substrate": KEY, "acc_A": acc_A, "acc_B": acc_B, "forgetting": forgetting, "test_valid": valid,
               "load_bearing_recover_all_attn": rec_all, "acc_restore_all_attn": acc_restore_all,
               "importance_shell": imp.tolist(), "recovery_shell": recovery.tolist(), "rec_acc_shell": rec_acc.tolist(),
               "norm_removed_shell": norm_removed.tolist(), "cells": cells, "stiff_imp_hi": si_hi, "stiff_recovers": s_rec,
               "flat_imp_hi": fi_hi, "flat_recovers": f_rec, "prediction_holds": pred_holds, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[REV2] saved reversibility_v2_{KEY}.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
