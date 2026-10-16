"""Force lane F0-F2 on the genuine-PSD SST-2 minimum, main FK6 pair (task-B = AG-News).
F0 binding = K-FAC per-matrix curvature eigenbasis at the A-min (gauge-free, FK3).
F1 pull = AG-News initial gradient at the A-min, rotated into the binding eigenbasis.
F2 = energy-form escape g^2/H per direction (FK1: g^2/H, NOT g/H Newton step) + the 2x2
occupancy (the first-knife): is the high-binding & high-pull cell populated, or are pull and
binding COLLINEAR (-> force frame collapses to localization, FK6)?

Two-scale caveat: high-binding dirs are the FEW stiff outliers; the load-bearing question is
whether AG-News pull lands on those few (stiff-AND-pulled -> forgets despite stiff = orthogonal
info) or only on the flat bulk (collinear -> binding alone explains it).
Saves per-direction (g^2, H) for the force-lane F3 regression (FK5).
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"; OUT = "D:/IC_experiments/cf_cascade/gateA/force_f2.json"
BATCH = 16; N_KFAC = 16; N_GRAD = 16; SUBSET = 2000; dev = "cuda"
LAB_SST = {0: " negative", 1: " positive"}; torch.manual_seed(0)


def sst_items(tok):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:SUBSET]]
    return [(f"{s}\nSentiment:", LAB_SST[l]) for s, l in rows]


def agnews_items():
    d = json.load(open(f"{AX}/low_agnews.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def batch_from(tok, items):
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
    t0 = time.time(); print("[F2] start; loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.eval()
    model.load_state_dict(torch.load(CKPT))
    A_items = sst_items(tok); B_items = agnews_items()
    print(f"[F2] loaded ({time.time()-t0:.0f}s)", flush=True)

    dense = {n: m for n, m in model.named_modules() if n.endswith("attention.dense")}
    A_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}
    G_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}; cnt = {n: 0 for n in dense}; cur = {}
    H = []
    for n, m in dense.items():
        def fwd(mod, i, o, nn=n): cur[nn] = i[0].detach()
        def bwd(mod, gi, go, nn=n):
            x = cur[nn].reshape(-1, 1024); g = go[0].detach().reshape(-1, 1024); k = g.abs().sum(1) > 0
            if k.any(): A_acc[nn] += x[k].T @ x[k]; G_acc[nn] += g[k].T @ g[k]; cnt[nn] += int(k.sum())
        H += [m.register_forward_hook(fwd), m.register_full_backward_hook(bwd)]
    gA = torch.Generator().manual_seed(7)
    for _ in range(N_KFAC):
        inp, lb, am = batch_from(tok, [A_items[k] for k in torch.randperm(len(A_items), generator=gA)[:BATCH].tolist()])
        model.zero_grad(set_to_none=True); model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
    for h in H: h.remove()
    basis = {}
    for n in dense:
        A = (A_acc[n] / max(cnt[n], 1)).double(); G = (G_acc[n] / max(cnt[n], 1)).double()
        eA, UA = torch.linalg.eigh(A); eG, UG = torch.linalg.eigh(G)
        basis[n] = (eA.clamp(min=1e-12).float(), UA.float(), eG.clamp(min=1e-12).float(), UG.float())
    print(f"[F2] F0 binding bases built ({time.time()-t0:.0f}s)", flush=True)

    # F1 pull = AG-News gradient at A-min
    gB = {n: torch.zeros(1024, 1024, device=dev) for n in dense}; gg = torch.Generator().manual_seed(11)
    for _ in range(N_GRAD):
        inp, lb, am = batch_from(tok, [B_items[k] for k in torch.randperm(len(B_items), generator=gg)[:BATCH].tolist()])
        model.zero_grad(set_to_none=True); (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_GRAD).backward()
        for n, m in dense.items(): gB[n] += m.weight.grad.detach()
    print(f"[F2] F1 pull computed ({time.time()-t0:.0f}s)", flush=True)

    # F2: per-direction g^2 (pull) and H (binding); energy-form escape g^2/H; 2x2 occupancy
    g2_all, H_all = [], []
    for n in dense:
        eA, UA, eG, UG = basis[n]
        C = (UG.T @ gB[n] @ UA)                       # pull in binding eigenbasis
        g2 = (C * C).flatten().double().cpu().numpy()
        Hd = torch.outer(eG, eA).flatten().double().cpu().numpy()
        # subsample per matrix to keep memory bounded
        idx = np.random.default_rng(0).integers(0, len(g2), 40000)
        g2_all.append(g2[idx]); H_all.append(Hd[idx])
    g2 = np.concatenate(g2_all); Hd = np.concatenate(H_all)
    keep = (g2 > 0) & (Hd > 0); g2, Hd = g2[keep], Hd[keep]
    escape = g2 / Hd                                  # FK1 energy-form

    # 2x2 occupancy: median split on H (stiff/sloppy) x g^2 (high/low pull)
    Hmed, gmed = np.median(Hd), np.median(g2)
    stiff = Hd >= Hmed; hipull = g2 >= gmed
    cell = {"stiff_hipull": int((stiff & hipull).sum()), "stiff_lopull": int((stiff & ~hipull).sum()),
            "sloppy_hipull": int((~stiff & hipull).sum()), "sloppy_lopull": int((~stiff & ~hipull).sum())}
    tot = sum(cell.values())
    # collinearity: correlation of log g^2 vs log H. high |corr| -> pull tracks binding -> frame collapses
    lg, lh = np.log(g2 + 1e-30), np.log(Hd + 1e-30)
    corr = float(np.corrcoef(lg, lh)[0, 1])
    # the discriminating signature (FK5/2x2): stiff-but-pulled present AND sloppy-but-unpulled present
    frac_stiff_hipull = cell["stiff_hipull"] / tot
    expected_if_independent = 0.25
    populated = frac_stiff_hipull > 0.5 * expected_if_independent       # not strongly anti-populated
    collinear = abs(corr) > 0.5

    print(f"\n[F2] === FORCE BALANCE 2x2 OCCUPANCY (task-B = AG-News) ===", flush=True)
    print(f"  2x2 cells (frac): stiff&hipull={cell['stiff_hipull']/tot:.3f}  stiff&lopull={cell['stiff_lopull']/tot:.3f}", flush=True)
    print(f"                    sloppy&hipull={cell['sloppy_hipull']/tot:.3f}  sloppy&lopull={cell['sloppy_lopull']/tot:.3f}", flush=True)
    print(f"  pull-binding collinearity corr(log g^2, log H) = {corr:+.3f}  ({'COLLINEAR' if collinear else 'not collinear'})", flush=True)
    print(f"  escape g^2/H: median={np.median(escape):.3e}  p99={np.percentile(escape,99):.3e}  p99/median={np.percentile(escape,99)/max(np.median(escape),1e-30):.1e}", flush=True)
    if collinear:
        verdict = "COLLINEAR -> force frame ~collapses to localization on AG-News (pull tracks binding); FK6 finding for this overlap regime"
    elif populated:
        verdict = "POPULATED + not-collinear -> high-binding/high-pull cell occupied, pull carries info beyond binding -> force frame has orthogonal content, proceed to F3"
    else:
        verdict = "stiff&hipull DEPLETED -> AG-News pull avoids the few stiff dirs -> binding alone may dominate; F3 regression decides"
    print(f"  VERDICT: {verdict}", flush=True)
    json.dump({"task_B": "AG-News", "cells_frac": {k: v / tot for k, v in cell.items()},
               "collinearity_corr_logg2_logH": corr, "collinear": collinear,
               "escape_median": float(np.median(escape)), "escape_p99": float(np.percentile(escape, 99)),
               "frac_stiff_hipull": frac_stiff_hipull, "high_binding_pulled_populated": bool(populated),
               "verdict": verdict, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[F2] saved force_f2.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
