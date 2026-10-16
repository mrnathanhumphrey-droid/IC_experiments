"""Robust PSD certification of the SST-2 2k-subset interpolation minimum.

The 18-iter power iteration gave INCONSISTENT extremes (lambda_max +21 vs lambda_min -49:
a -49 eig would dominate the direct largest-magnitude iteration, which returned +21 ->
didn't converge). Fix: power iteration with early-stop-on-plateau (up to 200 iters),
2 random restarts, and THREE certification checks:
  (1) per-estimate PLATEAU: successive iterates agree to tol (converged, not truncated).
  (2) RESTART-AGREEMENT: independent seeds converge to the same eigenvalue.
  (3) CROSS-CHECK: direct largest-magnitude == max(|lambda_max+|, |lambda_min|).
All three green per estimate -> trust. lambda_min is load-bearing (the PSD gate) so it
gets the most scrutiny; the shift c = 1.5|lambda_lm|+10 must exceed the spectral radius
so (cI-H)'s top eigenvalue is the most-NEGATIVE original eigenvalue.

Certification-only on the saved checkpoint. No retraining.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
OUT = "D:/IC_experiments/cf_cascade/gateA/robust_cert.json"
BATCH = 8; HVP_BATCHES = 4; MAXITER = 200; PLATEAU_TOL = 1e-3; PLATEAU_WIN = 10; SUBSET = 2000
LABELS = {0: " negative", 1: " positive"}; dev = "cuda"


def load_sst2_subset():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    idx = np.random.default_rng(0).permutation(len(rows))[:SUBSET]
    return [rows[i] for i in idx]


def make_batch(tok, rows):
    seqs, labs = [], []
    for sent, lab in rows:
        p = tok(f"{sent}\nSentiment:", add_special_tokens=False)["input_ids"]
        c = tok(LABELS[lab], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    model.load_state_dict(torch.load(CKPT)); model.eval()
    for p in model.parameters(): p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    train = load_sst2_subset()
    g0 = torch.Generator().manual_seed(1)
    fixed = [make_batch(tok, [train[k] for k in torch.randperm(len(train), generator=g0)[:BATCH].tolist()]) for _ in range(HVP_BATCHES)]

    def hvp(vec):
        acc = [torch.zeros_like(p) for p in params]
        for inp, lb, am in fixed:
            loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((gp * v).sum() for gp, v in zip(grads, vec))
            hv = torch.autograd.grad(dot, params, retain_graph=False)
            for a, h in zip(acc, hv): a += h.detach()
            del grads, hv, loss
        return [a / len(fixed) for a in acc]

    def power(matvec, seed, tag):
        torch.manual_seed(seed)
        v = [torch.randn_like(p) for p in params]
        nrm = float(sum(x.pow(2).sum() for x in v) ** 0.5); v = [x / nrm for x in v]
        trace = []
        for it in range(MAXITER):
            Hv = matvec(v)
            lam = float(sum((hv * x).sum() for hv, x in zip(Hv, v)))
            nrm = float(sum(hv.pow(2).sum() for hv in Hv) ** 0.5) + 1e-12
            v = [hv / nrm for hv in Hv]
            trace.append(lam)
            if len(trace) >= PLATEAU_WIN:
                rec = trace[-PLATEAU_WIN:]
                if (max(rec) - min(rec)) / (abs(trace[-1]) + 1e-9) < PLATEAU_TOL:
                    break
        rec = trace[-PLATEAU_WIN:]
        plat = (max(rec) - min(rec)) / (abs(trace[-1]) + 1e-9) < PLATEAU_TOL
        print(f"    [{tag} seed{seed}] lam={lam:+.3f} iters={len(trace)} plateaued={plat}", flush=True)
        return lam, plat, len(trace)

    def estimate(matvec, tag):
        r0 = power(matvec, 0, tag); r1 = power(matvec, 1, tag)
        lam = float(np.median([r0[0], r1[0]]))
        plat = r0[1] and r1[1]
        agree = abs(r0[0] - r1[0]) / (abs(lam) + 1e-9) < 0.05
        return lam, plat, agree

    print(f"[cert] direct (largest magnitude)...", flush=True)
    lam_lm, lm_plat, lm_agree = estimate(lambda v: hvp(v), "direct")
    c = abs(lam_lm) * 1.5 + 10.0
    print(f"[cert] shift c={c:.2f}; most-negative via (cI - H)...", flush=True)
    res_neg, neg_plat, neg_agree = estimate(lambda v: [c * x - hv for hv, x in zip(hvp(v), v)], "neg")
    lam_min = c - res_neg
    if lam_lm > 0:
        lam_maxp = lam_lm; maxp_plat, maxp_agree = lm_plat, lm_agree
    else:
        print(f"[cert] most-positive via (H + cI)...", flush=True)
        res_pos, maxp_plat, maxp_agree = estimate(lambda v: [hv + c * x for hv, x in zip(hvp(v), v)], "pos")
        lam_maxp = res_pos - c

    cross_ok = abs(max(abs(lam_maxp), abs(lam_min)) - abs(lam_lm)) / (abs(lam_lm) + 1e-9) < 0.08
    all_converged = lm_plat and neg_plat and maxp_plat
    all_agree = lm_agree and neg_agree and maxp_agree
    shift_ok = c > abs(lam_maxp)        # shift exceeded most-positive -> negatives surfaced
    trustworthy = all_converged and all_agree and cross_ok and shift_ok
    is_psd = lam_min > -1e-2

    print(f"\n[cert] === ROBUST CERTIFICATION ===", flush=True)
    print(f"  lambda_max+ = {lam_maxp:+.3f}", flush=True)
    print(f"  lambda_min  = {lam_min:+.3f}   (THE PSD GATE)", flush=True)
    print(f"  direct largest-magnitude = {lam_lm:+.3f}", flush=True)
    print(f"  CHECK plateaued (all)     : {all_converged}", flush=True)
    print(f"  CHECK restart-agreement   : {all_agree}", flush=True)
    print(f"  CHECK cross (direct==maxmag): {cross_ok}  (maxmag={max(abs(lam_maxp),abs(lam_min)):.2f} vs |direct|={abs(lam_lm):.2f})", flush=True)
    print(f"  CHECK shift>lambda_max+   : {shift_ok}", flush=True)
    print(f"  -> TRUSTWORTHY: {trustworthy}", flush=True)
    if trustworthy:
        print(f"  -> VERDICT: {'GENUINE PSD MINIMUM (lambda_min>0) - proceed to curvature read' if is_psd else 'REAL SADDLE (lambda_min<0, converged) - switch task A'}", flush=True)
    else:
        print(f"  -> NOT TRUSTWORTHY - estimator still unconverged; increase iters/batches, do NOT gate", flush=True)
    json.dump({"lambda_max_pos": lam_maxp, "lambda_min": lam_min, "direct_largest_mag": lam_lm,
               "plateaued": bool(all_converged), "restart_agree": bool(all_agree),
               "cross_check": bool(cross_ok), "shift_ok": bool(shift_ok),
               "trustworthy": bool(trustworthy), "is_psd_minimum": bool(is_psd),
               "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[cert] saved robust_cert.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
