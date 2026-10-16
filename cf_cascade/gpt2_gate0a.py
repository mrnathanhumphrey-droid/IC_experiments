"""JOB 2 Gate-0(a): does GPT-2-medium clear the substrate floors Pythia cleared, so the eigen-shell
geometry is valid? Two gates BEFORE any training (the K7/K8 requirements):
  K8 weight-ESD multi-decade on the square attention matrices (Pythia: 5.9 full / 4.4 robust decades).
     GPT-2-medium's square d×d attention matrix = attn.c_proj (Conv1D, weight (nx,nf)=(1024,1024),
     the output-projection analog of Pythia attention.dense). Conv1D weight is (in,out) — TRANSPOSED
     vs nn.Linear (out,in); ESD of W^T W gives the same nonzero spectrum either way (handle on damage).
  K7 index-autocorrelation ~0 (matrix-layout void -> eigen-shell is the right basis; Pythia 0.0011).
If GPT-2 ESD is narrow or autocorr nonzero -> finding about substrate-dependence, report before build.
CPU; downloads gpt2-medium (~1.5GB) clean from HF (no gating).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, numpy as np, torch
torch.set_num_threads(4)
OUT = "D:/IC_experiments/cf_cascade/gateA/gpt2_gate0a.json"


def decades(W):
    ev = torch.linalg.eigvalsh(W.T @ W / W.shape[0]).clamp(min=0).double().numpy()
    ev = ev[ev > 0]
    full = float(np.log10(ev.max() / ev.min()))
    robust = float(np.log10(np.percentile(ev, 97.5) / np.percentile(ev, 2.5)))
    return full, robust


def autocorr(W):
    # lag-1 autocorrelation along rows and cols (matrix-layout structure); ~0 => layout void
    A = W.numpy()
    def lag1(M):
        x = M[:, :-1].ravel(); y = M[:, 1:].ravel()
        return np.corrcoef(x, y)[0, 1]
    return float(np.median([abs(lag1(A)), abs(lag1(A.T))]))


def main():
    from transformers import AutoModelForCausalLM
    print("[G0a] downloading/loading gpt2-medium...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium", dtype=torch.float32)
    sd = model.state_dict()
    cproj = [n for n in sd if n.endswith("attn.c_proj.weight") and tuple(sd[n].shape) == (1024, 1024)]
    print(f"[G0a] found {len(cproj)} square attn.c_proj matrices (1024x1024)", flush=True)
    fulls, robs, acs = [], [], []
    for n in cproj:
        W = sd[n].detach().double()
        f, r = decades(W); fulls.append(f); robs.append(r); acs.append(autocorr(sd[n].detach()))
    res = {"n_matrices": len(cproj), "esd_decades_full_median": float(np.median(fulls)),
           "esd_decades_robust_median": float(np.median(robs)), "esd_full_min": float(np.min(fulls)),
           "index_autocorr_median": float(np.median(acs)),
           "K8_pass": bool(np.median(robs) > 2.0), "K7_pass": bool(np.median(acs) < 0.05)}
    print(f"\n[G0a] === GPT-2-medium SUBSTRATE FLOOR ===", flush=True)
    print(f"  K8 weight-ESD decades: full median={res['esd_decades_full_median']:.2f} (min {res['esd_full_min']:.2f}), robust median={res['esd_decades_robust_median']:.2f}  [Pythia 5.9/4.4]", flush=True)
    print(f"     -> K8 {'PASS' if res['K8_pass'] else 'FAIL'} (robust>2 decades for eigen-shell geometry)", flush=True)
    print(f"  K7 index-autocorr median={res['index_autocorr_median']:.4f}  [Pythia 0.0011]", flush=True)
    print(f"     -> K7 {'PASS' if res['K7_pass'] else 'FAIL'} (matrix-layout void, <0.05)", flush=True)
    print(f"  -> GATE-0a: {'CLEARED, proceed to task-A build' if (res['K8_pass'] and res['K7_pass']) else 'NOT CLEARED — substrate-dependence finding, report before build'}", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[G0a] saved gpt2_gate0a.json", flush=True)


if __name__ == "__main__":
    main()
