"""Orchestrate the beta x arm x seed grid. Staged: pass --seeds 0 first (10 runs), eyeball,
then --seeds 0 1 2 to fill in. Each run trains then immediately evaluates.

Arm A: lam=0, use_super=0 ; Arm B: lam=0.5, use_super=1.  K fixed at 8 for v0.1.
"""
import argparse, subprocess, sys, os, itertools

BETAS = [0.0, 0.001, 0.01, 0.1, 1.0]
ARMS = {"A": dict(lam=0.0, use_super=0), "B": dict(lam=0.5, use_super=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", type=str, nargs="+", default=["A", "B"])
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    combos = list(itertools.product(args.arms, args.betas, args.seeds))
    print(f"=== sweep: {len(combos)} runs (arms={args.arms} betas={args.betas} seeds={args.seeds}) ===")

    for i, (arm, beta, seed) in enumerate(combos):
        cfg = ARMS[arm]
        run = f"arm{arm}_beta{beta}_K{args.K}_seed{seed}"
        feat = os.path.join(here, args.out_dir, f"features_{run}.npz")
        print(f"\n--- [{i+1}/{len(combos)}] {run} ---")
        if os.path.exists(feat):
            print("  features exist, skipping train; (re)evaluating")
        else:
            train_cmd = [py, os.path.join(here, "train.py"),
                         "--beta", str(beta), "--lam", str(cfg["lam"]),
                         "--use_super", str(cfg["use_super"]), "--arm", arm,
                         "--K", str(args.K), "--seed", str(seed), "--epochs", str(args.epochs),
                         "--out_dir", args.out_dir, "--ckpt_dir", args.ckpt_dir]
            subprocess.run(train_cmd, check=True, cwd=here)
        eval_cmd = [py, os.path.join(here, "evaluate.py"),
                    "--features", feat, "--out_dir", args.out_dir]
        subprocess.run(eval_cmd, check=True, cwd=here)

    print("\n=== sweep complete ===")


if __name__ == "__main__":
    main()
