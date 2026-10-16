"""Sweep Arm C (superclass-primary) over beta x seed; train then evaluate each."""
import argparse, subprocess, sys, os, itertools

BETAS = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    combos = list(itertools.product(args.betas, args.seeds))
    print(f"=== armC sweep: {len(combos)} runs (betas={args.betas} seeds={args.seeds}) ===")
    for i, (beta, seed) in enumerate(combos):
        run = f"armC_beta{beta}_K{args.K}_seed{seed}"
        feat = os.path.join(here, args.out_dir, f"features_{run}.npz")
        print(f"\n--- [{i+1}/{len(combos)}] {run} ---")
        if not os.path.exists(feat):
            subprocess.run([py, os.path.join(here, "train_armC.py"),
                            "--beta", str(beta), "--K", str(args.K), "--seed", str(seed),
                            "--epochs", str(args.epochs), "--out_dir", args.out_dir,
                            "--ckpt_dir", args.ckpt_dir], check=True, cwd=here)
        subprocess.run([py, os.path.join(here, "evaluate.py"),
                        "--features", feat, "--out_dir", args.out_dir], check=True, cwd=here)
    print("\n=== armC sweep complete ===")


if __name__ == "__main__":
    main()
