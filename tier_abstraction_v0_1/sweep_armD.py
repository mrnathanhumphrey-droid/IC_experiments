"""Arm D sweep: alpha (adversarial strength) at fixed beta=0.1. alpha=0 reproduces Arm C."""
import argparse, subprocess, sys, os, itertools

ALPHAS = [0.0, 0.1, 0.5, 1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="results"); ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
    combos = list(itertools.product(args.alphas, args.seeds))
    print(f"=== armD sweep: {len(combos)} runs (alphas={args.alphas} seeds={args.seeds} beta={args.beta}) ===")
    for i, (alpha, seed) in enumerate(combos):
        run = f"armD_beta{args.beta}_a{alpha}_K{args.K}_seed{seed}"
        feat = os.path.join(here, args.out_dir, f"features_{run}.npz")
        print(f"\n--- [{i+1}/{len(combos)}] {run} ---")
        if not os.path.exists(feat):
            subprocess.run([py, os.path.join(here, "train_armD.py"), "--alpha", str(alpha),
                            "--beta", str(args.beta), "--K", str(args.K), "--seed", str(seed),
                            "--epochs", str(args.epochs), "--out_dir", args.out_dir, "--ckpt_dir", args.ckpt_dir],
                           check=True, cwd=here)
        subprocess.run([py, os.path.join(here, "evaluate.py"), "--features", feat, "--out_dir", args.out_dir],
                       check=True, cwd=here)
    print("\n=== armD sweep complete ===")


if __name__ == "__main__":
    main()
