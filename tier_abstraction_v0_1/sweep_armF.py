"""Arm F/U sweep: {asym, uniform} x beta. Asym=arm F, uniform=arm U (internal control)."""
import argparse, subprocess, sys, os, itertools

BETAS = [0.01, 0.1, 0.2, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--betas", type=float, nargs="+", default=BETAS)
    ap.add_argument("--mults", type=str, nargs="+", default=["asym", "uniform"])
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="results"); ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
    combos = list(itertools.product(args.mults, args.betas, args.seeds))
    print(f"=== armF/U sweep: {len(combos)} runs (mults={args.mults} betas={args.betas} seeds={args.seeds}) ===")
    for i, (mult, beta, seed) in enumerate(combos):
        arm = "F" if mult == "asym" else "U"
        run = f"arm{arm}_beta{beta}_K{args.K}_seed{seed}"
        feat = os.path.join(here, args.out_dir, f"features_{run}.npz")
        print(f"\n--- [{i+1}/{len(combos)}] {run} ({mult}) ---")
        if not os.path.exists(feat):
            subprocess.run([py, os.path.join(here, "train_armF.py"), "--beta", str(beta), "--mult", mult,
                            "--K", str(args.K), "--seed", str(seed), "--epochs", str(args.epochs),
                            "--out_dir", args.out_dir, "--ckpt_dir", args.ckpt_dir], check=True, cwd=here)
        subprocess.run([py, os.path.join(here, "evaluate.py"), "--features", feat, "--out_dir", args.out_dir],
                       check=True, cwd=here)
    print("\n=== armF/U sweep complete ===")


if __name__ == "__main__":
    main()
