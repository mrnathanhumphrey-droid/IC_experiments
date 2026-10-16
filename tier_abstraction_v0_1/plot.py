"""Aggregate eval_*.json + train_*.json across seeds and render the sweet-spot plot.

Two-panel-per-arm figure vs beta (mean +/- std over seeds):
  - probe accuracies (super/fine on z_low and z_trans)
  - within-super selectivity index, CKA, KL rate
Also writes a tidy summary.csv.
"""
import argparse, json, os, glob, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_RE = re.compile(r"arm(?P<arm>[A-Z])_beta(?P<beta>[0-9.]+)_K(?P<K>\d+)_seed(?P<seed>\d+)")


def load_rows(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "eval_*.npz".replace("npz", "json")))):
        with open(p) as f:
            r = json.load(f)
        m = RUN_RE.search(r["run"])
        if not m:
            continue
        r["arm"] = m.group("arm"); r["beta"] = float(m.group("beta")); r["seed"] = int(m.group("seed"))
        rows.append(r)
    return rows


def agg(rows, arm, key):
    """mean,std of `key` vs beta for a given arm."""
    by_beta = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key])):
            by_beta[r["beta"]].append(r[key])
    betas = sorted(by_beta)
    mean = [float(np.mean(by_beta[b])) for b in betas]
    std = [float(np.std(by_beta[b])) for b in betas]
    return betas, mean, std


def write_csv(rows, path):
    cols = ["run", "arm", "beta", "seed", "probe_super_zlow", "probe_super_ztrans",
            "probe_fine_zlow", "probe_fine_ztrans", "within_super_zlow", "within_super_ztrans",
            "within_super_chance", "selectivity_index", "cka_zlow_ztrans", "kl_rate"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: (x["arm"], x["beta"], x["seed"])):
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def plot(rows, out_dir):
    arms = sorted({r["arm"] for r in rows})
    fig, axes = plt.subplots(2, len(arms), figsize=(7 * len(arms), 9), squeeze=False)
    for j, arm in enumerate(arms):
        ax = axes[0][j]
        for key, lab in [("probe_super_zlow", "super<-z_low"), ("probe_super_ztrans", "super<-z_trans"),
                         ("probe_fine_zlow", "fine<-z_low"), ("probe_fine_ztrans", "fine<-z_trans")]:
            b, m, s = agg(rows, arm, key)
            if b:
                ax.errorbar(b, m, yerr=s, marker="o", capsize=3, label=lab)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_title(f"Arm {arm}: probe accuracy vs beta")
        ax.set_xlabel("beta"); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax2 = axes[1][j]
        for key, lab in [("selectivity_index", "selectivity (WS_trans-WS_low)"),
                         ("cka_zlow_ztrans", "CKA(z_low,z_trans)")]:
            b, m, s = agg(rows, arm, key)
            if b:
                ax2.errorbar(b, m, yerr=s, marker="s", capsize=3, label=lab)
        ax2.axhline(0, color="k", lw=0.6)
        ax2b = ax2.twinx()
        b, m, s = agg(rows, arm, "kl_rate")
        if b:
            ax2b.errorbar(b, m, yerr=s, marker="^", color="purple", capsize=3, label="KL rate (MI proxy)")
            ax2b.set_ylabel("KL rate (nats)", color="purple")
        ax2.set_xscale("symlog", linthresh=1e-3)
        ax2.set_title(f"Arm {arm}: abstraction signature vs beta")
        ax2.set_xlabel("beta"); ax2.set_ylabel("selectivity / CKA"); ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(out_dir, "sweet_spot.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()
    rows = load_rows(args.out_dir)
    if not rows:
        print("no eval_*.json found in", args.out_dir); return
    write_csv(rows, os.path.join(args.out_dir, "summary.csv"))
    print("wrote", os.path.join(args.out_dir, "summary.csv"), f"({len(rows)} rows)")
    plot(rows, args.out_dir)


if __name__ == "__main__":
    main()
