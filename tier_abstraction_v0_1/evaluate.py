"""Run the 7 measurements for one run from its saved features npz. Writes eval_<run>.json.

  1. probe z_low  -> superclass acc
  2. probe z_trans-> superclass acc
  3. probe z_low  -> fine acc
  4. probe z_trans-> fine acc
  5. within-superclass confusion concentration (PRIMARY) + selectivity index
  6. CKA(z_low, z_trans)
  7. KL rate (read from train_<run>.json; the MI proxy)
"""
import argparse, json, os, glob
import numpy as np

from probes import linear_probe, within_super_confusion, linear_cka, chance_within_super


def eval_run(features_path, out_dir):
    run = os.path.basename(features_path)[len("features_"):-len(".npz")]
    d = np.load(features_path)
    zl_tr, zt_tr = d["z_low_tr"], d["z_trans_tr"]
    zl_te, zt_te = d["z_low_te"], d["z_trans_te"]
    fine_tr, coarse_tr = d["fine_tr"], d["coarse_tr"]
    fine_te, coarse_te = d["fine_te"], d["coarse_te"]

    # 1-2: superclass probes
    sup_low, _ = linear_probe(zl_tr, coarse_tr, zl_te, coarse_te)
    sup_trans, _ = linear_probe(zt_tr, coarse_tr, zt_te, coarse_te)
    # 3-4: fine probes (keep predictions for the confusion metric)
    fine_low, pred_fine_low = linear_probe(zl_tr, fine_tr, zl_te, fine_te)
    fine_trans, pred_fine_trans = linear_probe(zt_tr, fine_tr, zt_te, fine_te)

    # 5: within-superclass confusion concentration
    ws_low, nerr_low = within_super_confusion(pred_fine_low, fine_te)
    ws_trans, nerr_trans = within_super_confusion(pred_fine_trans, fine_te)
    selectivity = ws_trans - ws_low  # >0 => z_trans errors concentrate within superclass

    # 6: CKA
    cka = linear_cka(zl_te, zt_te)

    # 7: KL rate from training meta
    kl_rate = None
    train_meta = os.path.join(out_dir, f"train_{run}.json")
    if os.path.exists(train_meta):
        with open(train_meta) as f:
            kl_rate = json.load(f)["test"]["kl_rate"]

    res = {
        "run": run,
        "probe_super_zlow": sup_low,
        "probe_super_ztrans": sup_trans,
        "probe_fine_zlow": fine_low,
        "probe_fine_ztrans": fine_trans,
        "within_super_zlow": ws_low,
        "within_super_ztrans": ws_trans,
        "within_super_chance": chance_within_super(),
        "selectivity_index": selectivity,
        "n_err_zlow": nerr_low,
        "n_err_ztrans": nerr_trans,
        "cka_zlow_ztrans": cka,
        "kl_rate": kl_rate,
    }
    with open(os.path.join(out_dir, f"eval_{run}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{run}] sup low/trans={sup_low:.3f}/{sup_trans:.3f}  "
          f"fine low/trans={fine_low:.3f}/{fine_trans:.3f}  "
          f"WS low/trans={ws_low:.3f}/{ws_trans:.3f} sel={selectivity:+.3f}  "
          f"cka={cka:.3f} kl={kl_rate}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default=None, help="single features npz; omit to eval all in out_dir")
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    if args.features:
        eval_run(args.features, args.out_dir)
    else:
        paths = sorted(glob.glob(os.path.join(args.out_dir, "features_*.npz")))
        print(f"evaluating {len(paths)} runs")
        for p in paths:
            eval_run(p, args.out_dir)


if __name__ == "__main__":
    main()
