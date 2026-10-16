"""Aggregate summary.csv across seeds -> mean +/- sd per (arm, beta) for the verdict."""
import csv, collections, statistics as st

rows = list(csv.DictReader(open("results/summary.csv")))
keys = ["kl_rate", "selectivity_index", "probe_super_ztrans", "probe_fine_ztrans",
        "probe_super_zlow", "probe_fine_zlow", "cka_zlow_ztrans"]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for k in keys:
        agg[(r["arm"], float(r["beta"]))][k].append(float(r[k]))

hdr = f'{"arm":>3} {"beta":>6} {"KL":>7}  {"selectivity (mean+/-sd)":>24}  {"sup<-tr":>7} {"fine<-tr":>8}  {"cka":>5}'
print(hdr)
print("-" * len(hdr))
for (a, b), d in sorted(agg.items()):
    n = len(d["selectivity_index"])
    sel_m = st.mean(d["selectivity_index"])
    sel_s = st.pstdev(d["selectivity_index"])
    print(f'{a:>3} {b:>6} {st.mean(d["kl_rate"]):>7.2f}  '
          f'{sel_m:>+10.3f} +/- {sel_s:<7.3f}  '
          f'{st.mean(d["probe_super_ztrans"]):>7.3f} {st.mean(d["probe_fine_ztrans"]):>8.3f}  '
          f'{st.mean(d["cka_zlow_ztrans"]):>5.3f}  (n={n})')
