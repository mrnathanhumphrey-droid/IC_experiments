"""Lambda cost ledger + threshold alerts. Called PRE-launch (estimate) and POST-terminate (log).

Alert thresholds (user-set 2026-06-13):
  PER-RUN  >= $70   -> alert + confirm before launching
  DAILY    >= $100  (~5 x $20 runs) -> alert + confirm
  DAILY runs >= 5   -> alert + confirm

Usage:
  python lambda_ledger.py estimate --rate 199 --hours 6          # pre-launch check
  python lambda_ledger.py log --id <id> --type gpu_1x_a100_sxm4 --rate 199 --minutes 360 [--note "..."]
"""
import argparse, csv, os, datetime

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lambda_cost_ledger.csv")
PER_RUN_ALERT = 70.0
DAILY_ALERT = 100.0          # user's hard gate: warn at $100 spend
DAILY_COUNT_ALERT = 999      # count alert disabled per user — $100 spend is the gate, not run count
COLS = ["date", "instance_id", "type", "rate_c_per_hr", "minutes", "cost_usd", "note"]


def today():
    return datetime.date.today().isoformat()


def day_totals(d):
    if not os.path.exists(LEDGER):
        return 0.0, 0
    t, c = 0.0, 0
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            if r["date"] == d:
                t += float(r["cost_usd"]); c += 1
    return round(t, 2), c


def alerts(run_cost, day_total, day_count, prefix):
    fired = []
    if run_cost >= PER_RUN_ALERT:
        fired.append(f"PER-RUN ${run_cost:.2f} >= ${PER_RUN_ALERT:.0f}")
    if day_total >= DAILY_ALERT:
        fired.append(f"DAILY ${day_total:.2f} >= ${DAILY_ALERT:.0f}")
    if day_count >= DAILY_COUNT_ALERT:
        fired.append(f"DAILY-RUNS {day_count} >= {DAILY_COUNT_ALERT}")
    if fired:
        print(f"\n*** COST ALERT ({prefix}) ***")
        for x in fired:
            print("   !! " + x)
        print("   -> STOP: confirm with user before spending.\n")
        return True
    print(f"  [{prefix} OK] run ${run_cost:.2f} | day ${day_total:.2f} over {day_count} run(s) — under thresholds")
    return False


def cmd_estimate(a):
    cost = round(a.rate / 100.0 * a.hours, 2)
    d = today(); tot, cnt = day_totals(d)
    print(f"ESTIMATE ${cost:.2f}  ({a.rate}c/hr x {a.hours}hr). Today so far ${tot:.2f}/{cnt} runs -> would become ${round(tot+cost,2):.2f}/{cnt+1}.")
    alerts(cost, round(tot + cost, 2), cnt + 1, "pre-launch")


def cmd_log(a):
    cost = round(a.rate / 100.0 * (a.minutes / 60.0), 2)
    d = today(); new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new:
            w.writeheader()
        w.writerow({"date": d, "instance_id": a.id, "type": a.type, "rate_c_per_hr": a.rate,
                    "minutes": a.minutes, "cost_usd": cost, "note": a.note})
    tot, cnt = day_totals(d)
    print(f"LOGGED ${cost:.2f}  ({a.type}, {a.minutes}min). TODAY: ${tot:.2f} over {cnt} run(s).")
    alerts(cost, tot, cnt, "post-run")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate"); e.add_argument("--rate", type=float, required=True); e.add_argument("--hours", type=float, required=True)
    l = sub.add_parser("log")
    l.add_argument("--id", default="?"); l.add_argument("--type", required=True)
    l.add_argument("--rate", type=float, required=True); l.add_argument("--minutes", type=float, required=True); l.add_argument("--note", default="")
    a = p.parse_args()
    {"estimate": cmd_estimate, "log": cmd_log}[a.cmd](a)


if __name__ == "__main__":
    main()
