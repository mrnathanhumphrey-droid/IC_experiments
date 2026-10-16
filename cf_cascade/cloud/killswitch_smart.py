"""STALL-DETECTING killswitch for floor-battery (or any ML-CF) instances. Polls each target's pl.log mtime via SSH;
if a target's log hasn't advanced in STALL_SEC (= it's hung, no heartbeat), terminate THAT instance. Hard MAX_SEC backstop.
Exact instance IDs only (never the key) -> cannot touch other lanes. Pairs with the per-step heartbeat in the harness:
normal run advances pl.log every ~15-40s, so STALL_SEC=600 (10 min, ~3 polls) catches a true hang without false-killing a
slow-but-alive run. Conservative: only kills on a VALID stale reading, never on a transient SSH failure.

Usage:  python killswitch_smart.py <id1> <id2> ...   (ids printed by the launchers)
"""
import sys, time, subprocess, requests
from pathlib import Path
PD = Path("D:/physics_detector"); KEY = PD / "cloud" / "ssh" / "ml_cf_ed25519"
IDS = sys.argv[1:]
assert IDS, "usage: killswitch_smart.py <instance_id> [<instance_id> ...]"
POLL_SEC = 180; STALL_SEC = 600; MAX_SEC = 72000   # poll 3min; stall=10min no log progress; hard cap 20h
API = "https://cloud.lambda.ai/api/v1"
key = [l.split("=", 1)[1].strip() for l in open(PD / ".env") if l.startswith("CLOUD_LAMBDA=")][0]
H = {"Authorization": "Bearer " + key}


def status_ip(iid):
    try:
        for i in requests.get(API + "/instances", headers=H, timeout=30).json()["data"]:
            if i["id"] == iid: return i.get("status"), i.get("ip")
        return "gone", None
    except Exception as e:
        print(f"[SMARTKILL] api err {str(e)[:60]}", flush=True); return "err", None


def log_mtime(ip):
    try:
        r = subprocess.run(["ssh", "-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                            "-o", "ConnectTimeout=10", f"ubuntu@{ip}", "stat -c %Y /home/ubuntu/pl.log 2>/dev/null"],
                           capture_output=True, text=True, timeout=20)
        s = r.stdout.strip()
        return int(s) if s.isdigit() else None
    except Exception:
        return None


def terminate(iid):
    try:
        r = requests.post(API + "/instance-operations/terminate", headers=H, json={"instance_ids": [iid]}, timeout=30)
        print(f"[SMARTKILL] TERMINATED {iid} rc={r.status_code}", flush=True)
    except Exception as e:
        print(f"[SMARTKILL] terminate FAILED {iid} ({e}) -- MANUAL CHECK", flush=True)


def main():
    print(f"[SMARTKILL] watching {IDS}  poll={POLL_SEC}s stall={STALL_SEC}s maxfuse={MAX_SEC}s", flush=True)
    start = time.time(); alive = set(IDS)
    last_mtime = {i: None for i in IDS}; last_advance = {i: time.time() for i in IDS}
    while alive and (time.time() - start) < MAX_SEC:
        time.sleep(POLL_SEC)
        for iid in list(alive):
            st, ip = status_ip(iid)
            if st in ("gone", "terminated", "terminating"):
                print(f"[SMARTKILL] {iid[:12]} {st} -> done, drop", flush=True); alive.discard(iid); continue
            if st != "active" or not ip:
                continue  # still booting / transient api err -> wait
            m = log_mtime(ip)
            if m is None:
                continue  # transient ssh miss -> do NOT kill on one failure
            if last_mtime[iid] is None or m > last_mtime[iid]:
                last_mtime[iid] = m; last_advance[iid] = time.time()
                print(f"[SMARTKILL] {iid[:12]} alive (log advancing)", flush=True)
            else:
                stalled = time.time() - last_advance[iid]
                print(f"[SMARTKILL] {iid[:12]} NO log progress for {int(stalled)}s (thresh {STALL_SEC})", flush=True)
                if stalled > STALL_SEC:
                    print(f"[SMARTKILL] {iid[:12]} STALLED -> KILL", flush=True); terminate(iid); alive.discard(iid)
    for iid in alive:  # hard backstop
        st, _ = status_ip(iid)
        if st in ("active", "booting"):
            print(f"[SMARTKILL] {iid[:12]} hit MAX_SEC backstop -> KILL", flush=True); terminate(iid)
    print("[SMARTKILL] watcher exit", flush=True)


if __name__ == "__main__":
    main()
