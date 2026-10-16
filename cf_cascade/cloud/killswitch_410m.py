"""Overnight kill switch — terminate ONLY the two 410m recipe-matched instances if they haven't self-terminated in 3h.
Targets exact instance IDs (NOT the ML-CF key) so it cannot touch parallel-agent instances. Normally finds nothing (they
self-terminate ~30min); this is pure insurance against a hung launcher orphaning a GPU overnight.
"""
import time, requests
from pathlib import Path
IDS = ["8bf21a00f09a49b894db68c767fe216f", "c03abc8a21ef44f8bd5a08a533fd2c9d"]  # bn5kc2fz8 std, bc3tdg1k3 dedup
DELAY = 10800  # 3 hours
key = [l.split("=", 1)[1].strip() for l in open(Path("D:/physics_detector") / ".env") if l.startswith("CLOUD_LAMBDA=")][0]
H = {"Authorization": "Bearer " + key}
print(f"[KILLSWITCH] armed {DELAY}s; targets (exact IDs only) = {IDS}", flush=True)
time.sleep(DELAY)
try:
    live = requests.get("https://cloud.lambda.ai/api/v1/instances", headers=H, timeout=30).json()["data"]
    liveids = {i["id"] for i in live}
    tokill = [i for i in IDS if i in liveids]
    if tokill:
        r = requests.post("https://cloud.lambda.ai/api/v1/instance-operations/terminate", headers=H,
                          json={"instance_ids": tokill}, timeout=30)
        print(f"[KILLSWITCH] FIRED — terminated runaway {tokill} rc={r.status_code} {r.text[:200]}", flush=True)
    else:
        print("[KILLSWITCH] CLEAN — both instances already self-terminated; nothing to kill.", flush=True)
except Exception as e:
    print(f"[KILLSWITCH] ERROR {e} -- MANUAL CHECK instances {IDS}", flush=True)
