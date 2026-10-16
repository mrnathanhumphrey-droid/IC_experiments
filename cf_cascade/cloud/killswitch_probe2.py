"""Killswitch insurance — terminate ONLY the two Probe-2 gh200 instances if they haven't self-terminated in 4h.
Targets exact instance IDs (NOT the ML-CF key) so it CANNOT touch the unrelated a10 or any parallel lane. Normally finds
nothing (launchers finally-terminate ~2-3h); pure insurance against a hung/dead launcher orphaning a GPU."""
import time, requests
from pathlib import Path
IDS = ["bc0ebef907aa486d944ac3f70ac93f31",   # opt350m  (lambda_launch_probe2.py opt350m)
       "c014a1e2eef142b9897b497752e58eaa"]   # pythia410m (lambda_launch_probe2.py pythia410m)
DELAY = 14400  # 4 hours
key = [l.split("=", 1)[1].strip() for l in open(Path("D:/physics_detector") / ".env") if l.startswith("CLOUD_LAMBDA=")][0]
H = {"Authorization": "Bearer " + key}
print(f"[KILLSWITCH-P2] armed {DELAY}s; targets (exact IDs only) = {IDS}", flush=True)
time.sleep(DELAY)
try:
    live = requests.get("https://cloud.lambda.ai/api/v1/instances", headers=H, timeout=30).json()["data"]
    liveids = {i["id"] for i in live}
    tokill = [i for i in IDS if i in liveids]
    if tokill:
        r = requests.post("https://cloud.lambda.ai/api/v1/instance-operations/terminate", headers=H,
                          json={"instance_ids": tokill}, timeout=30)
        print(f"[KILLSWITCH-P2] FIRED — terminated runaway {tokill} rc={r.status_code} {r.text[:200]}", flush=True)
    else:
        print("[KILLSWITCH-P2] CLEAN — both Probe-2 instances already self-terminated; nothing to kill.", flush=True)
except Exception as e:
    print(f"[KILLSWITCH-P2] ERROR {e} -- MANUAL CHECK instances {IDS}", flush=True)
