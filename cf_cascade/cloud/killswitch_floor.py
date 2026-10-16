"""Killswitch insurance — terminate ONLY the two floor-battery drift instances if they haven't self-terminated in 7h.
Exact instance IDs only (NOT the ML-CF key) -> cannot touch the unrelated a10 (ecf7157c) or any parallel lane. a10 is
slow (~6h for the drift block) so the fuse is long. Pure insurance against a hung/dead launcher orphaning a GPU."""
import time, requests
from pathlib import Path
IDS = ["807c5365f8b94e84a4edd86ebf40c32d",   # opt350m bf16 block (a100, 5 repeats)
       "a0f46ba04ab1418d8912aacc31b9cdb0"]   # pythia410m bf16 block (a100, 5 repeats)
DELAY = 32400  # 9 hours (a100 bf16 5-repeat block ~6.5h; 9h catches a true hang)
key = [l.split("=", 1)[1].strip() for l in open(Path("D:/physics_detector") / ".env") if l.startswith("CLOUD_LAMBDA=")][0]
H = {"Authorization": "Bearer " + key}
print(f"[KILLSWITCH-FLOOR] armed {DELAY}s; targets (exact IDs only) = {IDS}", flush=True)
time.sleep(DELAY)
try:
    live = requests.get("https://cloud.lambda.ai/api/v1/instances", headers=H, timeout=30).json()["data"]
    liveids = {i["id"] for i in live}
    tokill = [i for i in IDS if i in liveids]
    if tokill:
        r = requests.post("https://cloud.lambda.ai/api/v1/instance-operations/terminate", headers=H,
                          json={"instance_ids": tokill}, timeout=30)
        print(f"[KILLSWITCH-FLOOR] FIRED — terminated runaway {tokill} rc={r.status_code} {r.text[:200]}", flush=True)
    else:
        print("[KILLSWITCH-FLOOR] CLEAN — both drift instances already self-terminated; nothing to kill.", flush=True)
except Exception as e:
    print(f"[KILLSWITCH-FLOOR] ERROR {e} -- MANUAL CHECK instances {IDS}", flush=True)
