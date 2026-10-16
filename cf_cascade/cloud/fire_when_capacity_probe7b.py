"""Fire-when-capacity watcher for Probe-7b opt350m. Polls Lambda capacity on non-a10 tiers; the instant one frees,
runs the approved launcher (which picks/launches/runs/self-terminates) and arms the smart killswitch on the real id.
NEVER drops to a10. Retries on NO-CAPACITY race. Gives up after 8h. Read p7b_fire.log for the launcher's own output.
"""
import subprocess, time, re, os, sys, requests
from pathlib import Path

PREF = "gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_gh200,gpu_1x_h100_sxm5"
TIERS = PREF.split(",")
HERE = Path(__file__).parent
key = [l.split("=", 1)[1].strip() for l in open(Path("D:/physics_detector/.env")) if l.startswith("CLOUD_LAMBDA=")][0]
H = {"Authorization": "Bearer " + key}
T0 = time.time()


def cap_regions():
    try:
        c = requests.get("https://cloud.lambda.ai/api/v1/instance-types", headers=H, timeout=30).json()["data"]
    except Exception as e:
        print(f"[watch] cap check err {str(e)[:60]} ({time.time()-T0:.0f}s)", flush=True); return {}
    out = {}
    for t in TIERS:
        info = c.get(t)
        if info:
            regs = [r["name"] for r in info.get("regions_with_capacity_available", [])]
            if regs: out[t] = regs
    return out


print(f"[watch] probe7b opt350m: waiting for non-a10 capacity {TIERS}", flush=True)
fired = False
while time.time() - T0 < 8 * 3600:
    avail = cap_regions()
    if avail:
        print(f"[watch] CAPACITY {avail} -> firing launcher ({time.time()-T0:.0f}s)", flush=True)
        env = dict(os.environ, GPU_PREF=PREF)
        logf = open(HERE / "p7b_fire.log", "w")
        p = subprocess.Popen([sys.executable, "lambda_launch_generic.py", "probe7b", "opt350m"],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
        iid = None
        for _ in range(60):
            if p.poll() is not None:
                break
            try:
                m = re.search(r"instance_id=([a-f0-9]+)", open(HERE / "p7b_fire.log").read())
                if m: iid = m.group(1); break
            except Exception: pass
            time.sleep(3)
        if iid:
            print(f"[watch] LAUNCHED {iid} -> arming killswitch; launcher running to completion", flush=True)
            subprocess.Popen([sys.executable, "killswitch_smart.py", iid], cwd=str(HERE))
            p.wait()
            print(f"[watch] launcher exited rc={p.returncode} ({time.time()-T0:.0f}s); DONE", flush=True)
            fired = True
            break
        else:
            p.wait()
            print(f"[watch] no id (rc={p.returncode}, capacity race) -> retry ({time.time()-T0:.0f}s)", flush=True)
    time.sleep(240)
if not fired:
    print(f"[watch] GAVE UP after {(time.time()-T0)/3600:.1f}h with no capacity", flush=True)
