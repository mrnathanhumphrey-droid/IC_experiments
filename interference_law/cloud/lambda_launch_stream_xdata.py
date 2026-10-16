"""
lambda_launch_ic_dose.py -- run the in-context dose-response sweep on Lambda (NOT the user's local box).
Uploads the two tested probe files (probe_stream_compose.py machinery + stream_xdata_lambda.py), runs the batched
dose curve on ONE A100, pulls back out/stream_xdata.json + log, ALWAYS terminates. Reuses ML-CF lane + Bearer auth +
scp-hang fix + watchdog + always-terminate from lambda_launch_alibi_tier1.py. a100/h100/gh200 only, NEVER a10.
"""
import os, sys, subprocess, time, random
from pathlib import Path
import requests

PD = Path("D:/physics_detector")
SSH_PRIV = PD / "cloud" / "ssh" / "ml_cf_ed25519"
SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
IL = Path("D:/IC_experiments/interference_law")
SRCS = [IL / "probe_stream_compose.py", IL / "stream_xdata_lambda.py"]
LOCAL_OUT = IL / "out"
RESULT = "stream_xdata.json"; LOG = "stream_xd_lambda.log"
PREF = os.environ.get("GPU_PREF", "gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_h100_sxm5,gpu_1x_gh200").split(",")
RUN_ENV = os.environ.get("RUN_ENV", "SC_PSTEP=60 SC_NSET=16 SC_NORD=8")


def key():
    for line in open(PD / ".env"):
        if line.startswith("CLOUD_LAMBDA="):
            return line.split("=", 1)[1].strip()
    print("[L] CLOUD_LAMBDA missing", flush=True); sys.exit(1)


H = {"Authorization": "Bearer " + key()}


def _req(method, p, body=None, tries=10):
    last = None
    for i in range(tries):
        try:
            r = requests.request(method, API + p, headers=H, json=body, timeout=30)
            if (500 <= r.status_code < 600) or r.status_code == 429:
                last = f"{r.status_code} {r.text[:120]}"; print(f"[L] transient {last} ({i+1}/{tries})", flush=True); time.sleep(20 if r.status_code == 429 else 8); continue
            if r.status_code >= 400:
                print(f"[L] API err {r.status_code}: {r.text}", flush=True); r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = str(e)[:120]; print(f"[L] transient net {last} ({i+1}/{tries})", flush=True); time.sleep(8)
    raise RuntimeError(f"API {method} {p} failed: {last}")


def get(p): return _req("GET", p)
def post(p, b): return _req("POST", p, b)


_KA = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8", "-o", "ConnectTimeout=15"]


def ssh(ip, cmd, _timeout=None, **kw):
    return subprocess.run(["ssh", "-i", str(SSH_PRIV)] + _KA + [f"ubuntu@{ip}", cmd], timeout=_timeout, **kw)


def scp(src, dst, tries=3):
    for i in range(tries):
        try:
            if subprocess.run(["scp", "-i", str(SSH_PRIV)] + _KA + [src, dst], timeout=1200).returncode == 0:
                return 0
            print(f"[L] scp rc!=0 ({i+1}/{tries})", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[L] scp stall ({i+1}/{tries})", flush=True)
        time.sleep(8)
    return 1


def pick():
    d = get("/instance-types")["data"]
    for name in PREF:
        t = d.get(name)
        if t and t.get("regions_with_capacity_available"):
            return name, t["regions_with_capacity_available"][0]["name"], t["instance_type"]["price_cents_per_hour"] / 100
    print("[L] NO CAPACITY", flush=True); sys.exit(2)


def main():
    time.sleep(random.uniform(0, 10))
    name, region, pr = pick()
    print(f"[L:xd] picked {name} @ {region} ${pr:.2f}/hr", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name,
                                               "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L:xd] instance_id={iid}  (manual teardown if I go dark)", flush=True)
    try:
        t0 = time.time(); ip = None; last = None
        while time.time() - t0 < 900:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last: print(f"[L:xd] status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
            if st == "active" and ip:
                try:
                    r = subprocess.run(["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
                                        "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "echo ok"], capture_output=True, text=True, timeout=15)
                    if r.returncode == 0 and "ok" in r.stdout: break
                except subprocess.TimeoutExpired:
                    pass
            time.sleep(15)
        else:
            raise TimeoutError("no active+ssh in 900s")
        print(f"[L:xd] SSH up at {ip} ({time.time()-t0:.0f}s)", flush=True)

        for s in SRCS:
            assert scp(str(s), f"ubuntu@{ip}:/home/ubuntu/{s.name}") == 0, f"scp {s.name}"
        print(f"[L:xd] uploaded {len(SRCS)} files ({time.time()-t0:.0f}s); pip install...", flush=True)
        ssh(ip, "pip install -q --upgrade pillow; python -c 'import torchvision' 2>/dev/null || pip install -q torchvision")   # use stack torch+torchvision; don't --upgrade torchvision (torch mismatch risk)
        print(f"[L:xd] running ({RUN_ENV}) ({time.time()-t0:.0f}s)...", flush=True)
        try:
            rc = ssh(ip, f"bash -c 'set -o pipefail; cd /home/ubuntu && {RUN_ENV} python stream_xdata_lambda.py 2>&1 | tee xd.log'",
                     _timeout=21600).returncode
        except subprocess.TimeoutExpired:
            print(f"[L:xd] WATCHDOG 6h -- tearing down", flush=True); rc = -1
        print(f"[L:xd] run rc={rc} ({time.time()-t0:.0f}s)", flush=True)

        ok = scp(f"ubuntu@{ip}:/home/ubuntu/out/{RESULT}", str(LOCAL_OUT / RESULT)) == 0   # base success on scp rc, not stale local file
        scp(f"ubuntu@{ip}:/home/ubuntu/xd.log", str(LOCAL_OUT / LOG))
        print(f"[L:xd] RESULT LANDED: {LOCAL_OUT/RESULT}" if ok else "[L:xd] WARN no result pulled (scp failed)", flush=True)
    finally:
        print(f"[L:xd] terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L:xd] terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L:xd] TERMINATE FAILED ({e}) -- MANUAL terminate {iid}", flush=True)


if __name__ == "__main__":
    main()
