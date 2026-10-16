"""
lambda_launch_plastic.py -- launcher for P2 PLASTIC-regime forward-transfer geometry (plastic_transfer_lambda.py).
Ships interference_law/*.py + PREREG_P2_PLASTIC, runs on ONE GPU (a10-first; pythia-1.4b bf16 fits 24GB), pulls back
out/plastic_transfer.json + out/plastic_rows.jsonl (incremental) + log, terminates, ledgers. Reuses ML-CF lane + Bearer auth.
SMOKE=1 via RUN_ENV for the cheap end-to-end validation before the real spend.
"""
import os, sys, subprocess, time, random, tarfile, glob, datetime
from pathlib import Path
import requests

PD = Path("D:/physics_detector")
SSH_PRIV = PD / "cloud" / "ssh" / "ml_cf_ed25519"
SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
ILAW = Path("D:/IC_experiments/interference_law")
LOCAL_OUT = ILAW / "out"
LEDGER = Path("D:/IC_experiments/lambda_cost_ledger.csv")
RESULT = os.environ.get("RESULT", "plastic_transfer.json"); ROWS = os.environ.get("ROWS", "plastic_rows.jsonl")
LOG = os.environ.get("LOGNAME", "plastic.log"); SCRIPT = os.environ.get("SCRIPT", "plastic_transfer_lambda.py")
PIP = os.environ.get("PIP", "pip install -q 'transformers==4.46.0' 'peft==0.13.2' 'datasets==2.21.0' accelerate 'Pillow<11' 2>/dev/null; "
                            "python -c 'import torch,transformers,peft,datasets;print(\"pins\",transformers.__version__,peft.__version__,datasets.__version__)'")
WATCHDOG = int(os.environ.get("WATCHDOG", "1800"))
PREF = os.environ.get("GPU_PREF", "gpu_1x_a10,gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_h100_sxm5,gpu_1x_gh200").split(",")
RUN_ENV = os.environ.get("RUN_ENV", "SMOKE=1")
TARBALL = LOCAL_OUT / "_plastic_ship.tar.gz"


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
    print("[L] NO CAPACITY in PREF", flush=True); sys.exit(2)


def build_tarball():
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(ILAW / "*.py"))) + [str(ILAW / "PREREG_P2_PLASTIC_2026_07_26.md")]
    with tarfile.open(TARBALL, "w:gz") as tf:
        for f in files:
            if Path(f).exists(): tf.add(f, arcname=Path(f).name)
    print(f"[L:pl] tarball {TARBALL.name} with {len([f for f in files if Path(f).exists()])} files", flush=True)


def main():
    build_tarball()
    time.sleep(random.uniform(0, 8))
    name, region, pr = pick()
    print(f"[L:pl] picked {name} @ {region} ${pr:.2f}/hr  RUN_ENV='{RUN_ENV}' wd={WATCHDOG}s", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name,
                                               "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L:pl] instance_id={iid}  (manual teardown if I go dark)", flush=True)
    t0 = time.time(); rc = -99; ip = None
    try:
        last = None
        while time.time() - t0 < 900:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last: print(f"[L:pl] status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
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
        print(f"[L:pl] SSH up at {ip} ({time.time()-t0:.0f}s)", flush=True)

        ssh(ip, "rm -rf /home/ubuntu/ilaw && mkdir -p /home/ubuntu/ilaw/out")
        assert scp(str(TARBALL), f"ubuntu@{ip}:/home/ubuntu/ilaw/ship.tar.gz") == 0, "scp tarball"
        ssh(ip, "cd /home/ubuntu/ilaw && tar xzf ship.tar.gz")
        print(f"[L:pl] uploaded+extracted ({time.time()-t0:.0f}s); pip...", flush=True)
        ssh(ip, PIP)
        print(f"[L:pl] running {SCRIPT} ({RUN_ENV}) wd={WATCHDOG}s ({time.time()-t0:.0f}s)...", flush=True)
        try:
            rc = ssh(ip, f"bash -c 'set -o pipefail; cd /home/ubuntu/ilaw && {RUN_ENV} python {SCRIPT} 2>&1 | tee pl.log'",
                     _timeout=WATCHDOG).returncode
        except subprocess.TimeoutExpired:
            print(f"[L:pl] WATCHDOG {WATCHDOG}s -- pulling partial + tearing down", flush=True); rc = -1
        print(f"[L:pl] run rc={rc} ({time.time()-t0:.0f}s)", flush=True)

        scp(f"ubuntu@{ip}:/home/ubuntu/ilaw/out/{RESULT}", str(LOCAL_OUT / RESULT))
        scp(f"ubuntu@{ip}:/home/ubuntu/ilaw/out/{ROWS}", str(LOCAL_OUT / ROWS))
        scp(f"ubuntu@{ip}:/home/ubuntu/ilaw/pl.log", str(LOCAL_OUT / LOG))
        got = (LOCAL_OUT / RESULT).exists() or (LOCAL_OUT / ROWS).exists()
        print(f"[L:pl] RESULT LANDED: {LOCAL_OUT/RESULT} rows={LOCAL_OUT/ROWS}" if got else "[L:pl] WARN nothing pulled", flush=True)
    finally:
        mins = (time.time() - t0) / 60
        print(f"[L:pl] terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L:pl] terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L:pl] TERMINATE FAILED ({e}) -- MANUAL terminate {iid}", flush=True)
        try:
            landed = (LOCAL_OUT / RESULT).exists()
            note = f"P2 plastic transfer geometry RUN_ENV='{RUN_ENV}' rc={rc} landed={landed}"
            with open(LEDGER, "a") as f:
                f.write(f"{datetime.date.today()},{iid},{name},{pr*100:.0f},{mins:.0f},{pr*mins/60:.2f},\"{note}\"\n")
            print(f"[L:pl] ledger += {mins:.0f}min ${pr*mins/60:.2f}", flush=True)
        except Exception as e:
            print(f"[L:pl] ledger write failed: {e}", flush=True)


if __name__ == "__main__":
    main()
