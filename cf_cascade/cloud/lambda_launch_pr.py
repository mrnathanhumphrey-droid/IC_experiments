"""
lambda_launch_pr.py -- launcher for #3 subspace-dimensionality PR-steering (pr_steering_lambda.py).
Ships the single self-contained harness, runs on ONE GPU (a10-first; gemma-2-2b/qwen-1.5b bf16 fit 24GB), pulls back
the result json + log, terminates, ledgers. Reuses ML-CF lane + Bearer auth (mirrors lambda_launch_plastic).
HF_TOKEN passed to the remote run env (for gemma gate); NOT printed/logged. SMOKE=1 via RUN_ENV for cheap de-risk.
Env knobs: SCRIPT, RESULT, LOGNAME, RUN_ENV, WATCHDOG, GPU_PREF, PIP.
"""
import os, sys, subprocess, time, random, datetime
from pathlib import Path
import requests

PD = Path("D:/physics_detector")
SSH_PRIV = PD / "cloud" / "ssh" / "ml_cf_ed25519"
SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
CF = Path("D:/IC_experiments/cf_cascade"); G = CF / "gateA"
LEDGER = Path("D:/IC_experiments/lambda_cost_ledger.csv")
SCRIPT = os.environ.get("SCRIPT", "pr_steering_lambda.py")
RESULT = os.environ.get("RESULT", "pr_steering_gemma-2-2b.json")
LOG = os.environ.get("LOGNAME", "pr_steering_lambda.log")
RUN_ENV = os.environ.get("RUN_ENV", "SMOKE=1")
WATCHDOG = int(os.environ.get("WATCHDOG", "1800"))
PREF = os.environ.get("GPU_PREF", "gpu_1x_a10,gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_h100_sxm5,gpu_1x_gh200").split(",")
PIP = os.environ.get("PIP", "pip install -q --user --upgrade 'transformers==4.46.0' accelerate sentencepiece protobuf; "
                            "python -c 'import torch,transformers;print(\"tf\",torch.__version__,transformers.__version__)'")


def _envval(k):
    for line in open(PD / ".env"):
        if line.startswith(k + "="):
            return line.split("=", 1)[1].strip()
    return None


API_KEY = _envval("CLOUD_LAMBDA")
HF_TOKEN = _envval("HF_TOKEN") or ""
if not API_KEY:
    print("[L] CLOUD_LAMBDA missing", flush=True); sys.exit(1)
H = {"Authorization": "Bearer " + API_KEY}


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


def main():
    time.sleep(random.uniform(0, 8))
    name, region, pr = pick()
    print(f"[L:pr] picked {name} @ {region} ${pr:.2f}/hr  RUN_ENV='{RUN_ENV}' wd={WATCHDOG}s hf_tok={'yes' if HF_TOKEN else 'NO'}", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name,
                                               "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L:pr] instance_id={iid}  (manual teardown if I go dark)", flush=True)
    t0 = time.time(); rc = -99; ip = None
    try:
        last = None
        while time.time() - t0 < 900:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last: print(f"[L:pr] status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
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
        print(f"[L:pr] SSH up at {ip} ({time.time()-t0:.0f}s)", flush=True)

        ssh(ip, "rm -rf /home/ubuntu/pr && mkdir -p /home/ubuntu/pr/gateA")
        assert scp(str(CF / SCRIPT), f"ubuntu@{ip}:/home/ubuntu/pr/{SCRIPT}") == 0, "scp script"
        print(f"[L:pr] uploaded ({time.time()-t0:.0f}s); pip...", flush=True)
        ssh(ip, PIP)
        print(f"[L:pr] running {SCRIPT} ({RUN_ENV}) wd={WATCHDOG}s ({time.time()-t0:.0f}s)...", flush=True)
        # HF_TOKEN + HF_HOME set inline (token not printed by us). OUTDIR -> /home/ubuntu/pr/gateA.
        remote = (f"bash -c 'set -o pipefail; cd /home/ubuntu/pr && export HF_TOKEN={HF_TOKEN} HF_HOME=/home/ubuntu/hf_cache "
                  f"OUTDIR=/home/ubuntu/pr/gateA; {RUN_ENV} python {SCRIPT} 2>&1 | tee pl.log'")
        try:
            rc = ssh(ip, remote, _timeout=WATCHDOG).returncode
        except subprocess.TimeoutExpired:
            print(f"[L:pr] WATCHDOG {WATCHDOG}s -- pulling partial + tearing down", flush=True); rc = -1
        print(f"[L:pr] run rc={rc} ({time.time()-t0:.0f}s)", flush=True)

        scp(f"ubuntu@{ip}:/home/ubuntu/pr/gateA/{RESULT}", str(G / RESULT))
        scp(f"ubuntu@{ip}:/home/ubuntu/pr/pl.log", str(CF / "cloud" / LOG))
        got = (G / RESULT).exists()
        print(f"[L:pr] RESULT LANDED: {G/RESULT}" if got else "[L:pr] WARN nothing pulled", flush=True)
    finally:
        mins = (time.time() - t0) / 60
        print(f"[L:pr] terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L:pr] terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L:pr] TERMINATE FAILED ({e}) -- MANUAL terminate {iid}", flush=True)
        try:
            note = f"#3 PR-steering RUN_ENV='{RUN_ENV}' rc={rc} landed={(G/RESULT).exists()}"
            with open(LEDGER, "a") as f:
                f.write(f"{datetime.date.today()},{iid},{name},{pr*100:.0f},{mins:.0f},{pr*mins/60:.2f},\"{note}\"\n")
            print(f"[L:pr] ledger += {mins:.0f}min ${pr*mins/60:.2f}", flush=True)
        except Exception as e:
            print(f"[L:pr] ledger write failed: {e}", flush=True)


if __name__ == "__main__":
    main()
