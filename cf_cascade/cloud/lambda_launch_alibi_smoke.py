"""launcher for the alibi break-n=1 SMOKE (alibi_smoke_loadcheck.py). Pins transformers==4.46.0 (Jais custom code needs
find_pruneable_heads_and_indices; also supports Qwen2+BLOOM). A10-first, 30-min watchdog, always-terminate, ledger. ML-CF lane."""
import os, sys, subprocess, time, random, datetime
from pathlib import Path
import requests

PD = Path("D:/physics_detector")
SSH_PRIV = PD / "cloud" / "ssh" / "ml_cf_ed25519"; SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
HERE = Path("D:/IC_experiments/cf_cascade/cloud")
LOCAL_OUT = Path("D:/IC_experiments/cf_cascade/out"); LOCAL_OUT.mkdir(parents=True, exist_ok=True)
LEDGER = Path("D:/IC_experiments/lambda_cost_ledger.csv")
SCRIPT = "alibi_smoke_loadcheck.py"; RESULT = "alibi_smoke.json"; LOG = "alibi_smoke.log"
PIP = ("pip install -q 'transformers==4.46.0' 'tokenizers>=0.20,<0.21' accelerate datasets sentencepiece 'protobuf==3.20.3' einops 2>/dev/null; "
       "python -c 'import transformers,torch;print(\"pins\",transformers.__version__,torch.__version__)'")
WATCHDOG = int(os.environ.get("WATCHDOG", "1800"))
PREF = os.environ.get("GPU_PREF", "gpu_1x_a10,gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_h100_sxm5,gpu_1x_gh200").split(",")


def key():
    for line in open(PD / ".env"):
        if line.startswith("CLOUD_LAMBDA="): return line.split("=", 1)[1].strip()
    print("[L] CLOUD_LAMBDA missing", flush=True); sys.exit(1)


H = {"Authorization": "Bearer " + key()}


def _req(method, p, body=None, tries=10):
    last = None
    for i in range(tries):
        try:
            r = requests.request(method, API + p, headers=H, json=body, timeout=30)
            if (500 <= r.status_code < 600) or r.status_code == 429:
                last = f"{r.status_code} {r.text[:120]}"; print(f"[L] transient {last} ({i+1}/{tries})", flush=True); time.sleep(20 if r.status_code == 429 else 8); continue
            if r.status_code >= 400: print(f"[L] API err {r.status_code}: {r.text}", flush=True); r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = str(e)[:120]; print(f"[L] transient net {last} ({i+1}/{tries})", flush=True); time.sleep(8)
    raise RuntimeError(f"API {method} {p} failed: {last}")


def get(p): return _req("GET", p)
def post(p, b): return _req("POST", p, b)
_KA = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8", "-o", "ConnectTimeout=15"]
def ssh(ip, cmd, _timeout=None, **kw): return subprocess.run(["ssh", "-i", str(SSH_PRIV)] + _KA + [f"ubuntu@{ip}", cmd], timeout=_timeout, **kw)
def scp(src, dst, tries=3):
    for i in range(tries):
        try:
            if subprocess.run(["scp", "-i", str(SSH_PRIV)] + _KA + [src, dst], timeout=900).returncode == 0: return 0
        except subprocess.TimeoutExpired: print(f"[L] scp stall ({i+1})", flush=True)
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
    time.sleep(random.uniform(0, 6))
    name, region, pr = pick(); print(f"[L:as] {name} @ {region} ${pr:.2f}/hr wd={WATCHDOG}s", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name, "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L:as] instance_id={iid}", flush=True)
    t0 = time.time(); rc = -99; ip = None
    try:
        last = None
        while time.time() - t0 < 900:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last: print(f"[L:as] status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
            if st == "active" and ip:
                try:
                    if subprocess.run(["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "echo ok"], capture_output=True, text=True, timeout=15).returncode == 0: break
                except subprocess.TimeoutExpired: pass
            time.sleep(15)
        else: raise TimeoutError("no active+ssh in 900s")
        print(f"[L:as] SSH up {ip} ({time.time()-t0:.0f}s)", flush=True)
        ssh(ip, "mkdir -p /home/ubuntu/gateA")
        assert scp(str(HERE / SCRIPT), f"ubuntu@{ip}:/home/ubuntu/{SCRIPT}") == 0, "scp script"
        ssh(ip, PIP); print(f"[L:as] pip done ({time.time()-t0:.0f}s); running...", flush=True)
        try:
            rc = ssh(ip, f"bash -c 'set -o pipefail; cd /home/ubuntu && python {SCRIPT} 2>&1 | tee as.log'", _timeout=WATCHDOG).returncode
        except subprocess.TimeoutExpired:
            print(f"[L:as] WATCHDOG {WATCHDOG}s -> pulling+teardown", flush=True); rc = -1
        print(f"[L:as] rc={rc} ({time.time()-t0:.0f}s)", flush=True)
        scp(f"ubuntu@{ip}:/home/ubuntu/gateA/{RESULT}", str(LOCAL_OUT / RESULT))
        scp(f"ubuntu@{ip}:/home/ubuntu/as.log", str(LOCAL_OUT / LOG))
        print(f"[L:as] LANDED {LOCAL_OUT/RESULT}" if (LOCAL_OUT / RESULT).exists() else "[L:as] WARN nothing pulled", flush=True)
    finally:
        mins = (time.time() - t0) / 60
        try: post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L:as] terminated {iid}", flush=True)
        except Exception as e: print(f"[L:as] TERMINATE FAIL ({e}) -- MANUAL {iid}", flush=True)
        try:
            with open(LEDGER, "a") as f:
                f.write(f"{datetime.date.today()},{iid},{name},{pr*100:.0f},{mins:.0f},{pr*mins/60:.2f},\"alibi break-n1 SMOKE rc={rc} landed={(LOCAL_OUT/RESULT).exists()}\"\n")
            print(f"[L:as] ledger += {mins:.0f}min ${pr*mins/60:.2f}", flush=True)
        except Exception as e: print(f"[L:as] ledger fail {e}", flush=True)


if __name__ == "__main__":
    main()
