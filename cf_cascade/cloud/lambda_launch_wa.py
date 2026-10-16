"""Lambda launcher for the w_A trainer (size-matrix feed). Args: MODEL TAG [GPU].
Fires wa_trainer_lambda.py MODEL TAG; pulls <TAG>_sst2_wA.pt + <TAG>_wa.json. Key=ML-CF. Hard finally-terminate.
GPU arg optional: 'a100' (default) or 'h100' to front-load H100 for big models.
"""
import os, sys, subprocess, time, random
from pathlib import Path
import requests

MODEL = sys.argv[1]; TAG = sys.argv[2]; GPU = sys.argv[3] if len(sys.argv) > 3 else "a100"
REVISION = sys.argv[4] if len(sys.argv) > 4 else "-"   # e.g. step16000 (Pythia length knob); '-' = default head
PD = Path("D:/physics_detector")
SSH_PRIV = PD / "cloud" / "ssh" / "ml_cf_ed25519"; SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
CF = Path("D:/IC_experiments/cf_cascade"); G = CF / "gateA"
PREF_A100 = ["gpu_1x_a100_sxm4", "gpu_1x_a100", "gpu_1x_h100_sxm5", "gpu_1x_gh200", "gpu_1x_a10"]
PREF_H100 = ["gpu_1x_h100_sxm5", "gpu_1x_a100_sxm4", "gpu_1x_a100", "gpu_1x_gh200"]
PREF = PREF_H100 if GPU == "h100" else PREF_A100
SCRIPT = "wa_trainer_lambda.py"
PULLS = [f"{TAG}_sst2_wA.pt", f"{TAG}_wa.json"]; LOG = f"wa_{TAG}.log"


def key():
    for line in open(PD / ".env"):
        if line.startswith("CLOUD_LAMBDA="): return line.split("=", 1)[1].strip()
    print("[L] CLOUD_LAMBDA missing", flush=True); sys.exit(1)


def hf_tok():
    for line in open(PD / ".env"):
        if line.startswith("HF_TOKEN="): return line.split("=", 1)[1].strip()
    return ""


TOK = hf_tok()
H = {"Authorization": "Bearer " + key()}


def _req(method, p, body=None, tries=10):
    last = None
    for i in range(tries):
        try:
            r = requests.request(method, API + p, headers=H, json=body, timeout=30)
            txt = r.text[:200]
            if (500 <= r.status_code < 600) or r.status_code == 429 or "1015" in txt or "rate limit" in txt.lower():
                last = f"{r.status_code} {txt[:80]}"; print(f"[L] transient {last} (try {i+1}/{tries})", flush=True); time.sleep(20); continue
            if r.status_code >= 400:
                print(f"[L] API err {r.status_code}: {r.text}", flush=True); r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = str(e)[:120]; print(f"[L] transient net err {last} (try {i+1}/{tries})", flush=True); time.sleep(8)
    raise RuntimeError(f"API {method} {p} failed after {tries} tries: {last}")


def get(p): return _req("GET", p)
def post(p, b): return _req("POST", p, b)
def ssh(ip, cmd, **kw):
    return subprocess.run(["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
                           "-o", "UserKnownHostsFile=/dev/null", f"ubuntu@{ip}", cmd], **kw)
def scp(src, dst):
    return subprocess.run(["scp", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
                           "-o", "UserKnownHostsFile=/dev/null", src, dst]).returncode
def pick():
    d = get("/instance-types")["data"]
    for name in PREF:
        t = d.get(name)
        if t and t.get("regions_with_capacity_available"):
            return name, t["regions_with_capacity_available"][0]["name"], t["instance_type"]["price_cents_per_hour"] / 100
    print("[L] NO CAPACITY", flush=True); sys.exit(2)


def main():
    _j = random.uniform(0, 20); print(f"[L] {TAG}: politeness jitter {_j:.1f}s", flush=True); time.sleep(_j)
    name, region, pr = pick()
    print(f"[L] {TAG}: picked {name} @ {region} ${pr:.2f}/hr", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name,
                                               "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L] {TAG}: instance_id={iid} (manual teardown if I go dark)", flush=True)
    try:
        t0 = time.time(); ip = None; last = None
        while time.time() - t0 < 900:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last:
                print(f"[L] {TAG}: status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
            if st == "active" and ip:
                try:
                    r = subprocess.run(["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
                                        "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "echo ok"], capture_output=True, text=True, timeout=15)
                    if r.returncode == 0 and "ok" in r.stdout: break
                except subprocess.TimeoutExpired:
                    print(f"[L] {TAG}: ssh probe hung (retrying) {time.time()-t0:.0f}s", flush=True)
            time.sleep(15)
        else:
            raise TimeoutError("instance never reached active+ssh in 900s")
        print(f"[L] {TAG}: SSH up at {ip} ({time.time()-t0:.0f}s)", flush=True)
        ssh(ip, "mkdir -p /home/ubuntu/gateA /home/ubuntu/hf_cache")
        assert scp(str(CF / "cloud" / SCRIPT), f"ubuntu@{ip}:/home/ubuntu/{SCRIPT}") == 0, "scp script"
        ssh(ip, "pip install -q --upgrade transformers datasets accelerate Pillow")
        print(f"[L] {TAG}: running ({time.time()-t0:.0f}s)...", flush=True)
        rc = ssh(ip, f"bash -c 'set -o pipefail; cd /home/ubuntu && HF_TOKEN={TOK} HUGGING_FACE_HUB_TOKEN={TOK} python {SCRIPT} {MODEL} {TAG} {REVISION} 2>&1 | tee wa.log'").returncode
        print(f"[L] {TAG}: run rc={rc} ({time.time()-t0:.0f}s)", flush=True)
        for f in PULLS:
            scp(f"ubuntu@{ip}:/home/ubuntu/gateA/{f}", str(G / f))
        scp(f"ubuntu@{ip}:/home/ubuntu/wa.log", str(CF / "cloud" / LOG))
        landed = [f for f in PULLS if (G / f).exists()]
        print(f"[L] {TAG}: LANDED {landed}" if landed else f"[L] {TAG}: WARN nothing pulled", flush=True)
    finally:
        print(f"[L] {TAG}: terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L] {TAG}: terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L] {TAG}: TERMINATE FAILED ({e}) -- MANUAL terminate {iid}", flush=True)


if __name__ == "__main__":
    main()
