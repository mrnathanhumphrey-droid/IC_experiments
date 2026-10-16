"""cf_cascade Lambda launcher — #2 DEVELOPMENTAL EXPERIMENT (childhood->sliver vs yank; 3 arms x 7 seeds, real training).
Heaviest run of the program. Uploads sst2 ckpt + agnews/subj jsons (emotion/dbpedia/yahoo/newscat from HF). Hard finally-terminate. Key = ML-CF.
"""
import os, sys, subprocess, time
from pathlib import Path
import requests

PD = Path("D:/physics_detector")
SSH_PRIV = PD / "cloud" / "ssh" / "ml_cf_ed25519"
SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
CF = Path("D:/IC_experiments/cf_cascade"); G = CF / "gateA"
PREF = ["gpu_1x_a100_sxm4", "gpu_1x_a100", "gpu_1x_h100_sxm5", "gpu_1x_gh200", "gpu_1x_a10"]
SCRIPT = "dev_experiment2b_lambda.py"; RESULT = "dev_experiment2b.json"
UPLOADS = ["pythia410m_sst2_subsetmin.pt"]
JSONS = ["low_agnews.json", "medium_subj.json"]


def key():
    for line in open(PD / ".env"):
        if line.startswith("CLOUD_LAMBDA="):
            return line.split("=", 1)[1].strip()
    print("[L] CLOUD_LAMBDA missing", flush=True); sys.exit(1)


H = {"Authorization": "Bearer " + key()}


def _req(method, p, body=None, tries=6):
    last = None
    for i in range(tries):
        try:
            r = requests.request(method, API + p, headers=H, json=body, timeout=30)
            if 500 <= r.status_code < 600:
                last = f"{r.status_code} {r.text[:120]}"; print(f"[L] transient {last} (try {i+1}/{tries})", flush=True); time.sleep(8); continue
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
    name, region, pr = pick()
    print(f"[L] picked {name} @ {region} ${pr:.2f}/hr", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name,
                                               "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L] instance_id={iid}  (manual teardown: terminate this id if I go dark)", flush=True)
    try:
        t0 = time.time(); ip = None; last = None
        while time.time() - t0 < 600:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last:
                print(f"[L] status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
            if st == "active" and ip:
                r = subprocess.run(["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
                                    "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "echo ok"], capture_output=True, text=True, timeout=15)
                if r.returncode == 0 and "ok" in r.stdout: break
            time.sleep(15)
        else:
            raise TimeoutError("instance never reached active+ssh in 600s")
        print(f"[L] SSH up at {ip} ({time.time()-t0:.0f}s)", flush=True)

        ssh(ip, "mkdir -p /home/ubuntu/gateA/overlap_axis /home/ubuntu/hf_cache")
        assert scp(str(CF / "cloud" / SCRIPT), f"ubuntu@{ip}:/home/ubuntu/{SCRIPT}") == 0, "scp script"
        print("[L] uploading checkpoint + jsons...", flush=True)
        for f in UPLOADS:
            assert scp(str(G / f), f"ubuntu@{ip}:/home/ubuntu/gateA/") == 0, f"scp {f}"
        for j in JSONS:
            assert scp(str(G / "overlap_axis" / j), f"ubuntu@{ip}:/home/ubuntu/gateA/overlap_axis/") == 0, f"scp {j}"
        print(f"[L] upload done ({time.time()-t0:.0f}s)", flush=True)

        print("[L] installing transformers/datasets + upgrading Pillow...", flush=True)
        ssh(ip, "pip install -q --upgrade transformers datasets accelerate Pillow")

        print(f"[L] running {SCRIPT} ({time.time()-t0:.0f}s)...", flush=True)
        rc = ssh(ip, f"bash -c 'set -o pipefail; cd /home/ubuntu && python {SCRIPT} 2>&1 | tee pl.log'").returncode
        print(f"[L] run rc={rc} ({time.time()-t0:.0f}s)", flush=True)

        print("[L] pulling result + log...", flush=True)
        scp(f"ubuntu@{ip}:/home/ubuntu/gateA/{RESULT}", str(G / RESULT))
        scp(f"ubuntu@{ip}:/home/ubuntu/gateA/dev_experiment2b.npz", str(G / "dev_experiment2b.npz"))
        scp(f"ubuntu@{ip}:/home/ubuntu/pl.log", str(CF / "cloud" / "dev_experiment2b_lambda.log"))
        print(f"[L] RESULT LANDED: gateA/{RESULT}" if (G / RESULT).exists() else "[L] WARN: no result pulled", flush=True)
    finally:
        print(f"[L] terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L] terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L] TERMINATE FAILED ({e}) -- MANUAL: terminate instance {iid}", flush=True)


if __name__ == "__main__":
    main()
