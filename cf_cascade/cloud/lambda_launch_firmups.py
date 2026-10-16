"""cf_cascade Lambda launcher — FIRM-UPS (TEST#1 k-sweep + 2nd/3rd substrate) on a single A100.
Non-interactive, hard finally-terminate. Same pattern as lambda_launch_cf.py.
"""
import os, sys, subprocess, time, json
from pathlib import Path
import requests

PD = Path("D:/physics_detector")
SSH_DIR = PD / "cloud" / "ssh"
SSH_PRIV = SSH_DIR / "ml_cf_ed25519"
SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
CF = Path("D:/IC_experiments/cf_cascade")
G = CF / "gateA"
PREF = ["gpu_1x_a100_sxm4", "gpu_1x_a100", "gpu_1x_h100_sxm5", "gpu_1x_gh200", "gpu_1x_a10"]
TASKB_JSONS = ["low_agnews.json", "high_yelp.json", "medium_subj.json"]


def key():
    for line in open(PD / ".env"):
        if line.startswith("CLOUD_LAMBDA="):
            return line.split("=", 1)[1].strip()
    print("[L] CLOUD_LAMBDA missing", flush=True); sys.exit(1)


H = {"Authorization": "Bearer " + key()}


def get(p):
    r = requests.get(API + p, headers=H, timeout=30); r.raise_for_status(); return r.json()


def post(p, b):
    r = requests.post(API + p, headers=H, json=b, timeout=30)
    if r.status_code >= 400:
        print(f"[L] API err {r.status_code}: {r.text}", flush=True); r.raise_for_status()
    return r.json()


def ssh(ip, cmd, **kw):
    return subprocess.run(
        ["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", f"ubuntu@{ip}", cmd], **kw)


def scp(src, dst):
    return subprocess.run(
        ["scp", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", src, dst]).returncode


def pick():
    d = get("/instance-types")["data"]
    for name in PREF:
        t = d.get(name)
        if not t:
            continue
        regs = t.get("regions_with_capacity_available", [])
        if regs:
            pr = t["instance_type"]["price_cents_per_hour"] / 100
            return name, regs[0]["name"], pr
    print("[L] NO CAPACITY across preferred types", flush=True); sys.exit(2)


def main():
    name, region, pr = pick()
    print(f"[L] picked {name} @ {region} ${pr:.2f}/hr", flush=True)
    resp = post("/instance-operations/launch",
                {"region_name": region, "instance_type_name": name,
                 "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})
    iid = resp["data"]["instance_ids"][0]
    print(f"[L] instance_id={iid}  (manual teardown: terminate this id if I go dark)", flush=True)
    try:
        t0 = time.time(); ip = None; last = None
        while time.time() - t0 < 600:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last:
                print(f"[L] status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
            if st == "active" and ip:
                r = subprocess.run(
                    ["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
                     "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "echo ok"],
                    capture_output=True, text=True, timeout=15)
                if r.returncode == 0 and "ok" in r.stdout:
                    break
            time.sleep(15)
        else:
            raise TimeoutError("instance never reached active+ssh in 600s")
        print(f"[L] SSH up at {ip} ({time.time()-t0:.0f}s)", flush=True)

        ssh(ip, "mkdir -p /home/ubuntu/gateA/overlap_axis /home/ubuntu/hf_cache")
        assert scp(str(CF / "cloud" / "firmups_lambda.py"), f"ubuntu@{ip}:/home/ubuntu/firmups_lambda.py") == 0, "scp script"
        print("[L] uploading 1.6GB checkpoint...", flush=True)
        assert scp(str(G / "pythia410m_sst2_subsetmin.pt"), f"ubuntu@{ip}:/home/ubuntu/gateA/") == 0, "scp ckpt"
        for j in TASKB_JSONS:
            assert scp(str(G / "overlap_axis" / j), f"ubuntu@{ip}:/home/ubuntu/gateA/overlap_axis/") == 0, f"scp {j}"
        print(f"[L] upload done ({time.time()-t0:.0f}s)", flush=True)

        print("[L] installing transformers/datasets + upgrading Pillow...", flush=True)
        ssh(ip, "pip install -q --upgrade transformers datasets accelerate Pillow")

        print(f"[L] running firmups_lambda.py ({time.time()-t0:.0f}s)...", flush=True)
        rc = ssh(ip, "bash -c 'set -o pipefail; cd /home/ubuntu && python firmups_lambda.py 2>&1 | tee fu.log'").returncode
        print(f"[L] run rc={rc} ({time.time()-t0:.0f}s)", flush=True)

        print("[L] pulling result + log...", flush=True)
        scp(f"ubuntu@{ip}:/home/ubuntu/gateA/firmups.json", str(G / "firmups.json"))
        scp(f"ubuntu@{ip}:/home/ubuntu/fu.log", str(CF / "cloud" / "fu_lambda.log"))
        if (G / "firmups.json").exists():
            print("[L] RESULT LANDED: gateA/firmups.json", flush=True)
        else:
            print("[L] WARN: no result file pulled", flush=True)
    finally:
        print(f"[L] terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]})
            print(f"[L] terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L] TERMINATE FAILED ({e}) -- MANUAL: terminate instance {iid} on Lambda dashboard", flush=True)


if __name__ == "__main__":
    main()
