"""
lambda_launch_ic_dose.py -- run the in-context dose-response sweep on Lambda (NOT the user's local box).
Uploads the two tested probe files (probe_representational_law.py machinery + probe_phase_rank.py), runs the batched
dose curve on ONE A100, pulls back out/phase_rank.json + log, ALWAYS terminates. Reuses ML-CF lane + Bearer auth +
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
SRCS = [IL / "probe_representational_law.py", IL / "probe_phase_rank.py"]
LOCAL_OUT = IL / "out"
RESULT = "phase_rank.json"; LOG = "phase_rank_lambda.log"
PREF = os.environ.get("GPU_PREF", "gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_h100_sxm5,gpu_1x_gh200").split(",")
RUN_ENV = os.environ.get("RUN_ENV", "PR_RANKS=1,2,4,8,16,32 PR_NB=6 PR_BSTEPS=1000")


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
    print(f"[L:pr] picked {name} @ {region} ${pr:.2f}/hr", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name,
                                               "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L:pr] instance_id={iid}  (manual teardown if I go dark)", flush=True)
    try:
        t0 = time.time(); ip = None; last = None
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

        for s in SRCS:
            assert scp(str(s), f"ubuntu@{ip}:/home/ubuntu/{s.name}") == 0, f"scp {s.name}"
        print(f"[L:pr] uploaded {len(SRCS)} files ({time.time()-t0:.0f}s); pip install...", flush=True)
        ssh(ip, "pip install -q --upgrade pillow datasets; python -c 'import torchvision' 2>/dev/null || pip install -q torchvision")   # use stack torch+torchvision; don't --upgrade torchvision (torch mismatch risk)
        # DEAD-MAN self-terminate: a DETACHED process on the instance kills it after DEADMAN_SEC no matter what happens LOCALLY
        # (if this launcher dies, the finally below never runs -> orphan/cost-bleed; the dead-man bounds the bleed).
        DEADMAN_SEC = int(os.environ.get("DEADMAN_SEC", "9000"))   # 150 min hard cap for a ~90 min job
        st = ("import requests\n"
              f"requests.post('https://cloud.lambda.ai/api/v1/instance-operations/terminate', "
              f"headers={{'Authorization':'Bearer {key()}'}}, json={{'instance_ids':['{iid}']}}, timeout=30)\n")
        ssh(ip, f"cat > /home/ubuntu/selfterm.py <<'PYEOF'\n{st}PYEOF")
        ssh(ip, f"nohup setsid bash -c 'sleep {DEADMAN_SEC}; python /home/ubuntu/selfterm.py' >/dev/null 2>&1 </dev/null &")
        print(f"[L:pr] DEAD-MAN armed: instance self-terminates in {DEADMAN_SEC}s if I die ({time.time()-t0:.0f}s)", flush=True)
        # run DETACHED so a local launcher death cannot kill the remote probe; poll for the DONE marker
        ssh(ip, f"nohup setsid bash -c 'cd /home/ubuntu && {RUN_ENV} python probe_phase_rank.py > pr.log 2>&1; touch /home/ubuntu/DONE' >/dev/null 2>&1 </dev/null &")
        print(f"[L:pr] probe DETACHED; polling ({time.time()-t0:.0f}s)...", flush=True)
        t_run = time.time(); MAXW = int(os.environ.get("MAXWAIT_SEC", "8400")); rc = -1
        while time.time() - t_run < MAXW:
            time.sleep(60)
            r = subprocess.run(["ssh", "-i", str(SSH_PRIV)] + _KA + [f"ubuntu@{ip}",
                 "test -f /home/ubuntu/DONE && echo FIN; tail -n 1 /home/ubuntu/pr.log 2>/dev/null"],
                 capture_output=True, text=True, timeout=30)
            out = (r.stdout or "").strip()
            tail = [l for l in out.splitlines() if l != "FIN"]
            if tail: print(f"    ...{tail[-1][-130:]} ({time.time()-t0:.0f}s)", flush=True)
            if "FIN" in out: rc = 0; print(f"[L:pr] probe FINISHED ({time.time()-t0:.0f}s)", flush=True); break
        else:
            print(f"[L:pr] poll hit MAXWAIT {MAXW}s -- pulling whatever exists", flush=True)

        ok = scp(f"ubuntu@{ip}:/home/ubuntu/out/{RESULT}", str(LOCAL_OUT / RESULT)) == 0   # base success on scp rc, not stale local file
        scp(f"ubuntu@{ip}:/home/ubuntu/pr.log", str(LOCAL_OUT / LOG))
        print(f"[L:pr] RESULT LANDED: {LOCAL_OUT/RESULT}" if ok else "[L:pr] WARN no result pulled (scp failed)", flush=True)
    finally:
        print(f"[L:pr] terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L:pr] terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L:pr] TERMINATE FAILED ({e}) -- MANUAL terminate {iid}", flush=True)


if __name__ == "__main__":
    main()
