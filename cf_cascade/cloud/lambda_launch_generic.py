import sys
"""Generic cf_cascade Lambda launcher. Usage: python lambda_launch_generic.py <probe> <cell>
  probe in {probe3, probe4, probe5}; cell in {opt350m, pythia410m}. Key = ML-CF. Mirrors lambda_launch_probe2."""
PROBES = {
    "probe3": ("probe3_dose_lambda.py",     "probe3_dose"),
    "probe4": ("probe4_dual_lambda.py",     "probe4_dual"),
    "probe5": ("probe5_residual_lambda.py", "probe5_residual"),
    "probe6": ("probe6_displacement_lambda.py", "probe6_displacement"),
    "probe7": ("probe7_displacement_lambda.py", "probe7_displacement"),
    "probe7b": ("probe7b_displacement_lambda.py", "probe7b_displacement"),
    "probe8": ("probe8_curvature_lambda.py", "probe8_curvature"),
    "probe8v3": ("probe8v3_curvature_lambda.py", "probe8v3_curvature"),
    "probe8v4": ("probe8v4_specificity_lambda.py", "probe8v4_specificity"),
    "probe9": ("probe9_aiming_lambda.py", "probe9_aiming"),
    "probe9b": ("probe9b_phase2lever_lambda.py", "probe9b_phase2lever"),
    "probe10": ("probe10_simrehearsal_lambda.py", "probe10_simrehearsal"),
    "probe10post": ("probe10post_dirsim_lambda.py", "probe10post_dirsim"),
    "buildwa": ("build_wA_lambda.py", "build_wA"),
    "probe11": ("probe11_realizedalt_lambda.py", "probe11_realizedalt"),
    "probe13": ("probe13_doseresponse_lambda.py", "probe13_doseresponse"),
    "probe16": ("probe16_curvature_lambda.py", "probe16_curvature"),
    "probe18": ("probe18_steer_lambda.py", "probe18_steer"),
    "floorbattery": ("floor_battery_lambda.py", "floor_battery"),
}
CKPT_MAP = {"opt350m": "opt350m_sst2_wA.pt", "pythia410m": "pythia410m_sst2_subsetmin.pt", "bloom560m": "bloom560m_sst2_wA.pt", "opt13b": "opt13b_sst2_wA.pt", "xglm564m": "xglm564m_sst2_wA.pt", "falcon_rw1b": "falcon_rw1b_sst2_wA.pt", "bloom560m_noln": "bloom560m_noln_sst2_wA.pt"}
assert len(sys.argv) >= 3, "usage: lambda_launch_generic.py <probe3|probe4|probe5|floorbattery> <cell> [block]"
PROBE = sys.argv[1]; CELL = sys.argv[2]
BLOCK = sys.argv[3] if len(sys.argv) > 3 else ""
assert PROBE in PROBES, f"probe must be one of {list(PROBES)}; got {PROBE}"
assert CELL in CKPT_MAP, f"cell must be one of {list(CKPT_MAP)}; got {CELL}"
SCRIPT, RPREFIX = PROBES[PROBE]
if PROBE == "floorbattery":
    assert BLOCK in ("drift", "bf16", "fp32", "seedsample", "seedcurve"), f"floorbattery needs a block; got {BLOCK!r}"
    RPREFIX = f"{RPREFIX}"  # result becomes floor_battery_<cell>_<block>.json via _suffix below
EXTRA_ENV = f"BLOCK={BLOCK} " if BLOCK else ""
_suffix = f"_{BLOCK}" if (PROBE == "floorbattery") else ""
import os, subprocess, time, random
from pathlib import Path
import requests

PD = Path("D:/physics_detector")
SSH_PRIV = PD / "cloud" / "ssh" / "ml_cf_ed25519"
SSH_KEY_NAME = "ML-CF"
API = "https://cloud.lambda.ai/api/v1"
CF = Path("D:/IC_experiments/cf_cascade"); G = CF / "gateA"
PREF = os.environ.get("GPU_PREF", "gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_h100_sxm5,gpu_1x_gh200,gpu_1x_a10").split(",")
if PROBE == "probe9":  # MODE/STREAM env pass-through; harness OUT = probe9_<mode>_<cell>[_<stream>].json
    _MODE = os.environ.get("MODE", "screen"); _STREAM = os.environ.get("STREAM", "mix5")
    EXTRA_ENV += f"MODE={_MODE} STREAM={_STREAM} "
    RPREFIX = f"probe9_{_MODE}"; _suffix = f"_{_STREAM}" if _MODE == "measure" else ""
if PROBE == "probe9b":  # MODE/PHASE2/CHILD_LR pass-through; OUT = probe9b_<mode>_<cell>[_<phase2>].json
    _MODE = os.environ.get("MODE", "screen"); _PHASE2 = os.environ.get("PHASE2", "subj"); _CLR = os.environ.get("CHILD_LR", "")
    EXTRA_ENV += f"MODE={_MODE} PHASE2={_PHASE2} " + (f"CHILD_LR={_CLR} " if _CLR else "")
    RPREFIX = f"probe9b_{_MODE}"; _suffix = f"_{_PHASE2}" if _MODE == "measure" else ""
if PROBE == "probe10":  # CHILD_LR/NSEEDS/PROBE_T pass-through; OUT = probe10_simrehearsal_<cell>.json
    for _k in ("CHILD_LR", "NSEEDS", "PROBE_T"):
        _v = os.environ.get(_k, "")
        if _v: EXTRA_ENV += f"{_k}={_v} "
if PROBE == "probe10post":  # KDIR pass-through; OUT = probe10post_dirsim_<cell>.json
    _v = os.environ.get("KDIR", "")
    if _v: EXTRA_ENV += f"KDIR={_v} "
if PROBE == "probe11":  # LR_GRID pass-through (v3 gentle re-run)
    _v = os.environ.get("LR_GRID", "")
    if _v: EXTRA_ENV += f"LR_GRID={_v} "
if PROBE in ("probe13", "probe16", "probe18"):  # DOSES/NSEEDS/ANCHOR pass-through (ANCHOR = buildwa-measured w_A gate for XGLM/Falcon)
    for _k in ("DOSES", "NSEEDS", "ANCHOR"):
        _v = os.environ.get(_k, "")
        if _v: EXTRA_ENV += f"{_k}={_v} "
if PROBE == "buildwa":  # WA_LR/WA_STEPS pass-through; builds ckpt FROM base (no upload), pulls the .pt back
    for _k in ("WA_LR", "WA_STEPS"):
        _v = os.environ.get(_k, "")
        if _v: EXTRA_ENV += f"{_k}={_v} "
RESULT = f"{RPREFIX}_{CELL}{_suffix}.json"; LOG = f"{RPREFIX}_{CELL}{_suffix}_lambda.log"
UPLOADS = [] if PROBE == "buildwa" else [CKPT_MAP[CELL]]        # buildwa fine-tunes from the base model download
JSONS = [] if PROBE == "buildwa" else ["low_agnews.json", "medium_subj.json"]


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
                last = f"{r.status_code} {r.text[:120]}"; print(f"[L] transient {last} (try {i+1}/{tries})", flush=True); time.sleep(20 if r.status_code==429 else 8); continue
            if r.status_code >= 400:
                print(f"[L] API err {r.status_code}: {r.text}", flush=True); r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = str(e)[:120]; print(f"[L] transient net err {last} (try {i+1}/{tries})", flush=True); time.sleep(8)
    raise RuntimeError(f"API {method} {p} failed after {tries} tries: {last}")


def get(p): return _req("GET", p)
def post(p, b): return _req("POST", p, b)


# Connection-level keepalives: abort a SILENT stall (no bytes for ~Interval*CountMax s)
# without penalizing a slow-but-progressing transfer (keepalive is per-connection, not per-throughput).
# ServerAliveInterval=15 * CountMax=8 => a dead/hung link drops in ~120s instead of blocking forever.
_KEEPALIVE = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
              "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8", "-o", "ConnectTimeout=15"]


def ssh(ip, cmd, _timeout=None, **kw):
    # _timeout: absolute backstop (s). None for short cmds; the long run-phase passes a generous cap.
    return subprocess.run(["ssh", "-i", str(SSH_PRIV)] + _KEEPALIVE + [f"ubuntu@{ip}", cmd],
                          timeout=_timeout, **kw)


def scp(src, dst, tries=3):
    # RECURRING BUG FIX: scp used to hang forever on a silent stall (SSH up, transfer frozen) and bill.
    # Now: keepalive kills a silent stall in ~120s, a 1200s subprocess backstop caps any single try,
    # and we retry the whole transfer up to `tries` times (transient stalls usually clear on retry).
    for i in range(tries):
        try:
            rc = subprocess.run(["scp", "-i", str(SSH_PRIV)] + _KEEPALIVE + [src, dst], timeout=1200).returncode
            if rc == 0:
                return 0
            print(f"[L] scp rc={rc} (try {i+1}/{tries}) {src} -> {dst}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[L] scp TIMEOUT/stall (try {i+1}/{tries}) {src} -> {dst}", flush=True)
        time.sleep(8)
    print(f"[L] scp FAILED after {tries} tries: {src} -> {dst}", flush=True)
    return 1


def pick():
    d = get("/instance-types")["data"]
    for name in PREF:
        t = d.get(name)
        if t and t.get("regions_with_capacity_available"):
            return name, t["regions_with_capacity_available"][0]["name"], t["instance_type"]["price_cents_per_hour"] / 100
    print("[L] NO CAPACITY", flush=True); sys.exit(2)


def main():
    _j = random.uniform(0, 20); print(f"[L:{PROBE}:{CELL}] POLITENESS jitter {_j:.1f}s before launch", flush=True); time.sleep(_j)
    name, region, pr = pick()
    print(f"[L:{PROBE}:{CELL}] picked {name} @ {region} ${pr:.2f}/hr", flush=True)
    iid = post("/instance-operations/launch", {"region_name": region, "instance_type_name": name,
                                               "ssh_key_names": [SSH_KEY_NAME], "quantity": 1})["data"]["instance_ids"][0]
    print(f"[L:{PROBE}:{CELL}] instance_id={iid}  (manual teardown: terminate this id if I go dark)", flush=True)
    try:
        t0 = time.time(); ip = None; last = None
        while time.time() - t0 < 900:
            info = get(f"/instances/{iid}")["data"]; st = info.get("status"); ip = info.get("ip")
            if st != last:
                print(f"[L:{PROBE}:{CELL}] status={st} ip={ip} {time.time()-t0:.0f}s", flush=True); last = st
            if st == "active" and ip:
                try:
                    r = subprocess.run(["ssh", "-i", str(SSH_PRIV), "-o", "StrictHostKeyChecking=no",
                                        "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "echo ok"], capture_output=True, text=True, timeout=15)
                    if r.returncode == 0 and "ok" in r.stdout: break
                except subprocess.TimeoutExpired:
                    print(f"[L:{PROBE}:{CELL}] ssh probe hung (retrying) {time.time()-t0:.0f}s", flush=True)
            time.sleep(15)
        else:
            raise TimeoutError("instance never reached active+ssh in 900s")
        print(f"[L:{PROBE}:{CELL}] SSH up at {ip} ({time.time()-t0:.0f}s)", flush=True)

        ssh(ip, "mkdir -p /home/ubuntu/gateA/overlap_axis /home/ubuntu/hf_cache")
        assert scp(str(CF / "cloud" / SCRIPT), f"ubuntu@{ip}:/home/ubuntu/{SCRIPT}") == 0, "scp script"
        for f in UPLOADS:
            assert scp(str(G / f), f"ubuntu@{ip}:/home/ubuntu/gateA/") == 0, f"scp {f}"
        for j in JSONS:
            assert scp(str(G / "overlap_axis" / j), f"ubuntu@{ip}:/home/ubuntu/gateA/overlap_axis/") == 0, f"scp {j}"
        print(f"[L:{PROBE}:{CELL}] upload done ({time.time()-t0:.0f}s)", flush=True)

        ssh(ip, "pip install -q --upgrade transformers datasets accelerate Pillow")
        print(f"[L:{PROBE}:{CELL}] running {SCRIPT} ({time.time()-t0:.0f}s)...", flush=True)
        try:
            rc = ssh(ip, f"bash -c 'set -o pipefail; cd /home/ubuntu && {EXTRA_ENV}CELL={CELL} python {SCRIPT} 2>&1 | tee pl.log'",
                     _timeout=21600).returncode   # 6h absolute backstop: only fires on a true alive-but-hung run
        except subprocess.TimeoutExpired:
            print(f"[L:{PROBE}:{CELL}] RUN WATCHDOG: exceeded 6h -- treating as hung, tearing down", flush=True); rc = -1
        print(f"[L:{PROBE}:{CELL}] run rc={rc} ({time.time()-t0:.0f}s)", flush=True)

        scp(f"ubuntu@{ip}:/home/ubuntu/gateA/{RESULT}", str(G / RESULT))
        scp(f"ubuntu@{ip}:/home/ubuntu/pl.log", str(CF / "cloud" / LOG))
        print(f"[L:{PROBE}:{CELL}] RESULT LANDED: gateA/{RESULT}" if (G / RESULT).exists() else f"[L:{PROBE}:{CELL}] WARN: no result pulled", flush=True)
        if PROBE == "buildwa":   # pull the freshly-built w_A checkpoint back to local gateA
            _ck = CKPT_MAP[CELL]
            scp(f"ubuntu@{ip}:/home/ubuntu/gateA/{_ck}", str(G / _ck))
            print(f"[L:{PROBE}:{CELL}] CKPT PULLED: gateA/{_ck}" if (G / _ck).exists() else f"[L:{PROBE}:{CELL}] WARN: no ckpt pulled", flush=True)
    finally:
        print(f"[L:{PROBE}:{CELL}] terminating {iid}...", flush=True)
        try:
            post("/instance-operations/terminate", {"instance_ids": [iid]}); print(f"[L:{PROBE}:{CELL}] terminated {iid}", flush=True)
        except Exception as e:
            print(f"[L:{PROBE}:{CELL}] TERMINATE FAILED ({e}) -- MANUAL: terminate instance {iid}", flush=True)


if __name__ == "__main__":
    main()
