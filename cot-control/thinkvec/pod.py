"""RunPod pod for the think-vector experiment: one H100 SXM (secure cloud), ssh exposed, no network volume.

    python pod.py create      # creates the pod, writes its id to .pod_id
    python pod.py status      # desired status, public ip and ssh port, cost per hour
    python pod.py ssh         # prints the ssh command (target: 'ip port', for scripts)
    python pod.py stop | terminate

The key is RUNPOD_MATS_API_KEY in ~/.env (REST needs a User-Agent header or Cloudflare returns 403).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = re.search(r"^RUNPOD_MATS_API_KEY=(.*)$", (Path.home() / ".env").read_text(), re.M).group(1).strip().strip("'\"")
POD_ID_FILE = HERE / ".pod_id"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}", "User-Agent": "thinkvec-pod/1.0"}
CREATE = {"name": "thinkvec-h100", "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
          "gpuTypeIds": ["NVIDIA H100 80GB HBM3"], "gpuCount": 1, "containerDiskInGb": 200, "volumeInGb": 0,
          "ports": ["22/tcp"], "cloudType": "SECURE", "supportPublicIp": True,
          "env": {"PUBLIC_KEY": (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()}}


def rest(path: str, method: str = "GET", body: dict | None = None):
    req = urllib.request.Request("https://rest.runpod.io/v1" + path, data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode()
    return json.loads(text) if text else None


def pod_id() -> str:
    return POD_ID_FILE.read_text().strip()


def ssh_target() -> tuple[str, int]:
    p = rest(f"/pods/{pod_id()}")
    return p["publicIp"], int(p["portMappings"]["22"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "create":
        p = rest("/pods", "POST", CREATE)
        POD_ID_FILE.write_text(p["id"])
        print(p["id"], p.get("costPerHr"), p.get("desiredStatus"), p.get("machineId"))
    elif cmd == "status":
        p = rest(f"/pods/{pod_id()}")
        print({k: p.get(k) for k in ("id", "desiredStatus", "publicIp", "portMappings", "costPerHr", "gpu", "lastStatusChange")})
    elif cmd == "target":
        ip, port = ssh_target()
        print(ip, port)
    elif cmd == "ssh":
        ip, port = ssh_target()
        print(f"ssh -p {port} -o StrictHostKeyChecking=no root@{ip}")
    elif cmd == "stop":
        print(rest(f"/pods/{pod_id()}/stop", "POST", {}))
    elif cmd == "terminate":
        print(rest(f"/pods/{pod_id()}", "DELETE"))
    else:
        raise SystemExit(f"unknown command {cmd}")
