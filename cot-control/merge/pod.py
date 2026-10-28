"""RunPod pod for the LoRA-merge eval: 4 H100 SXM (one vLLM server, bf16 base + unmerged LoRAs, tensor parallel 4), ssh, no network volume.

    python pod.py create [n_gpus]   # default 4; writes the id to .pod_id
    python pod.py status | target | ssh | stop | terminate
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thinkvec"))
from pod import rest  # noqa: E402

POD_ID_FILE = HERE / ".pod_id"
CREATE = {"name": "merge-4xh100", "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
          "gpuTypeIds": ["NVIDIA H100 80GB HBM3"], "gpuCount": 4, "containerDiskInGb": 400, "volumeInGb": 0,
          "ports": ["22/tcp"], "cloudType": "SECURE", "supportPublicIp": True,
          "env": {"PUBLIC_KEY": (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()}}


def pod_id() -> str:
    return POD_ID_FILE.read_text().strip()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "create":
        body = dict(CREATE, gpuCount=int(sys.argv[2])) if len(sys.argv) > 2 else CREATE
        p = rest("/pods", "POST", body)
        POD_ID_FILE.write_text(p["id"])
        print(p["id"], p.get("costPerHr"), p.get("desiredStatus"), p.get("machineId"))
    elif cmd == "status":
        p = rest(f"/pods/{pod_id()}")
        print({k: p.get(k) for k in ("id", "desiredStatus", "publicIp", "portMappings", "costPerHr", "gpu", "lastStatusChange")})
    elif cmd == "target":
        p = rest(f"/pods/{pod_id()}")
        print(p["publicIp"], int(p["portMappings"]["22"]))
    elif cmd == "ssh":
        p = rest(f"/pods/{pod_id()}")
        print(f"/usr/bin/ssh -p {int(p['portMappings']['22'])} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@{p['publicIp']}")
    elif cmd == "stop":
        print(rest(f"/pods/{pod_id()}/stop", "POST", {}))
    elif cmd == "terminate":
        print(rest(f"/pods/{pod_id()}", "DELETE"))
    else:
        raise SystemExit(f"unknown command {cmd}")
