"""Poll RunPod for a 2x H200 SXM pod ($10.58/hr, approved by John 2026-10-19). US-CA-2 with network volume vol1 first;
after 30 min also any secure data center without the volume (results then live on the container disk until pulled)."""
import json, os, sys, time, datetime
sys.path.insert(0, os.path.expanduser('~/code/rh_opsd'))
from rp import rest
LOG = os.path.expanduser('~/code/tricky_little_guys/auditbench-opsd/pod/poll.log')
OUT = os.path.expanduser('~/code/tricky_little_guys/auditbench-opsd/pod/POD_ID')
base = {"name": "ab-opsd", "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "gpuTypeIds": ["NVIDIA H200"],
        "gpuCount": 2, "ports": ["22/tcp", "8888/http"], "cloudType": "SECURE", "supportPublicIp": True}
vol = {**base, "dataCenterIds": ["US-CA-2"], "networkVolumeId": "o9x5ta7190", "volumeMountPath": "/workspace", "containerDiskInGb": 500}
novol = {**base, "containerDiskInGb": 900, "volumeInGb": 0}
t0 = time.time()
def log(m):
    s = f"{datetime.datetime.utcnow().isoformat(timespec='seconds')}Z {m}"
    print(s, flush=True); open(LOG, 'a').write(s + '\n')
while time.time() - t0 < 3 * 3600:
    for name, body in [("vol", vol)] + ([("novol", novol)] if time.time() - t0 > 1800 else []):
        try:
            p = rest('/pods', 'POST', body)
            if p.get('id'):
                log(f"OBTAINED {name} pod {p['id']} cost={p.get('costPerHr')} dc={p.get('machine',{}).get('dataCenterId') if isinstance(p.get('machine'),dict) else None}")
                open(OUT, 'w').write(json.dumps({"id": p['id'], "mode": name}))
                sys.exit(0)
            log(f"{name}: {json.dumps(p)[:200]}")
        except Exception as e:
            log(f"{name}: {getattr(e,'read',lambda:b'')().decode()[:160]}")
    time.sleep(150)
log("gave up after 3 h")
