import time, json, sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rp import rest
LOG = os.path.expanduser('~/code/opsd/poll2.log')
base = {"imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "dataCenterIds": ["US-CA-2"],
        "networkVolumeId": "o9x5ta7190", "volumeMountPath": "/workspace", "containerDiskInGb": 500,
        "ports": ["22/tcp", "8888/http"], "cloudType": "SECURE", "supportPublicIp": True}
# cheapest total $/hr first; all fit the 49B bf16 model (~100 GB) with KV cache
opts = [("NVIDIA A40", 4), ("NVIDIA A100-SXM4-80GB", 2), ("NVIDIA A100 80GB PCIe", 2), ("NVIDIA RTX 6000 Ada Generation", 4),
        ("NVIDIA L40S", 4), ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 2), ("NVIDIA H200", 1),
        ("NVIDIA H100 PCIe", 2), ("NVIDIA H100 NVL", 2), ("NVIDIA H100 80GB HBM3", 2)]
def log(m):
    s = f"{datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')} {m}"; print(s, flush=True); open(LOG, 'a').write(s + '\n')
while True:
    for g, n in opts:
        try:
            p = rest('/pods', 'POST', {**base, "name": "opsd-check", "gpuTypeIds": [g], "gpuCount": n})
            if p.get('id'):
                log(f"OBTAINED {p['id']} {n}x {g} costPerHr={p.get('costPerHr')}")
                open(os.path.expanduser('~/code/opsd/POD2_ID'), 'w').write(p['id']); sys.exit(0)
            log(f"{n}x {g}: {json.dumps(p)[:150]}")
        except Exception as e:
            log(f"{n}x {g}: {e} {getattr(e, 'read', lambda: b'')().decode()[:120]}")
    try:
        r = rest('/pods/9x9gn11isv7f3t/start', 'POST', {})
        log(f"restart opsd-2xh100: {json.dumps(r)[:150]}")
        st = rest('/pods/9x9gn11isv7f3t')
        if st.get('desiredStatus') == 'RUNNING':
            log("OBTAINED 9x9gn11isv7f3t 2x NVIDIA H100 80GB HBM3 (restarted opsd-2xh100) costPerHr=7.98")
            open(os.path.expanduser('~/code/opsd/POD2_ID'), 'w').write('9x9gn11isv7f3t'); sys.exit(0)
    except Exception as e:
        log(f"restart opsd-2xh100: {e} {getattr(e, 'read', lambda: b'')().decode()[:120]}")
    time.sleep(120)
