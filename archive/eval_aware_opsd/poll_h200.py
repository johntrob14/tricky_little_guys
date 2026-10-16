import time,json,sys,datetime,os
sys.path.insert(0,os.path.dirname(__file__))
from rp import rest,gql
LOG=os.path.expanduser('~/code/opsd/poll.log')
START='2026-10-15T04:00:54Z'
body={"name":"opsd-h200","imageName":"runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404","gpuTypeIds":["NVIDIA H200"],"gpuCount":1,
 "dataCenterIds":["US-CA-2"],"networkVolumeId":"o9x5ta7190","volumeMountPath":"/workspace","containerDiskInGb":500,
 "ports":["22/tcp","8888/http"],"cloudType":"SECURE","supportPublicIp":True}
def log(m):
    s=f"{datetime.datetime.utcnow().isoformat(timespec='seconds')}Z {m}"
    print(s,flush=True); open(LOG,'a').write(s+'\n')
import calendar
PIVOT=calendar.timegm(time.strptime('2026-10-15T04:45:45Z','%Y-%m-%dT%H:%M:%SZ'))
h100={**body,'name':'opsd-2xh100','gpuTypeIds':['NVIDIA H100 80GB HBM3'],'gpuCount':2}
while True:
    try:
        if time.time()>=PIVOT:
            log('trying 2x H100 SXM (user: pivot if 10 more min without H200)')
            try: p=rest('/pods','POST',h100)
            except Exception as e:
                log(f"2xH100 create failed: {e} {getattr(e,'read',lambda:b'')().decode()[:200]}"); p={}
            if not p.get('id'): p=rest('/pods','POST',body)
        else:
            p=rest('/pods','POST',body)
        if p.get('id'):
            log(f"OBTAINED pod {p['id']} costPerHr={p.get('costPerHr')} machine={p.get('machineId')}")
            open(os.path.expanduser('~/code/opsd/POD_ID'),'w').write(p['id'])
            break
        log(f"no pod: {json.dumps(p)[:300]}")
    except Exception as e:
        msg=getattr(e,'read',lambda:b'')().decode()[:300]
        log(f"create failed: {e} {msg}")
    time.sleep(300 if time.time()<PIVOT else 120)
