"""Persistent sequential pilot continuation; shares one $20 budget ledger."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
OUT=Path('/home/john/code/results/nocotrl-b-pilot')

def main():
    stages=[
      ('train_resume',['pilot.py','train','--resume','--repo',str(REPO),'--out',str(OUT)]),
      ('eval_base_full',['pilot.py','eval','--label','base','--repo',str(REPO),'--out',str(OUT)]),
      ('eval_pilot',['pilot.py','eval','--label','pilot','--repo',str(REPO),'--out',str(OUT)]),
      ('tiktok_base',['tiktok_eval.py','--label','base','--repo',str(REPO),'--out',str(OUT)]),
      ('tiktok_pilot',['tiktok_eval.py','--label','pilot','--repo',str(REPO),'--out',str(OUT)]),
      ('verify_agents',['verify_agents.py','--out',str(OUT)]),
      ('analyze',['analyze.py','--repo',str(REPO),'--out',str(OUT)]),
      ('checkpoints',['checkpoint_record.py','--out',str(OUT)]),
    ]
    for name,argv in stages:
        status=dict(stage=name,status='running',time=time.time(),pid=os.getpid())
        (OUT/'workflow_status.json').write_text(json.dumps(status))
        with open(OUT/(name+'.log'),'a') as log:
            r=subprocess.run([sys.executable,str(ROOT/argv[0]),*argv[1:]],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=REPO)
        status.update(status='done' if r.returncode==0 else 'failed',returncode=r.returncode,finished=time.time())
        with open(OUT/'workflow_events.jsonl','a') as f:f.write(json.dumps(status)+'\n')
        (OUT/'workflow_status.json').write_text(json.dumps(status))
        if r.returncode:raise SystemExit(r.returncode)
    (OUT/'workflow_status.json').write_text(json.dumps(dict(stage='all',status='done',time=time.time())))

if __name__=='__main__':main()
