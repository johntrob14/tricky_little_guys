"""Continue the additive run from s60 to s80 (budget raised to $40 total), evaluating s72 and s80 with the standard eval plus 3 extra Astra draws each."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path('/home/john/code/results/nocotrl-b-pilot');O=P/'additive';R=Path('/home/john/code/nocotrl-b-pilot');B=R/'cot-control/rl/b_pilot';D=P/'followup_noise_elicit'
ENDS=(72,80)
def status(stage,**kw):(O/'status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),time=time.time(),**kw),indent=2))
def py(args,log):
    with open(log,'a') as f:return subprocess.Popen([sys.executable,*map(str,args)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=R)
def noise(label,path):return py([B/'followup_noise_elicit.py','--out',D/f'draws_{label}','--budget',0.6,'--noise',f'{label}={path}'],D/f'draws_{label}.log')
def main():
    side=[]
    for end in ENDS:
        status('training',step_target=end)
        assert py([B/'additive.py','train','--end-step',end,'--out',O,'--repo',R,'--resume','--segment',f'to{end}','--resume-reason','Continue s60->80 to match nocotrl1 step 80.'],O/f'training_{end}.log').wait()==0
        ck=json.load(open(O/'checkpoints.json'))[-1];assert ck['step']==end,'Stopped before target; inspect budget'
        status('evaluation',step=end)
        side.append(noise(f'add_s{end}',ck['sampler']))
        assert py([B/'additive.py','eval','--label',f's{end}','--out',O,'--repo',R],O/f'evaluation_{end}.log').wait()==0
    status('waiting_for_extra_draws')
    assert all(p.wait()==0 for p in side)
    status('done')
if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise
