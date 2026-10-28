"""Resume the additive run from its s36 state to step 60 (manual stop at 39 was premature), evaluating s48 and s60
with the standard single-draw eval plus 3 extra Astra draws each; also 3 extra draws of the original run's s60."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path('/home/john/code/results/nocotrl-b-pilot');O=P/'additive';R=Path('/home/john/code/nocotrl-b-pilot');B=R/'cot-control/rl/b_pilot';D=P/'followup_noise_elicit'
OLD60='tinker://d4aacddf-181a-5d36-ac92-03c74818841d:train:0/sampler_weights/nocotrl-b-pilot-s60'
def status(stage,**kw):(O/'status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),time=time.time(),**kw),indent=2))
def py(args,log):
    with open(log,'a') as f:return subprocess.Popen([sys.executable,*map(str,args)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=R)
def noise(label,path):return py([B/'followup_noise_elicit.py','--out',D/f'draws_{label}','--budget',0.6,'--noise',f'{label}={path}'],D/f'draws_{label}.log')
def main():
    side=[noise('old_s60',OLD60)]
    for end in (48,60):
        status('training',step_target=end)
        assert py([B/'additive.py','train','--end-step',end,'--out',O,'--repo',R,'--resume','--segment',f'to{end}','--resume-reason','Resume s36->60 after premature manual stop at step 39; steps 37-39 rerun with fresh rollouts.'],O/f'training_{end}.log').wait()==0
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
