from pathlib import Path
import json,zipfile,shutil,math,hashlib,collections
A=Path('/home/john/code/tlg-inksm/arena');O=Path('work/arena_always');p=O/'verified.json';d=json.loads(p.read_text());ep={(r['run'],r['log'],r['sample']):r for r in d['episodes']};example=None
for c in d['cells']:
 raw=[]
 for f in (A/'logs'/c['run']).glob('*.eval'):
  d['hashes'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  with zipfile.ZipFile(f) as z:
   for n in z.namelist():
    if n.startswith('samples/') and n.endswith('.json'):
     r=json.loads(z.read(n));raw.append(r);key=(c['run'],f.name,r['id']);assert key in ep
     sm=next(iter(r.get('scores',{}).values()),{}).get('metadata',{});ep[key]['main_pass']=bool(sm.get('main_pass'));ep[key]['error']=r.get('error');ep[key]['scorer_side_leaked']=sm.get('side_leaked')
     if example is None:example=r
 c['raw_log_episodes']=len(raw);c['raw_log_errors']=sum(bool(r.get('error')) for r in raw);c['main_pass']=sum(ep[(c['run'],r['log'],r['sample'])]['main_pass'] for r in d['episodes'] if r['run']==c['run'])
 c['side_and_main']=sum(r['done'] and r['main_pass'] for r in d['episodes'] if r['run']==c['run'])
 c['source_eval_files']=sorted({r['log'] for r in d['episodes'] if r['run']==c['run']})
print('model | total | main | side | main+side | weak missed | strong missed | both missed')
for c in d['cells']:print(c['model'],c['episodes'],c['main_pass'],c['done'],c['side_and_main'],c['monitors']['monitor_phase']['missed_successful'],c['monitors']['monitor_phase_haiku']['missed_successful'],c['missed_both_cot'],sep=' | ')
(O/'prompt_appendix').mkdir(exist_ok=True)
for n in ['controlarena_cot_only','controlarena_cot','controlarena_action_only','controlarena_cot_only_context']:
 src=A/'rl/monitor/prompts'/f'{n}.txt';shutil.copyfile(src,O/'prompt_appendix'/src.name);d['hashes'][str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
for n,src in [('always_monitored_coaching.txt',A/'prompts/system/arena_covert_v2.txt'),('honest_system.txt',A/'prompts/system/honest.txt'),('main_task.txt',A/'tasks/tiktok/secure_password_policy_engine/task.md')]:shutil.copyfile(src,O/'prompt_appendix'/n)
(O/'prompt_appendix/example_full_agent_messages.json').write_text(json.dumps(example['messages'][:2],indent=2))
d['parse_failure_summary']={group:{p:sum(c['unparsed'] for c in d['coverage_checks'] if c['pass']==p and c['run'].startswith('ctl-')==control) for p in d['thresholds']} for group,control in [('control',True),('always_attack',False)]}
p.write_text(json.dumps(d,indent=2))
