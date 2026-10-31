from pathlib import Path
import json,collections,shutil
O=Path('work/arena_startgate');d=json.loads((O/'results.json').read_text());A=Path('/home/john/code/tlg-inksm/arena')
for cell in d['cells']:
 rr={(r['file'],r['sample']):r for r in map(json.loads,open(cell['episode_file']))};eps=[e for e in d['episodes'] if e['run']==cell['run']]
 for e in eps:e['main_pass']=bool(rr[e['file'],e['sample']].get('main_pass'))
 cell['main_pass']=sum(e['main_pass'] for e in eps);cell['main_and_side']=sum(e['main_pass'] and e['done'] for e in eps);cell['main_side_and_joint_cot_miss']=sum(e['main_pass'] and e['joint_cot_miss'] for e in eps)
 for p,v in cell['monitors'].items():
  r=[json.loads(l) for f in (A/'logs'/cell['run']/p).rglob('monitor.jsonl') for l in f.open()];v['empty_input_rows']=sum(bool(x.get('empty_cot')) for x in r);v['unparsed_rows']=sum(bool(x.get('unparsed')) for x in r);v['no_score_rows']=v.pop('missing_score_rows')
 print(cell['run'],'main',cell['main_pass'],'side',cell['done'],'both',cell['main_and_side'],'both+cotmiss',cell['main_side_and_joint_cot_miss'])
(O/'results.json').write_text(json.dumps(d,indent=2))
P=O/'prompts';P.mkdir(exist_ok=True)
for name in ['controlarena_cot_only','controlarena_cot','controlarena_action','controlarena_cot_only_context']:
 p=A/f'rl/monitor/prompts/{name}.txt'
 if p.exists():shutil.copyfile(p,P/(name+'.txt'))
for name in ['direct_system.txt','direct_user.md']:shutil.copyfile(A/'honeypot/startgate/cruft'/name,P/name)
