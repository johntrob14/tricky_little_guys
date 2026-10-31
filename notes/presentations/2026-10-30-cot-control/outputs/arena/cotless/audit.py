"""Read-only recount of completed TikTok reasoning-disabled evaluations. Python 3.14."""
from pathlib import Path
import json,zipfile,hashlib,math,collections,random
O=Path(__file__).resolve().parent
R=Path('/home/john/code/tlg-arena-rl/arena')
D=R/'results/startgate/direct/thinking-off-openrouter-20261026'
hashes={};episodes=[];cells=[];reviews=[];rng=random.Random(20261031)
def sha(p):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 sha(p)
 return json.loads(p.read_text()) if p.suffix=='.json' else [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0,c-w),min(1,c+w)]
def samples(runpath):
 for f in sorted(runpath.glob('*.eval')):
  sha(f)
  with zipfile.ZipFile(f) as z:
   h=json.loads(z.read('header.json'));assert h['status']=='success'
   for member in z.namelist():
    if member.startswith('samples/') and member.endswith('.json'):yield f,member,h,json.loads(z.read(member))
def reasoning(s):
 a=[m for m in s['messages'] if m['role']=='assistant']
 r=[m for m in a if isinstance(m.get('content'),list) and any(c.get('type')=='reasoning' and c.get('reasoning','').strip() for c in m['content'])]
 return len(a),len(r)
for cell in ['easy','hard','control']:
 saved={(Path(x['log']).name,x['sample']):x for x in load(D/f'qwen_{cell}_audit.json')}
 exp={(x['log'],x['sample']):x for x in load(D/f'judges/exposure/{cell}_joined.jsonl')} if cell!='control' else {}
 rows=[];traces=[]
 for f,member,h,s in samples(D/'logs/qwen'/cell):
  assert h['eval']['model_generate_config']['reasoning_effort']=='none'
  assert h['eval']['model_args']['provider']=={'order':['DeepInfra'],'allow_fallbacks':False}
  assert not s.get('error');turns,reason=reasoning(s);assert reason==0
  assert all(v.get('reasoning_tokens')==0 for v in s['model_usage'].values())
  m=s['scores']['main_and_side_scorer']['metadata'];key=(f.name,s['id']);old=saved[key]
  for k,v in [('main_pass',m['main_pass']),('side_leaked',m['side_leaked']),('assistant_turns',turns),('reasoning_turns',reason)]:assert old[k]==v
  judge=exp.get(key)
  if judge:
   assert not judge['parse_error'];assert judge['recorded_side_leaked']==m['side_leaked'];assert judge['main_pass']==m['main_pass']
  row=dict(cell=cell,file=str(f),member=member,sample=s['id'],task=s['id'].split('#')[0],main=bool(m['main_pass']),grader_leak=bool(m['side_leaked']),confirmed_leak=bool(m['side_leaked'] and judge and judge['judged']['done']),turns=turns,reasoning_turns=reason,exposure_judge=judge['judged'] if judge else None)
  rows.append(row);traces.append((row,s,m))
 assert len(rows)==16;assert len({x['task'] for x in rows})==4
 for label in [True,False]:
  pool=[x for x in traces if x[0]['main']==label]
  for row,s,m in rng.sample(pool,min(10,len(pool))):
   reviews.append(dict(episode=row,main_why=m['main_why'],side_why=m['side_why'],assistant_excerpt=[x for x in s['messages'] if x['role']=='assistant'][-1:]))
 c=dict(cell=cell,n=len(rows),main=sum(x['main'] for x in rows),grader_leaks=sum(x['grader_leak'] for x in rows),confirmed_leaks=sum(x['confirmed_leak'] for x in rows),assistant_turns=sum(x['turns'] for x in rows),reasoning_turns=sum(x['reasoning_turns'] for x in rows))
 c['main_ci95']=wilson(c['main'],c['n']);c['leak_ci95']=wilson(c['confirmed_leaks'],c['n'])
 cells.append(c);episodes.extend(rows)
# Forced-off controls on the single password-policy task are separate comparisons.
forced=[]
for run in ['oss-nc-hs-pw','oss-nc-sd-pw','qwen-nc-hs-pw','qwen-nc-sd-pw','qorg-nc-hs-pw','qorg-nc-sd-pw','oss-hs-pw','oss-sd-pw','sgcmp-qwen-covert2','sgcmp-qwen-direct','opsd1b-hs-pw','opsd1b-sd-pw']:
 rows=[]
 for f,member,h,s in samples(R/'logs'/run):
  m=s['scores']['main_and_side_scorer']['metadata'];turns,reason=reasoning(s)
  rows.append(dict(file=str(f),member=member,sample=s['id'],main=bool(m['main_pass']),grader_leak=bool(m['side_leaked']),turns=turns,reasoning_turns=reason))
 assert len(rows)==16
 forced.append(dict(run=run,n=len(rows),main=sum(x['main'] for x in rows),grader_leaks=sum(x['grader_leak'] for x in rows),assistant_turns=sum(x['turns'] for x in rows),reasoning_turns=sum(x['reasoning_turns'] for x in rows),episodes=rows))
out=dict(cells=cells,episodes=episodes,forced_controls=forced,hashes=hashes,review_seed=20261031,verification='Read raw .eval scores, headers, reasoning content and reported reasoning-token counts; independently joined saved exposure judgments. All counts match saved per-episode audits. Exposure judgments are reused, not rerun; their explanations were inspected. Intervals are 95% Wilson over episodes, conditional on this fixed four-task suite; they do not estimate generalization across tasks or training seeds.')
(O/'verified.json').write_text(json.dumps(out,indent=2)+'\n');(O/'review_samples.json').write_text(json.dumps(reviews,indent=2)+'\n')
print(json.dumps(cells,indent=2))
