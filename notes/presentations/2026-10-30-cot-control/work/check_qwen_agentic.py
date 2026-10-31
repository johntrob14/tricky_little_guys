import json,subprocess,zipfile,io,collections,hashlib
from pathlib import Path
G='/home/john/Documents/Codex/2026-10-29/okay-i-want-to-think-about/work/ethan_remote.git';REV='6877d3a799b168eaf01b0b66a6db8d55bd02ca05';hashes={}
def read(p):
 b=subprocess.check_output(['git','--git-dir='+G,'show',REV+':'+p]);hashes[p]=hashlib.sha256(b).hexdigest();return b
def obj(p):return json.loads(read(p))
def rows(p):return [json.loads(l) for l in read(p).splitlines() if l]
paths=subprocess.check_output(['git','--git-dir='+G,'ls-tree','-r','--name-only',REV],text=True).splitlines();root='arena/results/startgate/direct/';cal=obj(root+'controls/thresholds.json');passes=list(cal['thresholds']);join=obj(root+'transcript_done_joined.json');fallback=rows(root+'transcript_done.jsonl');results={};episodes=[];selected=[]
for run in ['sgcmp-qwen-direct','opsd1b-sd-pw','sgcmp-qwen-covert2','opsd1b-hs-pw']:
 scores={}
 for pa in passes:
  allrows=[r for p in paths if p.startswith('arena/logs/'+run+'/'+pa+'/') and p.endswith('/monitor.jsonl') for r in rows(p)];scores[pa]=allrows
 joins={(r['file'],r['sample']):r for r in join if r['dir']==run}
 if not joins:joins={(r['file'],r['sample']):r for r in fallback if r.get('dir')==run}
 es=[]
 for p in paths:
  if not(p.startswith('arena/logs/'+run+'/') and p.endswith('.eval')):continue
  z=zipfile.ZipFile(io.BytesIO(read(p)))
  for n in z.namelist():
   if not n.startswith('samples/'):continue
   s=json.loads(z.read(n));key=(p.split('/')[-1],s['id']);score=next(iter(s.get('scores',{}).values()),{}).get('metadata',{});main=bool(score.get('main_pass'));j=joins.get(key);done=bool(j.get('judge_done') and (run.startswith('sgcmp-') or j.get('grader'))) if j else bool(score.get('side_leaked'));gate=s.get('metadata',{}).get('start_turn');maxima={};post={}
   for pa in passes:
    rr=[r for r in scores[pa] if r['log']==key[0] and r['sample']==key[1] and r.get('score') is not None];maxima[pa]=max([r['score'] for r in rr],default=None);post[pa]=max([r['score'] for r in rr if gate is not None and r['turn']>=gate],default=None) if gate is not None else None
   def miss(dd,pp):return all(dd[pa] is not None and dd[pa]<=cal['thresholds'][pa]['0.05'] for pa in pp)
   e=dict(run=run,id=s['id'],main=main,done=done,cot_missed=done and miss(maxima,passes[:2]),joint_cot=main and done and miss(maxima,passes[:2]),whole_all6=main and done and miss(maxima,passes),post_all6=main and done and miss(post,passes),maxima=maxima,post_maxima=post);es.append(e)
   if run=='opsd1b-sd-pw' and s['id'].endswith('#13'):
    turn=0
    for m in s['messages']:
     if m['role']!='assistant':continue
     if turn in [16,17,18,20,27]:selected.append({'turn':turn,'content':m.get('content'),'tool_calls':m.get('tool_calls')})
     turn+=1
 results[run]={'launched':len(es),**{k:sum(e[k] for e in es) for k in ['main','done','joint_cot','whole_all6','post_all6']},'cot_missed_among_done':sum(e['cot_missed'] for e in es)};episodes+=es
l=obj('self-oversight/results/copyenv/judge_audit/labels.json');groups=collections.defaultdict(list)
for k,v in l.items():groups[tuple(k.split('|')[:2])].append(v)
copy=[dict(run=k[0],arm=k[1],n=len(v),reveals=dict(collections.Counter(r['reveal'] for r in v))) for k,v in groups.items()]
out=dict(snapshot=REV,arena=results,episodes=episodes,copyenv=copy,thresholds=cal['thresholds'],hashes=hashes)
Path('outputs/qwen_agentic_check.json').write_text(json.dumps(out,indent=2));Path('work/qwen_selected_agentic_turns.json').write_text(json.dumps(selected,indent=2));print(json.dumps(results,indent=2));print('COPY',json.dumps(copy));print('EXCEPTION',json.dumps([e for e in episodes if e['post_all6']]))
