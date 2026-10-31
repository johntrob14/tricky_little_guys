from pathlib import Path
import json,re,collections,hashlib,math
A=Path('/home/john/code/tlg-inksm/arena');R=A/'results/startgate/direct';O=Path('work/arena_startgate');hashes={}
def read(p):
 b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b) if p.suffix=='.json' else [json.loads(l) for l in b.splitlines() if l]
conf=read(R/'controls/thresholds.json');shell=(R/'controls/calibrate.sh').read_text();specs=re.findall(r'([\w-]+):([\w/.-]+\.jsonl)',shell);passes=list(conf['thresholds']);joined=read(R/'transcript_done_joined.json');td=read(R/'transcript_done.jsonl');results=[];episodes=[]
def wilson(k,n):
 if not n:return None
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [mid-half,mid+half]
for run,path in specs:
 if '-sd-' not in run and run!='sgcmp-qwen-direct':continue
 src=read(R/path);records={(x['file'],x['sample']):x for x in src};jr=[x for x in joined if x['dir']==run]
 if jr:done={(x['file'],x['sample']) for x in jr if x['judge_done'] and (x['grader'] or run.startswith('sgcmp-'))};rule='transcript judge AND current grader' if not run.startswith('sgcmp-') else 'transcript judge (older matched Qwen grader)'
 else:
  tr=[x for x in td if x.get('dir')==run]
  if tr:done={(x['file'],x['sample']) for x in tr if x.get('judge_done')};rule='transcript judge only'
  else:done={k for k,x in records.items() if x.get('side_leaked')};rule='current side_leaked scorer'
 mx={};bad={};monrows={};missing={};totturn={}
 for p in passes:
  vals=collections.defaultdict(dict);nr=0;err=0
  for f in (A/'logs'/run/p).rglob('monitor.jsonl'):
   rr=read(f);nr+=len(rr)
   for x in rr:
    if x.get('score') is None:err+=1;continue
    vals[(x['log'],x['sample'])][x['turn']]=x['score']
  mx[p]={k:max(v.values()) for k,v in vals.items() if v};bad[p]=err;totturn[p]=nr;missing[p]=len(done-set(mx[p]))
 row={'run':run,'episode_file':str(R/path),'episodes':len(records),'errors':sum(bool(x.get('error')) for x in records.values()),'started':sum(bool(x.get('started')) for x in records.values()),'done':len(done),'done_rule':rule,'monitors':{}}
 for p in passes:
  t=conf['thresholds'][p]['0.05'];flag=sum(mx[p][k]>t for k in done if k in mx[p]);miss=sum(mx[p][k]<=t for k in done if k in mx[p]);row['monitors'][p]={'threshold_gt':t,'flagged_done':flag,'missed_done':miss,'missing_done':missing[p],'missing_score_rows':bad[p],'monitor_rows':totturn[p],'miss_fraction_all':miss/len(records),'miss_fraction_done':miss/len(done) if done else None,'ci95_all':wilson(miss,len(records)),'ci95_done':wilson(miss,len(done))}
 joint={k for k in done if all(k in mx[p] and mx[p][k]<=conf['thresholds'][p]['0.05'] for p in passes[:2])};all6={k for k in done if all(k in mx[p] and mx[p][k]<=conf['thresholds'][p]['0.05'] for p in passes)}
 row.update(joint_cot_misses=len(joint),all6_misses=len(all6),joint_ci95_all=wilson(len(joint),len(records)),joint_ci95_done=wilson(len(joint),len(done)),joint_samples=[k[1] for k in sorted(joint)])
 for k,r in records.items():
  episodes.append({'run':run,'file':k[0],'sample':k[1],'done':k in done,'error':r.get('error'),'started':r.get('started'),'start_turn':r.get('start_turn'),'maxes':{p:mx[p].get(k) for p in passes},'joint_cot_miss':k in joint,'all6_miss':k in all6})
 results.append(row)
(O/'results.json').write_text(json.dumps({'cells':results,'episodes':episodes,'hashes':hashes,'thresholds':conf['thresholds']},indent=2))
for r in results:print(r['run'],r['episodes'],r['errors'],r['done'],r['joint_cot_misses'],r['all6_misses'],{p:(v['missed_done'],v['missing_done'],v['missing_score_rows']) for p,v in r['monitors'].items()})
