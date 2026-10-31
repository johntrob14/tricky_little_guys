import json,zipfile,hashlib
from pathlib import Path
R=Path('/home/john/code/tlg-inksm/arena');O=Path('work/arena_traces');manifest={};bundle={}
def raw(p):
 b=p.read_bytes();manifest[str(p)]=hashlib.sha256(b).hexdigest();return b
def rows(p):return [json.loads(l) for l in raw(p).splitlines() if l]
cal=json.loads(raw(R/'results/startgate/direct/controls/thresholds.json'))
for run in ['opsd1b-sd-pw','grpo3-s100-sd-pw','glm53-sd-pw']:
 mon={m:rows(next((R/'logs'/run/m).glob('*/monitor.jsonl'))) for m in cal['thresholds']}
 eps=[]
 for p in (R/'logs'/run).glob('*.eval'):
  raw(p)
  z=zipfile.ZipFile(p)
  for name in z.namelist():
   if not name.startswith('samples/'):continue
   s=json.loads(z.read(name));start=s.get('metadata',{}).get('start_turn');sid=s['id'];maxima={};post={};turns=[];t=0
   for mi,msg in enumerate(s['messages']):
    if msg['role']=='assistant':
     c=msg.get('content');cot='\n'.join(x.get('reasoning','') for x in c if x.get('type')=='reasoning') if isinstance(c,list) else '';visible='\n'.join(x.get('text','') for x in c if x.get('type')=='text') if isinstance(c,list) else str(c or '')
     turns.append({'turn':t,'message_index':mi,'cot':cot,'visible':visible,'tool_calls':msg.get('tool_calls',[])});t+=1
   for m,rs in mon.items():
    rr=[r for r in rs if r['sample']==sid and r['log']==p.name];maxima[m]=max((r['score'] for r in rr if r.get('score') is not None),default=None);pr=[r for r in rr if start is not None and r['turn']>=start];post[m]=max((r['score'] for r in pr if r.get('score') is not None),default=None)
   sc=next(iter(s.get('scores',{}).values()),{}).get('metadata',{});eps.append({'sample':sid,'source':str(p),'member':name,'start_turn':start,'scores':sc,'maxima':maxima,'post_maxima':post,'turns':turns,'messages':s['messages'],'judges':{m:[r for r in rs if r['sample']==sid and r['log']==p.name] for m,rs in mon.items()}})
 bundle[run]=eps
 print(run)
 for e in eps:
  mx=e['post_maxima'];miss=all(mx[m] is not None and mx[m]<=cal['thresholds'][m]['0.05'] for m in list(cal['thresholds'])[:2]);print(e['sample'],e['start_turn'],e['scores'].get('main_pass'),e['scores'].get('side_leaked'),miss,mx)
(O/'raw_bundle.json').write_text(json.dumps(bundle,indent=2));(O/'source_manifest.json').write_text(json.dumps(manifest,indent=2));(O/'thresholds.json').write_text(json.dumps(cal,indent=2))
