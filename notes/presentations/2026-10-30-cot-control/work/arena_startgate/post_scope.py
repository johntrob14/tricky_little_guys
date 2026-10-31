from pathlib import Path
import json
A=Path('/home/john/code/tlg-inksm/arena');O=Path('work/arena_startgate');d=json.loads((O/'results.json').read_text());out=[]
for cell in d['cells']:
 r=[json.loads(l) for l in Path(cell['episode_file']).read_text().splitlines()];starts={(x['file'],x['sample']):x.get('start_turn',x.get('split')) for x in r};eps=[x for x in d['episodes'] if x['run']==cell['run']];ps={}
 for p in d['thresholds']:
  vals={}
  for f in (A/'logs'/cell['run']/p).rglob('monitor.jsonl'):
   for l in f.open():
    x=json.loads(l);k=(x['log'],x['sample']);s=starts.get(k)
    if s is None or x['turn']<s or x.get('score') is None:continue
    vals[k]=max(vals.get(k,-1),x['score'])
  ps[p]=vals
 for e in eps:
  k=(e['file'],e['sample']);e['poststart_allpass_maxes']={p:vals.get(k) for p,vals in ps.items()};e['all6_poststart_miss']=e['done'] and all(k in v and v[k]<=d['thresholds'][p]['0.05'] for p,v in ps.items())
 out.append({'run':cell['run'],'done':cell['done'],'poststart_missed':{p:sum(e['done'] and e['poststart_allpass_maxes'][p] is not None and e['poststart_allpass_maxes'][p]<=d['thresholds'][p]['0.05'] for e in eps) for p in ps},'all6_poststart_misses':sum(e['all6_poststart_miss'] for e in eps),'all6_poststart_samples':[e['sample'] for e in eps if e['all6_poststart_miss']]})
(O/'poststart_scopes.json').write_text(json.dumps(out,indent=2));(O/'results.json').write_text(json.dumps(d,indent=2));print(json.dumps(out,indent=2))
