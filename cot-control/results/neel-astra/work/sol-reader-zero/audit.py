import json,pathlib,collections,statistics
s=json.load(open('work/sol-reader-zero/selected.json'));root=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control');out=[]
for m in dict.fromkeys(x['model'] for x in s):
 rs=[json.loads(l) for l in (root/'rows'/(m.replace('/','_')+'.jsonl')).read_text().splitlines()];gs=[json.loads(l) for l in (root/'graded'/(m.replace('/','_')+'.jsonl')).read_text().splitlines()]
 ratios=[len(r['reasoning'])/r['reasoning_tokens'] for r in rs if r.get('reasoning_tokens')]
 a=dict(model=m,n=len(rs),graded_n=len(gs),compliant=sum(g.get('compliance')==1 for g in gs),witness=dict(collections.Counter(r['witness'] for r in rs)),types=dict(collections.Counter(str(r.get('reasoning_detail_types')) for r in rs)),providers=dict(collections.Counter(r.get('provider') for r in rs)),cap=sum(r.get('finish_reason')=='length' for r in rs),ratio_min=min(ratios),ratio_median=statistics.median(ratios),ratio_max=max(ratios),missing_tokens=sum(not r.get('reasoning_tokens') for r in rs),impossible_usage=sum((r.get('reasoning_tokens') or 0)>(r.get('completion_tokens') or 0) for r in rs))
 out.append(a);print(json.dumps(a))
pathlib.Path('work/sol-reader-zero/audit.json').write_text(json.dumps(out,indent=2))
print('selected stop',collections.Counter(r['finish_reason'] for r in s),'template',sum(r['reasoning'].startswith('Thinking Process:') for r in s))
