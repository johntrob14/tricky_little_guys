import json,pathlib,collections,re
P=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control');O=pathlib.Path('work/sol-ncri/confounds')
out={}
for p in (P/'rows').glob('*qwen3.5*'):
 rr=[json.loads(l) for l in open(p)]; rr=[r for r in rr if not r.get('salt')]
 a={k:dict(collections.Counter(str(r.get(k)) for r in rr)) for k in ['status','finish_reason','provider','witness','channel','fallback','reasoning_detail_types']}
 a['n']=len(rr);a['empty']=sum(not(r.get('reasoning') or '').strip() for r in rr)
 a['nonempty_missing_tokens']=sum(bool(r.get('reasoning')) and not r.get('reasoning_tokens') for r in rr)
 a['under2char_per_token']=sum(len(r.get('reasoning') or '')/r['reasoning_tokens']<2 for r in rr if r.get('reasoning_tokens'))
 a['cap_with_content']=sum(r.get('finish_reason')=='length' and bool(r.get('content')) for r in rr)
 # maximum repetition of 100-char blocks is a crude degeneration diagnostic, not an evaluative label.
 a['most_repeated100']=max(max(collections.Counter((r.get('reasoning') or '')[i:i+100] for i in range(0,len(r.get('reasoning') or '')-99,100)).values(),default=0) for r in rr)
 out[p.stem]=a
json.dump(out,open(O/'qwen_serving.json','w'),indent=2);print(json.dumps(out,indent=2))
