import json,random,collections,pathlib
p='/home/john/code/tlg-arena-rl/cot-control/results/outctl/more/cot/base/graded.json'
r=json.load(open(p));by=collections.defaultdict(list)
for i,x in enumerate(r): by[x['constraint']].append((i,x))
rng=random.Random(20261026);s=[rng.choice(v) for k,v in sorted(by.items())];rest=[(i,x) for i,x in enumerate(r) if i not in [j for j,_ in s]];s+=rng.sample(rest,10-len(s))
pathlib.Path('work/sol-reader-zero/base-selected.json').write_text(json.dumps([dict(index=i,**x) for i,x in s],indent=2))
for i,x in s: print('\nINDEX',i,'KEY',x['key'],'SAMPLE',x['sample_idx'],'STRICT',x['strict'],'STOP',x['stop'],'TOKENS',x['n_tokens'],'\nREQUIREMENT',x['user'].split('Requirement:')[-1],'\nOUT FULL:',x['out'],'\nRESP FULL:',x['resp'])
