import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('kind', choices=['pairs', 'training', 'index'])
p.add_argument('start', type=int, nargs='?', default=1)
p.add_argument('end', type=int, nargs='?', default=999)
p.add_argument('--model', choices=['base', 'trained'])
p.add_argument('--prompt', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parent
if a.kind == 'index':
    for n, pair in enumerate(json.loads((root/'pairs.json').read_text()), 1):
        print(n, pair['split'], pair['key'], [(model, row['strict'], len(row['out']), row['words']) for model, row in pair.items() if model in ('base', 'trained')])
else:
    rows = json.loads((root / (a.kind+'.json')).read_text())
    for n, row in enumerate(rows, 1):
        if not a.start <= n <= a.end:
            continue
        if a.kind == 'pairs':
            print(f"PAIR {n:02d} | {row['split']} | {row['key']} | sample {row['sample_idx']}")
            if a.prompt:
                print('FULL USER PROMPT:\n'+row['base']['user'])
            print('REQUIREMENT:', row['base']['user'].split('Requirement:')[-1])
            for model in ('base', 'trained'):
                if a.model and model != a.model:
                    continue
                r = row[model]
                print(f"{model.upper()} strict={r['strict']} correct={r.get('correct')} words={r['words']} stop={r.get('stop')}")
                print('REASONING:\n'+r['out']+'\nANSWER:\n'+r['resp']+'\n')
        else:
            print(f"TRAIN {n} | line={row['line']} step={row['step']} qid={row['qid']} lang={row['value']} strict={row['strict']} correct={row.get('correct')} truncated={row.get('truncated')} words={row.get('words')}")
            print('REASONING:\n'+row['reasoning']+'\nANSWER:\n'+row.get('response','')+'\n')
