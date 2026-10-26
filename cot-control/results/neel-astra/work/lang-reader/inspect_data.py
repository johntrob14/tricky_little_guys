import json
from collections import Counter
from pathlib import Path

root = Path('/home/john/code/tlg-arena-rl/cot-control/results')
for split in ('more', 'big'):
    for model in ('base', 'lang1-s40'):
        path = root / 'outctl' / split / 'cot' / model / 'graded.json'
        data = json.loads(path.read_text())
        print(path, type(data).__name__)
        if isinstance(data, dict):
            print('keys', list(data)[:20])
        else:
            print('n', len(data), 'types', Counter(str(r.get('constraint')) for r in data))
            print('first', json.dumps(data[0], ensure_ascii=False)[:5000])
path = root / 'rl/lang1/rollouts.jsonl'
rows = [json.loads(l) for l in path.open()]
print('rollouts', len(rows), 'steps', Counter(r.get('step') for r in rows))
print('first', json.dumps(rows[0], ensure_ascii=False)[:8000])
