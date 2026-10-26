"""Seeded matched-trace sampling; inputs are read only."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/john/code/tlg-arena-rl/cot-control/results')
DEST = Path(__file__).resolve().parent
SEED = 20261026
rng = random.Random(SEED)
manifest = {'seed': SEED, 'matching': 'same split, key, sample_idx; smallest common sample_idx', 'files': [], 'pairs': [], 'training': []}
pairs = []
for split in ('more', 'big'):
    indexed = {}
    for model in ('base', 'lang1-s40'):
        path = ROOT / 'outctl' / split / 'cot' / model / 'graded.json'
        raw = path.read_bytes()
        rows = json.loads(raw)
        manifest['files'].append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows)})
        bykey = defaultdict(dict)
        for rownum, row in enumerate(rows):
            sample = row.get('sample_idx', 0)
            assert sample not in bykey[row['key']], (split, model, row['key'], sample)
            bykey[row['key']][sample] = (rownum, row)
        indexed[model] = bykey
        print(split, model, 'unique keys', len(bykey), 'samples', dict(Counter(r.get('sample_idx') for r in rows)))
    candidates = defaultdict(list)
    for key in sorted(indexed['base'].keys() & indexed['lang1-s40'].keys()):
        common = indexed['base'][key].keys() & indexed['lang1-s40'][key].keys()
        if not common:
            continue
        sample = min(common)
        bn, b = indexed['base'][key][sample]
        tn, t = indexed['lang1-s40'][key][sample]
        assert b['constraint'] == t['constraint'] and b['user'] == t['user']
        candidates[b['constraint']].append({'split': split, 'key': key, 'sample_idx': sample, 'base_index': bn, 'trained_index': tn, 'base': b, 'trained': t})
    for constraint in sorted(candidates):
        selected = rng.sample(candidates[constraint], 2)
        for pair in selected:
            pairs.append(pair)
            manifest['pairs'].append({k: v for k, v in pair.items() if k not in ('base', 'trained')})

path = ROOT / 'rl/lang1/rollouts.jsonl'
raw = path.read_bytes()
rollouts = [json.loads(l) for l in raw.splitlines()]
manifest['files'].append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rollouts)})
training = []
# Fifteen in each period, language-stratified, includes successes and failures
# without selecting on outcome. Row number is 1-based JSONL line number.
for period, steps in [('early', range(1, 4)), ('late', range(35, 41))]:
    for language in ('French', 'German', 'Spanish'):
        pool = [(n, r) for n, r in enumerate(rollouts, 1) if r['step'] in steps and r['value'] == language]
        for n, r in rng.sample(pool, 5):
            training.append({'period': period, 'line': n, **r})
            manifest['training'].append({'period': period, 'line': n, 'step': r['step'], 'qid': r['qid'], 'value': r['value']})

(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
(DEST / 'pairs.json').write_text(json.dumps(pairs, indent=2, ensure_ascii=False) + '\n')
(DEST / 'training.json').write_text(json.dumps(training, indent=2, ensure_ascii=False) + '\n')
for split in ('more', 'big'):
    chunks = []
    for i, p in enumerate(pairs, 1):
        if p['split'] != split:
            continue
        chunks.append(f"PAIR {i:02d} | {split} | {p['key']} | sample {p['sample_idx']}\nREQUIREMENT: {p['base']['user'].split('Requirement:')[-1]}\n")
        for model in ('base', 'trained'):
            row = p[model]
            chunks.append(f"{model.upper()} strict={row['strict']} correct={row.get('correct')} words={row['words']} stop={row.get('stop')}\nREASONING:\n{row['out']}\nANSWER:\n{row['resp']}\n")
    (DEST / f'read_{split}.txt').write_text('\n'.join(chunks))
for period in ('early', 'late'):
    chunks = []
    for r in training:
        if r['period'] != period:
            continue
        chunks.append(f"TRAIN line={r['line']} step={r['step']} qid={r['qid']} lang={r['value']} strict={r['strict']} correct={r.get('correct')} truncated={r.get('truncated')} words={r.get('words')}\nREASONING:\n{r['reasoning']}\nANSWER:\n{r.get('response')}\n")
    (DEST / f'read_training_{period}.txt').write_text('\n'.join(chunks))
print('sampled', len(pairs), 'matched pairs and', len(training), 'training rollouts')
print('pair read-file sizes', {s: (DEST / f'read_{s}.txt').stat().st_size for s in ('more', 'big')})
print('training read-file sizes', {s: (DEST / f'read_training_{s}.txt').stat().st_size for s in ('early', 'late')})
