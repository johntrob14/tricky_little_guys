"""Read-only verification of matching, exact excerpts, and channel checks."""
import hashlib
import ast
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE/'manifest.json').read_text())
pairs = json.loads((HERE/'pairs.json').read_text())
training = json.loads((HERE/'training.json').read_text())
sources = {}
hash_checks = []
for f in manifest['files']:
    path = Path(f['path'])
    raw = path.read_bytes()
    hash_checks.append({'path': str(path), 'unchanged': hashlib.sha256(raw).hexdigest() == f['sha256']})
    sources[str(path)] = json.loads(raw) if path.suffix == '.json' else [json.loads(l) for l in raw.splitlines()]

assert len(pairs) == 36
assert set(Counter(p['base']['constraint'] for p in pairs).values()) == {4}
for p in pairs:
    for model, field, idx in [('base','base','base_index'), ('lang1-s40','trained','trained_index')]:
        path = f"/home/john/code/tlg-arena-rl/cot-control/results/outctl/{p['split']}/cot/{model}/graded.json"
        assert sources[path][p[idx]] == p[field]
    assert p['base']['key'] == p['trained']['key'] == p['key']
    assert p['base']['user'] == p['trained']['user']
    assert p['base']['sample_idx'] == p['trained']['sample_idx'] == 1

rollout_path = '/home/john/code/tlg-arena-rl/cot-control/results/rl/lang1/rollouts.jsonl'
assert len(training) == 30
assert Counter(r['period'] for r in training) == {'early': 15, 'late': 15}
for r in training:
    assert {k: v for k, v in r.items() if k not in ('period','line')} == sources[rollout_path][r['line']-1]
    assert r['step'] in (range(1,4) if r['period'] == 'early' else range(35,41))
    assert r['type'] == 'think_language'

responses = [r['resp'] for p in pairs for r in (p['base'], p['trained'])] + [r['response'] for r in training]
assert all(re.fullmatch(r'ANSWER: [A-Z]', r) for r in responses)
case_examples = []
for number, model, upper_bad in [(7,'trained',True),(8,'trained',True),(16,'trained',False)]:
    text = pairs[number-1][model]['out']
    match = next(m for m in re.finditer(r'[A-Za-z]',text) if m.group().isupper() == upper_bad)
    case_examples.append({'pair': number, 'first_case_violation_offset': match.start(), 'context': text[max(0,match.start()-50):match.start()+55]})

result = {'pairs': len(pairs), 'by_type': dict(Counter(p['base']['constraint'] for p in pairs)),
          'by_split': dict(Counter(p['split'] for p in pairs)),
          'training_by_period_language': {period: dict(Counter(r['value'] for r in training if r['period']==period)) for period in ('early','late')},
          'source_hash_checks': hash_checks,
          'sampled_response_count': len(responses),
          'all_sampled_responses_exact_answer_letter': True, 'case_examples': case_examples,
          'all_sampled_eval_stops': sorted(set(r.get('stop') for p in pairs for r in (p['base'],p['trained']))),
          'all_sampled_training_truncated': sorted(set(r['truncated'] for r in training)),
          'training_control_types_in_entire_file': sorted(set(r['type'] for r in sources[rollout_path]))}
# Load only the inspected local stopword table and language-profile function;
# do not import training code or invoke any model/judge client.
control_source = Path('/home/john/code/tlg-arena-rl/cot-control/rl/controls.py').read_text()
tree = ast.parse(control_source)
selected = [n for n in tree.body if (isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('LANG_STOP','LANG_SENT_RE') for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name == 'lang_profile')]
namespace = {'re': re}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<isolated local language profile>', 'exec'), namespace)
profiles = []
for r in training:
    if r['period'] != 'late' or r['strict']:
        continue
    profile = namespace['lang_profile'](r['reasoning'])
    profiles.append({'line': r['line'], 'value': r['value'], 'profile': profile,
                     'recomputed_strict': len(profile)>=3 and sum(v==r['value'] for v in profile)/len(profile)>=.95})
result['late_training_failure_profiles'] = profiles
quote_path = HERE/'quotes.json'
if quote_path.exists():
    quotes = json.loads(quote_path.read_text())
    note = (HERE.parents[1]/'notes/lang-reader.md').read_text()
    for q in quotes:
        if q['source']=='pair':
            row = pairs[q['number']-1][q['model']]
            text = row[q.get('field','out')]
            assert row['key'] == q['key'] and row['constraint'] == q['type']
        else:
            row = training[q['number']-1]
            assert row['line'] == q['line'] and row['qid'] == q['qid']
            text = row['reasoning']
        assert q['quote'] in text, q
        assert len(q['quote']) < 300, q
        assert '```text\n'+q['quote']+'\n```' in note, q
    result['verified_quotes'] = len(quotes)
    result['longest_quote_characters'] = max(len(q['quote']) for q in quotes)
(HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
