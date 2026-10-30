"""Independent aggregation from raw inputs and raw scorer JSONL."""
import collections
import hashlib
import json
import pathlib

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT
manifest=json.load(open(ROOT/'manifest.json'))
reported=json.load(open(OUT/'results.json'))
input_by_id={}
score_by_id={}
for i in range(3):
    for row in map(json.loads,(ROOT/f'items_{i}.jsonl').open()):
        assert row['id'] not in input_by_id
        input_by_id[row['id']]=row
    for row in map(json.loads,(ROOT/f'scores_{i}.jsonl').open()):
        assert row['id'] not in score_by_id
        score_by_id[row['id']]=row
assert len(input_by_id)==len(score_by_id)==450
values=collections.defaultdict(list)
ratings=collections.defaultdict(collections.Counter)
for m in manifest:
    for view,id_field in [('full','id'),('excerpt','excerpt_id')]:
        rid=m[id_field];inp=input_by_id[rid];score=score_by_id[rid]
        n=sum(u['words'] for u in inp['units'])
        f=0
        seen=set()
        for first,last,label in score['runs']:
            for j in range(first,last+1):
                assert j not in seen
                seen.add(j)
                if label=='F':f+=inp['units'][j]['words']
        assert len(seen)==len(inp['units'])
        key=(view,m['model'],m['channel'])
        values[key].append(f/n)
        if view=='full':ratings[key][str(score['whole_rating'])]+=1
for group in reported['groups']:
    key=group['view'],group['model'],group['channel']
    assert len(values[key])==35
    mean=sum(values[key])/35
    assert abs(mean-group['F_mean'])<1e-12
    if key[0]=='full':assert dict(ratings[key])==group['whole_counts']
for scenario in {r['scenario_id'] for r in manifest}:
    ns={sum(u['words'] for u in input_by_id[r['excerpt_id']]['units'])
        for r in manifest if r['scenario_id']==scenario}
    assert len(ns)==1
for path,h in reported['config']['source_sha256'].items():
    assert hashlib.sha256((ROOT.parent / pathlib.Path(path).parent.name / pathlib.Path(path).name).read_bytes()).hexdigest()==h
checks={'all_group_means_independently_reproduced':True,
        'all_whole_rating_counts_independently_reproduced':True,
        'all_six_excerpt_lengths_match_within_every_scenario':True,
        'all_source_hashes_still_match':True,
        'n_primary_full':210,'n_primary_excerpt':210,'n_independent_repeats':30,
        'no_new_target_model_calls':True,
        'limitations':['Scorers are Codex models; no human validation.',
                      'The parent excerpt audit is hypothesis-aware.',
                      'Advice domain,35scenarios,one chosen saved rep0 each.',
                      'Rating of presentation does not identify computational function.']}
(OUT/'independent_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
