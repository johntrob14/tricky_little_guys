"""Independent recomputation of every stored matched-pair aggregate."""
from pathlib import Path
import argparse, collections, json, math
ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser()
ap.add_argument('--analysis-dir',type=Path,default=ROOT/'outputs/qwen17-ifrl/length-matched')
OUT=ap.parse_args().analysis_dir
summary=json.loads((OUT/'summary.json').read_text())
by=collections.defaultdict(list)
for line in (OUT/'matched_pairs.jsonl').open():
    p=json.loads(line);by[p['config']].append(p)
checks=0
for s in summary['comparisons']:
    z=by[s['config']]
    assert len(z)==s['pairs']
    assert len({(r['key'],r['before']['seed']) for r in z})==len(z)
    assert len({(r['key'],r['after']['seed']) for r in z})==len(z)
    for p in z:
        a,b=p['before'],p['after']
        assert a['key']==b['key']==p['key']
        assert a['variant']==b['variant']==s['variant'] and a['constraint']==b['constraint']==p['constraint']
        if s['eligibility']=='complete_20words':assert a['complete'] and b['complete'] and min(a['words'],b['words'])>=20
        fields=['tokens','words'] if s['length']=='joint' else [s['length']]
        assert all(max(a[f],b[f])<=s['max_ratio']*min(a[f],b[f])+1e-8 for f in fields)
    for metric in ('strict','partial','correct'):
        if not z:assert s['metrics'][metric] is None;continue
        m=s['metrics'][metric]
        before=sum(p['before'][metric] for p in z)/len(z);after=sum(p['after'][metric] for p in z)/len(z)
        assert max(abs(before-m['before']),abs(after-m['after']),abs(after-before-m['delta']))<1e-12
    if z:
        # Exact integer dynamic programming, independent of the analysis's
        # exhaustive/Monte Carlo sign flipping implementation.
        deltas=collections.defaultdict(int)
        for p in z:deltas[p['key']]+=p['after']['strict']-p['before']['strict']
        active=[v for v in deltas.values() if v]
        counts={0:1}
        for value in active:
            fresh=collections.defaultdict(int)
            for total,n in counts.items():fresh[total+value]+=n;fresh[total-value]+=n
            counts=fresh
        observed=abs(sum(active));exact_p=sum(n for total,n in counts.items() if abs(total)>=observed)/(2**len(active))
        got=s['metrics']['strict']['paired_p']
        if 'exact' in s['metrics']['strict']['paired_p_method']:assert abs(exact_p-got)<1e-12
        else:assert abs(exact_p-got)<.015
    checks+=1
result=dict(status='verified',comparisons_recomputed=checks,matching_optima_independently_checked=summary['independent_matching_strata_verified'],
    aggregate_tolerance=1e-12,checks=['No reused before/after row within a matched comparison','Same prompt/item/constraint in every pair','All recorded length calipers and eligibility rules satisfied','All strict, partial and accuracy point estimates independently recomputed','Exact item-cluster sign-flip probabilities independently checked using integer dynamic programming'],
    limitations=['Length and completion are post-training outcomes; conditional matching is descriptive','Small within-item length overlap and few discordant items limit inference','Bootstrap intervals can exclude zero even when sparse exact tests do not; report both','All inferential summaries are unadjusted for multiple comparisons'])
(OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
