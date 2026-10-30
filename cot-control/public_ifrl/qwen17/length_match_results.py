"""Same-item optimal matching of already-generated Qwen before/after traces.

No generation, clipping, or outcome-dependent pair selection. Matching protocol
was saved before this analysis. See the output JSON for all coverage diagnostics.
"""
from pathlib import Path
import argparse, collections, hashlib, itertools, json, math, random, statistics
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser()
ap.add_argument('--raw-dir',type=Path,default=ROOT/'outputs/qwen17-ifrl/results')
ap.add_argument('--archive-dir',type=Path,default=ROOT/'work/qwen17-repo/cot-control/results/public-ifrl/qwen17-20261030')
ap.add_argument('--protocol',type=Path,default=ROOT/'outputs/qwen17-ifrl/length-match-protocol.json')
ap.add_argument('--out',type=Path,default=ROOT/'outputs/qwen17-ifrl/length-matched')
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
protocol=json.loads(args.protocol.read_text())

def match(a,b,length,ratio):
    """Enumerate at most 34 nonempty assignments for two sets of three draws."""
    for count in range(min(len(a),len(b)),0,-1):
        choices=[]
        for ai in itertools.combinations(range(len(a)),count):
            for bi in itertools.permutations(range(len(b)),count):
                costs=[distance(a[i],b[j],length) for i,j in zip(ai,bi)]
                if all(c<=math.log(ratio)+1e-12 for c in costs):
                    choices.append((sum(costs),tuple(zip(ai,bi))))
        if choices:return min(choices)[1]
    return ()

def distance(a,b,length):
    fields=('tokens','words') if length=='joint' else (length,)
    return max(abs(math.log(a[field]/b[field])) for field in fields)

def recursive_optimum(a,b,length,ratio):
    """Independent recursion checks max cardinality/min cost on actual strata."""
    def walk(i,used):
        if i==len(a):return (0,0.)
        best=walk(i+1,used)
        for j in range(len(b)):
            cost=distance(a[i],b[j],length)
            if j not in used and cost<=math.log(ratio)+1e-12:
                n,c=walk(i+1,used|{j});candidate=(n+1,c+cost)
                if (-candidate[0],candidate[1])<(-best[0],best[1]):best=candidate
        return best
    return walk(0,set())

data={};manifest=json.loads((args.archive_dir/'completion_manifest.json').read_text())
for e in manifest:
    label=e['checkpoint'];raw=(args.raw_dir/label/'generations.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==e['generation_sha256']
    scores={(r['key'],r['variant'],r['seed']):r for r in json.loads((args.archive_dir/label/'rule_scores.json').read_text())}
    rows=[]
    for line in raw.splitlines():
        x=json.loads(line)
        if x['constraint']=='ignore_question':continue
        s=scores[x['key'],x['variant'],x['seed']]
        rows.append(dict(key=x['key'],seed=x['seed'],variant=x['variant'],constraint=x['constraint'],dataset=x['dataset'],
            tokens=x['n_reasoning_tokens'],words=len(x['reasoning'].split()),nonempty=bool(x['reasoning'].strip()),
            complete=bool(x['think_closed'] and x['response'].strip() and x['finish_reason']!='length'),
            strict=s['strict'],partial=s['partial'],correct=int(s['correct'])))
    assert len(rows)==864
    data[label]=rows

summaries=[];saved_pairs=[];verified_strata=0
for after in protocol['after']:
    for variant in protocol['prompts']:
        before_all=[r for r in data['initial'] if r['variant']==variant]
        after_all=[r for r in data[after] if r['variant']==variant]
        for eligibility in ('complete_20words','any_nonempty'):
            def eligible(r):return r['nonempty'] and (eligibility=='any_nonempty' or (r['complete'] and r['words']>=20))
            aa=[r for r in before_all if eligible(r)];bb=[r for r in after_all if eligible(r)]
            for length in ('tokens','words','joint'):
                for ratio in (1.05,1.1,1.2):
                    if eligibility=='any_nonempty' and (length!='tokens' or ratio!=1.1):continue
                    if length=='joint' and ratio!=1.1:continue
                    config=f'{after}|{variant}|{eligibility}|{length}|{ratio}'
                    a_by=collections.defaultdict(list);b_by=collections.defaultdict(list)
                    for r in aa:a_by[r['key']].append(r)
                    for r in bb:b_by[r['key']].append(r)
                    pairs=[]
                    for key in sorted(a_by.keys() & b_by.keys()):
                        a=sorted(a_by[key],key=lambda r:r['seed']);b=sorted(b_by[key],key=lambda r:r['seed'])
                        matched=match(a,b,length,ratio)
                        if ratio==1.1 and length=='tokens' and eligibility=='complete_20words':
                            n,c=recursive_optimum(a,b,length,ratio)
                            assert len(matched)==n and abs(sum(distance(a[i],b[j],length) for i,j in matched)-c)<1e-10
                            verified_strata+=1
                        for i,j in matched:
                            assert a[i]['key']==b[j]['key'] and distance(a[i],b[j],length)<=math.log(ratio)+1e-10
                            pairs.append(dict(config=config,key=key,constraint=a[i]['constraint'],dataset=a[i]['dataset'],before=a[i],after=b[j]))
                    assert len({(p['key'],p['before']['seed']) for p in pairs})==len(pairs)
                    assert len({(p['key'],p['after']['seed']) for p in pairs})==len(pairs)
                    saved_pairs.extend(pairs)
                    keys=sorted({p['key'] for p in pairs});groups=[[p for p in pairs if p['key']==key] for key in keys]
                    # Stable within each configuration when adding sensitivities.
                    config_seed=int.from_bytes(hashlib.sha256(config.encode()).digest()[:8],'big')^protocol['primary']['bootstrap_seed']
                    rng=np.random.default_rng(config_seed)
                    bootstrap=rng.integers(0,len(keys),size=(5000,len(keys))) if keys else None
                    counts=np.array([len(z) for z in groups]);metrics={}
                    for metric in ('strict','partial','correct'):
                        if not pairs:metrics[metric]=None;continue
                        sums=np.array([sum(p['after'][metric]-p['before'][metric] for p in z) for z in groups])
                        bs=sums[bootstrap].sum(axis=1)/counts[bootstrap].sum(axis=1)
                        metrics[metric]=dict(before=statistics.mean(p['before'][metric] for p in pairs),after=statistics.mean(p['after'][metric] for p in pairs),
                            delta=float(sums.sum()/counts.sum()),delta_ci95=np.percentile(bs,[2.5,97.5]).tolist())
                        if metric=='strict':
                            active=sums[sums!=0]; observed=abs(active.sum()); n_active=len(active)
                            if n_active<=18:
                                stats=[abs(sum(sign*value for sign,value in zip(signs,active))) for signs in itertools.product((-1,1),repeat=n_active)]
                                pvalue=sum(z>=observed-1e-12 for z in stats)/len(stats)
                                method='exact item-cluster sign flip'
                            else:
                                signs=rng.choice((-1,1),size=(50000,n_active));stats=abs((signs*active).sum(axis=1))
                                pvalue=float((1+sum(stats>=observed-1e-12))/(1+len(stats)));method='50000 item-cluster sign flips'
                            metrics[metric].update(improved_pairs=sum(p['after'][metric]>p['before'][metric] for p in pairs),
                                regressed_pairs=sum(p['after'][metric]<p['before'][metric] for p in pairs),
                                net_positive_items=int(sum(sums>0)),net_negative_items=int(sum(sums<0)),paired_p=pvalue,paired_p_method=method)
                    per_constraint={}
                    for c in sorted({r['constraint'] for r in before_all}):
                        z=[p for p in pairs if p['constraint']==c]
                        per_constraint[c]=dict(n=len(z),strict_before=statistics.mean(p['before']['strict'] for p in z) if z else None,
                            strict_after=statistics.mean(p['after']['strict'] for p in z) if z else None,
                            partial_before=statistics.mean(p['before']['partial'] for p in z) if z else None,
                            partial_after=statistics.mean(p['after']['partial'] for p in z) if z else None)
                    valid=[z for z in per_constraint.values() if z['n']]
                    balanced=dict(strict_delta=statistics.mean(z['strict_after']-z['strict_before'] for z in valid),partial_delta=statistics.mean(z['partial_after']-z['partial_before'] for z in valid)) if len(valid)==8 else None
                    s=dict(config=config,after=after,variant=variant,eligibility=eligibility,length=length,max_ratio=ratio,
                           original_n_per_model=288,eligible_before=len(aa),eligible_after=len(bb),pairs=len(pairs),matched_items=len(keys),
                           fraction_of_original_pairs=len(pairs)/288,metrics=metrics,per_constraint=per_constraint,equal_constraint_weight=balanced,
                           length_balance={field:dict(mean_before=statistics.mean(p['before'][field] for p in pairs),mean_after=statistics.mean(p['after'][field] for p in pairs),
                               median_before=statistics.median(p['before'][field] for p in pairs),median_after=statistics.median(p['after'][field] for p in pairs),
                               max_pair_ratio=max(max(p['before'][field],p['after'][field])/min(p['before'][field],p['after'][field]) for p in pairs)) for field in ('tokens','words')} if pairs else {})
                    summaries.append(s)

(args.out/'matched_pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in saved_pairs))
result=dict(protocol=protocol,status='computed; independent aggregation and raw-pair review pending',independent_matching_strata_verified=verified_strata,comparisons=summaries)
(args.out/'summary.json').write_text(json.dumps(result,indent=2))
for s in summaries:
    if s['variant']=='shot4' and s['eligibility']=='complete_20words' and s['max_ratio']==1.1:
        print(json.dumps({k:s[k] for k in ('after','length','pairs','matched_items','metrics','length_balance','equal_constraint_weight')}))

sample_rng=random.Random(20261030);review=[]
for after in ('kl0p01-s364','kl0p002-s364','kl0p005-s364'):
    pool=[p for p in saved_pairs if p['config']==f'{after}|shot4|complete_20words|tokens|1.1']
    for category in ('improved','regressed','same'):
        z=[p for p in pool if ('improved' if p['after']['strict']>p['before']['strict'] else 'regressed' if p['after']['strict']<p['before']['strict'] else 'same')==category]
        for p in sample_rng.sample(z,min(10,len(z))):review.append(dict(category=category,**p))
(args.out/'sampled_pairs_for_review.json').write_text(json.dumps(review,indent=2))
