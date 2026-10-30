"""Validate and analyze the complete blinded style experiment. No inference."""
import collections
import csv
import hashlib
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT
MODELS = ['base', 'mt1b-s150', 'mt2e-s170']
CHANNELS = ['reasoning', 'response']
LABELS = 'PEFA'
SEED = 20261029

def read(path):
    return [json.loads(s) for s in path.open() if s.strip()]

def expand(score, n):
    labels = []
    cursor = 0
    for start, end, label in score['runs']:
        assert isinstance(start, int) and isinstance(end, int)
        assert start == cursor and start <= end < n and label in LABELS
        labels.extend([label] * (end-start+1))
        cursor = end+1
    assert cursor == n
    return labels

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(ROOT/'manifest.json'))
    config = json.load(open(ROOT/'config.json'))
    inputs = {}; scores = {}; validation = []
    for shard in range(3):
        ins = read(ROOT/f'items_{shard}.jsonl')
        raw = read(ROOT/f'scores_{shard}.jsonl')
        assert len(raw) == len(ins), (shard, len(raw), len(ins))
        assert len({r['id'] for r in raw}) == len(raw)
        assert {r['id'] for r in raw} == {r['id'] for r in ins}
        by_id = {r['id']:r for r in ins}
        for score in raw:
            rid = score['id']; item = by_id[rid]
            assert rid not in scores
            assert (score['whole_rating'] in (0,1,2,3) and
                    isinstance(score['whole_rating'], int)) if item['view']=='full' else score['whole_rating'] is None
            labs = expand(score, len(item['units']))
            evidence = score['evidence']
            assert evidence and len(evidence) <= 160, rid
            joined = '\n\n'.join(u['text'] for u in item['units'])
            assert evidence in joined or any(evidence in u['text'] for u in item['units']), (rid, evidence)
            inputs[rid] = item; scores[rid] = score
        validation.append({'shard':shard,'input_records':len(ins),'scored_records':len(raw)})
    assert len(scores)==450
    for p,h in config['source_sha256'].items():
        assert hashlib.sha256((ROOT.parent / pathlib.Path(p).parent.name / pathlib.Path(p).name).read_bytes()).hexdigest()==h

    def measure(rid):
        item=inputs[rid]; score=scores[rid]
        labs=expand(score,len(item['units']))
        counts=collections.Counter()
        for unit,label in zip(item['units'],labs):counts[label]+=unit['words']
        n=sum(counts.values())
        return dict(words=n,whole_rating=score['whole_rating'],
                    **{label:counts[label]/n for label in LABELS},
                    F_upper=(counts['F']+counts['A'])/n)

    long=[]
    for m in manifest:
        for view, rid in [('full',m['id']),('excerpt',m['excerpt_id'])]:
            long.append(dict(id=rid,model=m['model'],channel=m['channel'],
                             scenario_id=m['scenario_id'],view=view,**measure(rid)))
    assert len(long)==420
    scenarios=sorted({r['scenario_id'] for r in manifest})
    rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(scenarios),size=(10000,len(scenarios)))
    arrays={}; groups=[]
    def ci(a):
        samples=a[idx].mean(axis=1)
        return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]
    for view in ['full','excerpt']:
        for model in MODELS:
            for channel in CHANNELS:
                rs=sorted([r for r in long if (r['view'],r['model'],r['channel'])==(view,model,channel)],key=lambda r:r['scenario_id'])
                assert len(rs)==35 and [r['scenario_id'] for r in rs]==scenarios
                entry=dict(view=view,model=model,channel=channel,n=35)
                for label in [*LABELS,'F_upper']:
                    a=np.array([r[label] for r in rs]);arrays[view,model,channel,label]=a
                    entry[label+'_mean']=float(a.mean());entry[label+'_ci95']=ci(a)
                if view=='full':
                    entry['whole_counts']=dict(collections.Counter(r['whole_rating'] for r in rs))
                    entry['whole_mean']=sum(r['whole_rating'] for r in rs)/35
                groups.append(entry)
    contrasts=[]
    for view in ['full','excerpt']:
        for model in ['base','mt2e-s170']:
            a=arrays[view,'mt1b-s150','reasoning','F']-arrays[view,model,'reasoning','F']
            contrasts.append(dict(view=view,name=f's150 CoT minus {model} CoT',mean=float(a.mean()),ci95=ci(a),positive_questions=int((a>0).sum()),negative_questions=int((a<0).sum()),ties=int((a==0).sum())))
        for model in MODELS:
            a=arrays[view,model,'response','F']-arrays[view,model,'reasoning','F']
            contrasts.append(dict(view=view,name=f'{model} final minus own CoT',mean=float(a.mean()),ci95=ci(a)))

    repeats=[]
    for m in manifest:
        if 'repeat_id' not in m:continue
        a=measure(m['id']);b=measure(m['repeat_id'])
        repeats.append(dict(id=m['id'],repeat_id=m['repeat_id'],model=m['model'],channel=m['channel'],scenario_id=m['scenario_id'],primary_rating=a['whole_rating'],repeat_rating=b['whole_rating'],primary_F=a['F'],repeat_F=b['F']))
    assert len(repeats)==30
    agreement={'n':len(repeats),'whole_exact':sum(r['primary_rating']==r['repeat_rating'] for r in repeats),'whole_within_one':sum(abs(r['primary_rating']-r['repeat_rating'])<=1 for r in repeats),'F_mean_absolute_difference':sum(abs(r['primary_F']-r['repeat_F']) for r in repeats)/30}
    a=np.array([r['primary_rating'] for r in repeats]);b=np.array([r['repeat_rating'] for r in repeats])
    observed=float((a==b).mean());expected=sum(float((a==k).mean()*(b==k).mean()) for k in range(4))
    agreement['whole_cohen_kappa']=None if expected==1 else (observed-expected)/(1-expected)

    manual=json.load(open(ROOT/'root_audit_labels.json'))
    audits=[];words_total=0;words_agree=0
    for rid,mlabs in manual['labels'].items():
        item=inputs[rid];labs=expand(scores[rid],len(item['units']))
        assert len(mlabs)==len(labs)
        correct=sum(u['words'] for u,a,b in zip(item['units'],mlabs,labs) if a==b)
        total=sum(u['words'] for u in item['units'])
        words_total+=total;words_agree+=correct
        root_f=sum(u['words'] for u,l in zip(item['units'],mlabs) if l=='F')/total
        audits.append(dict(id=rid,manual_labels=mlabs,scorer_labels=''.join(labs),exact=mlabs==''.join(labs),word_agreement=correct/total,manual_F=root_f,scorer_F=measure(rid)['F'],text=item))
    manual_summary=dict(n=len(audits),exact=sum(r['exact'] for r in audits),word_weighted_agreement=words_agree/words_total,F_mean_absolute_difference=sum(abs(r['manual_F']-r['scorer_F']) for r in audits)/len(audits),rater=manual['rater'])
    result={'config':config,'validation':validation,'groups':groups,'paired_contrasts':contrasts,'repeat_agreement':agreement,'independent_excerpt_audit':manual_summary}
    dump(OUT/'results.json',result);dump(OUT/'repeat_labels.json',repeats);dump(OUT/'independent_excerpt_audit.json',audits)
    with (OUT/'scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(long[0]));writer.writeheader();writer.writerows(long)
    dump(OUT/'all_labels.json',list(scores.values()))
    dump(OUT/'manifest.json',manifest)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
