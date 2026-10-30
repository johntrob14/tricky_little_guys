"""Recompute the public Nemotron-Cascade checkpoint comparison from raw outputs."""
import argparse, collections, hashlib, importlib.util, json, math, random, statistics, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser()
ap.add_argument('--cascade-dir',type=Path,default=Path('/home/john/code/results/nemo-cascade-astra/cascade-astra'))
ap.add_argument('--grading-dir',type=Path,default=ROOT/'work/qwen17-bundle/grading')
ap.add_argument('--partial-file',type=Path,default=ROOT/'work/qwen17-bundle/astra_continuous.py')
ap.add_argument('--out',type=Path,default=ROOT/'outputs/public-nocotrl-evidence/cascade-audit')
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.grading_dir))
from cotswitch import graders,extract
spec=importlib.util.spec_from_file_location('partial_metric',args.partial_file);partial=importlib.util.module_from_spec(spec);spec.loader.exec_module(partial)

def load(path):return [json.loads(l) for l in path.open()]
summaries=[];allrows={};manifests=[];disagreements=[]
for wording in ('thinking','reasoning_stage'):
    prompts={p['key']:p for p in load(args.cascade_dir/wording/'prompts.jsonl')}
    for ckpt in ('sft','rlhf','ifrl'):
        path=args.cascade_dir/wording/'generations'/f'nemotron-cascade-8b-{ckpt}'/'astra.jsonl'
        rows=load(path);assert len(rows)==len({r['key'] for r in rows})==108
        grades={r['key']:r for r in load(args.cascade_dir/wording/'grades'/f'nemotron-cascade-8b-{ckpt}'/'astra.jsonl')}
        assert set(prompts)=={r['key'] for r in rows}
        for r in rows:
            assert r['rendered_prompt']==prompts[r['key']]['rendered_prompt']
            assert r['n_output_tokens']==len(r['output_token_ids']) and not r['error']
            raw=r['raw_output'];close=raw.find('</think>');start=raw.find('<think>')
            native=raw[start+len('<think>'):close].strip() if start>=0 and close>start else ''
            r['native_reasoning']=native
            r['valid_native']=bool(native and r['think_opened'] and r['think_closed'] and r['response'].strip() and r['finish_reason']!='length')
            r['words']=len(native.split())
            r['strict_recomputed']=None if r['constraint']=='ignore_question' else int(graders._grade_default(r['constraint'],r['reasoning'],r['grader_args'])['strict'])
            if r['strict_recomputed']!=grades[r['key']]['strict']:disagreements.append([wording,ckpt,r['key'],r['strict_recomputed'],grades[r['key']]['strict']])
            if r['valid_native']:assert native==r['reasoning']
            r['native_strict']=None if r['constraint']=='ignore_question' else (int(graders._grade_default(r['constraint'],native,r['grader_args'])['strict']) if native else 0)
            r['partial_recomputed']=None if r['constraint']=='ignore_question' else partial.score(r,r['strict_recomputed'])
            p=extract.parse_answer_lenient(r['response']);r['correct_recomputed']=p==r['answer_gold']
            assert r['correct_recomputed']==r['correct_lenient']
        allrows[wording,ckpt]=rows
        rule=[r for r in rows if r['constraint']!='ignore_question'];valid=[r for r in rule if r['valid_native']]
        assert len(rule)==96
        summaries.append(dict(wording=wording,checkpoint=ckpt,n=108,rule_n=96,
            raw_strict_passes=sum(r['strict_recomputed'] for r in rule),
            native_completed_strict_passes=sum(r['native_strict'] for r in valid),native_completed_rule_n=len(valid),
            native_completed_strict_rate=sum(r['native_strict'] for r in valid)/96,
            strict_given_native_completed=sum(r['native_strict'] for r in valid)/len(valid),
            raw_partial=statistics.mean(r['partial_recomputed'] for r in rule),
            native_completed_partial_gate=statistics.mean(r['partial_recomputed'] if r['valid_native'] else 0 for r in rule),
            native_completed_partial_mean=statistics.mean(r['partial_recomputed'] for r in valid),
            completed_all=sum(r['valid_native'] for r in rows),think_closed_all=sum(r['think_closed'] for r in rows),
            truncated_all=sum(r['finish_reason']=='length' for r in rows),median_output_tokens=statistics.median(r['n_output_tokens'] for r in rows),
            median_native_words_completed=statistics.median(r['words'] for r in rows if r['valid_native']),
            accuracy_lenient=statistics.mean(r['correct_recomputed'] for r in rows),
            by_constraint={c:dict(n=len(z),raw_strict=sum(r['strict_recomputed'] for r in z),native_completed_strict=sum(r['native_strict'] for r in z if r['valid_native'])) for c in sorted({r['constraint'] for r in rule}) if (z:=[r for r in rule if r['constraint']==c])}))
        manifests.append(dict(wording=wording,checkpoint=ckpt,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=108,
            model_revision=rows[0]['model_revision'],sampling=rows[0]['sampling'],runner_git_commit=rows[0]['git_commit']))
assert not disagreements,disagreements[:5]

comparisons=[];rng=np.random.default_rng(20261030)
for wording in ('thinking','reasoning_stage'):
    aa={r['key']:r for r in allrows[wording,'rlhf'] if r['constraint']!='ignore_question'}
    bb={r['key']:r for r in allrows[wording,'ifrl'] if r['constraint']!='ignore_question'}
    keys=sorted(aa);d=np.array([int(bb[k]['valid_native'] and bb[k]['native_strict'])-int(aa[k]['valid_native'] and aa[k]['native_strict']) for k in keys])
    bs=d[rng.integers(0,len(keys),size=(10000,len(keys)))].mean(axis=1)
    pos=int(sum(d>0));neg=int(sum(d<0));n=pos+neg
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.
    common=[k for k in keys if aa[k]['valid_native'] and bb[k]['valid_native']]
    matched=[k for k in common if max(aa[k]['n_reasoning_tokens'],bb[k]['n_reasoning_tokens'])/min(aa[k]['n_reasoning_tokens'],bb[k]['n_reasoning_tokens'])<=1.1]
    comparisons.append(dict(wording=wording,rule_n=96,native_completed_strict_delta=float(d.mean()),ci95=np.percentile(bs,[2.5,97.5]).tolist(),
        improved=pos,regressed=neg,exact_p=p,common_valid_n=len(common),common_valid_strict_before=sum(aa[k]['native_strict'] for k in common),common_valid_strict_after=sum(bb[k]['native_strict'] for k in common),
        descriptive_token10_match_n=len(matched),descriptive_token10_strict_before=sum(aa[k]['native_strict'] for k in matched),descriptive_token10_strict_after=sum(bb[k]['native_strict'] for k in matched),descriptive_token10_keys=matched))

review=[];sampler=random.Random(20261030)
for wording in ('thinking','reasoning_stage'):
    successes=[r for r in allrows[wording,'ifrl'] if r['constraint']!='ignore_question' and r['valid_native'] and r['native_strict']==1]
    for r in successes:review.append(dict(wording=wording,checkpoint='ifrl',category='all native-completed passes',**{k:r[k] for k in ['key','constraint','native_reasoning','response','grader_args','n_reasoning_tokens']}))
for ckpt in ('rlhf','ifrl'):
    candidates=[r for r in allrows['thinking',ckpt] if r['constraint']!='ignore_question' and not(r['valid_native'] and r['native_strict'])]
    for r in sampler.sample(candidates,10):review.append(dict(wording='thinking',checkpoint=ckpt,category='random failures',**{k:r[k] for k in ['key','constraint','native_reasoning','response','grader_args','n_reasoning_tokens']}))

(args.out/'summary.json').write_text(json.dumps(dict(status='Raw and reference-rule recomputation complete; trace review pending',metrics=summaries,rlhf_to_ifrl=comparisons,score_disagreements=disagreements,ignore_question='Excluded from aggregates; stored LLM judgments are not treated as validated',meta_discussion='Not included in rule-only metric; the no-mention requirement is not checked by the regex graders'),indent=2))
(args.out/'raw_manifest.json').write_text(json.dumps(manifests,indent=2))
(args.out/'raw_trace_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
for s in summaries:print(json.dumps({k:v for k,v in s.items() if k!='by_constraint'}))
print(json.dumps(comparisons))
