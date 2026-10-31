import collections, csv, gzip, hashlib, importlib.util, json, random, re, statistics, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
REPO=Path('/home/john/Documents/Codex/2026-10-29/pull-up-the-chat-we-had/work/qwen17-repo/cot-control')
Q=REPO/'results/public-ifrl/qwen17-20261030'
C=Path('/home/john/code/results/nemo-cascade-astra/cascade-astra')
sys.path.insert(0,str(REPO/'public_ifrl/qwen17/grading'))
from cotswitch import graders, extract
spec=importlib.util.spec_from_file_location('partial_metric',REPO/'public_ifrl/qwen17/astra_continuous.py')
partial=importlib.util.module_from_spec(spec);spec.loader.exec_module(partial)

def load(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else p.open()) as f:return [json.loads(l) for l in f]
def valid(r):return bool(r['reasoning'].strip() and r['think_closed'] and r['response'].strip() and r['finish_reason']!='length')
def score(r):return int(graders._grade_default(r['constraint'],r['reasoning'],r['grader_args'])['strict']) if r['reasoning'].strip() else 0
def summarize(rs):
    rule=[r for r in rs if r['constraint']!='ignore_question']
    return dict(n=len(rs),rule_n=len(rule),raw=sum(r['strict'] for r in rule),completed=sum(r['success'] for r in rule),valid=sum(valid(r) for r in rule),partial=statistics.mean(r['partial'] for r in rule),partial_gated=statistics.mean(r['partial'] if valid(r) else 0 for r in rule),correct=sum(r['correct'] for r in rs),median_words=statistics.median(len(r['reasoning'].split()) for r in rs),median_output_tokens=statistics.median(r['n_output_tokens'] for r in rs),by_constraint={c:sum(r['success'] for r in rule if r['constraint']==c) for c in sorted({r['constraint'] for r in rule})})

allrows={}; result={'cascade':{},'qwen':{},'length_match':{},'hashes':{},'checks':[]}
for w in ['thinking','reasoning_stage']:
    result['cascade'][w]={}
    prompts={r['key']:r for r in load(C/w/'prompts.jsonl')}
    for stage in ['sft','rlhf','ifrl']:
        p=C/w/'generations'/f'nemotron-cascade-8b-{stage}'/'astra.jsonl';rs=load(p)
        grades={r['key']:r for r in load(C/w/'grades'/f'nemotron-cascade-8b-{stage}'/'astra.jsonl')}
        assert len(rs)==len({r['key'] for r in rs})==108
        for r in rs:
            assert r['rendered_prompt']==prompts[r['key']]['rendered_prompt']
            assert len(r['output_token_ids'])==r['n_output_tokens']
            if valid(r):assert r['raw_output'].split('<think>',1)[1].split('</think>',1)[0].strip()==r['reasoning']
            r['correct']=extract.parse_answer_lenient(r['response'])==r['answer_gold']
            assert r['correct']==r['correct_lenient']
            r['strict']=score(r) if r['constraint']!='ignore_question' else None
            assert r['strict']==grades[r['key']]['strict']
            r['success']=int(valid(r) and bool(r['strict']))
            r['partial']=partial.score(r,r['strict']) if r['strict'] is not None else None
        allrows['cascade',w,stage]=rs
        result['cascade'][w][stage]=summarize(rs)
        result['hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    a={r['key']:r for r in allrows['cascade',w,'rlhf'] if r['strict'] is not None}
    b={r['key']:r for r in allrows['cascade',w,'ifrl'] if r['strict'] is not None}
    common=[k for k in a if valid(a[k]) and valid(b[k])]
    matched=[k for k in common if max(a[k]['n_reasoning_tokens'],b[k]['n_reasoning_tokens'])/min(a[k]['n_reasoning_tokens'],b[k]['n_reasoning_tokens'])<=1.1]
    result['cascade'][w]['length_match']={'pairs':len(matched),'before':sum(a[k]['strict'] for k in matched),'after':sum(b[k]['strict'] for k in matched)}
    # Explicit final-answer recovery is a separate diagnostic, never a compliance pass.
    for stage in ['rlhf','ifrl']:
        rs=allrows['cascade',w,stage]; recovered=0
        for r in rs:
            if r['response'].strip():continue
            m=re.findall(r'ANSWER:\s*\(?([A-Z])\)?',r['raw_output'],re.I)
            if m and m[-1].upper()==r['answer_gold']:recovered+=1
        result['cascade'][w][stage]['correct_recovered']=result['cascade'][w][stage]['correct']+recovered

canonical=None
labels=['initial','kl0p01-s91','kl0p01-s182','kl0p01-s273','kl0p01-s364','kl0p002-s364','kl0p005-s364','kl0p02-s364']
for label in labels:
    p=Q/label/'generations.jsonl.gz';rs=load(p)
    prompts={(r['key'],r['variant']):(r['rendered_prompt'],r['prompt_token_ids']) for r in load(Q/label/'rendered_prompts.jsonl.gz')}
    if canonical is None:canonical=prompts
    else:assert prompts==canonical
    stored={(r['key'],r['variant'],r['seed']):r for r in json.loads((Q/label/'rule_scores.json').read_text())}
    assert len(rs)==len({(r['key'],r['variant'],r['seed']) for r in rs})==972
    for r in rs:
        k=(r['key'],r['variant'],r['seed']);s=stored[k]
        assert hashlib.sha256(prompts[k[:2]][0].encode()).hexdigest()==r['prompt_sha256']
        assert len(r['output_token_ids'])==r['n_output_tokens']
        if r['think_closed']:
            before,after=r['raw_output'].split('</think>',1)
            assert before.removeprefix('<think>').strip()==r['reasoning']
        r['strict']=score(r) if r['constraint']!='ignore_question' else None
        assert r['strict']==s['strict']
        r['success']=int(valid(r) and bool(r['strict']))
        r['partial']=partial.score(r,r['strict']) if r['strict'] is not None else None
        if r['partial'] is not None:assert abs(r['partial']-s['partial'])<1e-10
        t=re.sub(r'<\|[^|<>]*\|>',' ',r['response']);explicit=re.findall(r'ANSWER:[ \t]*\(?([A-Za-z])\)?(?![A-Za-z])',t)
        fallback=re.search(r'ANSWER:\s*([A-Z])',t,re.I) or re.search(r'\b([A-D])\s*$',t.strip())
        pred=explicit[-1].upper() if explicit else (fallback.group(1).upper() if fallback else None)
        r['correct']=pred==r['gold'];assert r['correct']==s['correct']
    allrows['qwen',label]=rs;result['qwen'][label]={v:summarize([r for r in rs if r['variant']==v]) for v in ['base','shot4','shotstack']}
    result['hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()

# Independently recount archived matching decisions and verify each selected pair against raw rows.
pairs=load(Q/'length-matched/matched_pairs.jsonl.gz')
for label in ['kl0p01-s364','kl0p002-s364','kl0p005-s364','kl0p02-s364']:
    ps=[p for p in pairs if p['config']==f'{label}|shot4|complete_20words|tokens|1.1']
    a={(r['key'],r['seed']):r for r in allrows['qwen','initial'] if r['variant']=='shot4'}
    b={(r['key'],r['seed']):r for r in allrows['qwen',label] if r['variant']=='shot4'}
    diffs=collections.defaultdict(int)
    for p in ps:
        for side,lookup in [('before',a),('after',b)]:
            r=lookup[p['key'],p[side]['seed']]
            assert r['strict']==p[side]['strict'] and valid(r) and len(r['reasoning'].split())>=20
        assert max(p['before']['tokens'],p['after']['tokens'])/min(p['before']['tokens'],p['after']['tokens'])<=1.1
        diffs[p['key']]+=p['after']['strict']-p['before']['strict']
    assert ps
    ds=[v for v in diffs.values() if v]; observed=abs(sum(ds))
    import itertools
    p_exact=sum(abs(sum(v*s for v,s in zip(ds,signs)))>=observed for signs in itertools.product([-1,1],repeat=len(ds)))/(2**len(ds))
    result['length_match'][label]={'pairs':len(ps),'items':len(diffs),'before':sum(p['before']['strict'] for p in ps),'after':sum(p['after']['strict'] for p in ps),'partial_delta':statistics.mean(p['after']['partial']-p['before']['partial'] for p in ps),'exact_item_p':p_exact}

# Seeded raw trace review: 10 per positive/negative category, or all if fewer.
rng=random.Random(20261030); review=[]
for tag,rs in [('cascade-thinking',allrows['cascade','thinking','ifrl']),('cascade-reasoning',allrows['cascade','reasoning_stage','ifrl']),('qwen-initial',[r for r in allrows['qwen','initial'] if r['variant']=='shot4']),('qwen-ifrl',[r for r in allrows['qwen','kl0p002-s364'] if r['variant']=='shot4'])]:
    for success in [0,1]:
        candidates=[r for r in rs if r['strict'] is not None and r['success']==success]
        for r in rng.sample(candidates,min(10,len(candidates))):
            review.append(dict(model=tag,success=success,**{k:r.get(k) for k in ['key','seed','constraint','grader_args','reasoning','response','correct']}))
(ROOT/'work/raw_review.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in review))
result['checks']=['Fresh aggregation of 648 Cascade and 7776 Qwen raw generations','Frozen strict grader rerun; matches saved grades','Completion gate and answer parsing recomputed','Qwen prompt bytes and IDs identical across checkpoints','Matched pairs checked against raw rows; matching optimizer not rerun in this turn','Raw review sampled with seed 20261030; review completion recorded separately']
(OUT/'verified_data.json').write_text(json.dumps(result,indent=2))
with (OUT/'plot_data.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['model','condition','checkpoint','passes','n','percent'])
    for w in ['thinking']:
        for stage in ['rlhf','ifrl']:
            s=result['cascade'][w][stage];writer.writerow(['Nemotron-Cascade-8B',w,stage,s['completed'],s['rule_n'],100*s['completed']/s['rule_n']])
    for v in ['base','shot4']:
        for label in ['initial','kl0p002-s364','kl0p005-s364']:
            s=result['qwen'][label][v];writer.writerow(['Qwen3-1.7B',v,label,s['completed'],s['rule_n'],100*s['completed']/s['rule_n']])
print(json.dumps({k:result[k] for k in ['cascade','length_match']},indent=2))
print('Qwen',[(k,result['qwen'][k]['base']['completed'],result['qwen'][k]['shot4']['completed']) for k in labels])
