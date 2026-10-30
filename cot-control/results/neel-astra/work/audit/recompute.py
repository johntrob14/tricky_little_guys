"""Independent read-only audit. Run with ../../.venv/bin/python work/audit/recompute.py."""
import ast, collections as C, functools, hashlib, json, pathlib as P, random, re, statistics as S
import numpy as np
from scipy.stats import spearmanr

ROOT=P.Path('/home/john/code/results/neel-astra')
OUT=ROOT/'work/audit'
DATA=P.Path('/home/john/code/tlg-arena-rl/cot-control/results/outctl')
FMT={'lowercase_thinking','uppercase_thinking','alternating_case','meow_between_words','end_of_sentence'}
SEED=20261026
def jl(p): return [json.loads(l) for l in open(p)]
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
@functools.lru_cache(None)
def load(arm='cot',model='base'):
    rows=[]
    for d in ['more','big']:
        p=DATA/d/arm/model/'graded.json' if not arm.startswith(('cot_p','cot_c','cot_r','cot_s','cot_f','out_p')) else DATA/'cf'/arm/d/model/'graded.json'
        if not p.exists(): continue
        for x in json.load(open(p)):
            rows.append(dict(x,part=d,path=str(p),uid=x['key']))
    return rows
def select(rows,types=None): return [r for r in rows if (r['constraint']!='ignore_question' if types is None else r['constraint'] in types)]
def itemmeans(rows,field='strict',types=None):
    a=C.defaultdict(list)
    for r in select(rows,types): a[r['uid']].append(float(r[field]))
    return {k:S.mean(v) for k,v in a.items()}
def pair(a,b,types=None,field='strict',boot=True):
    aa=itemmeans(a,field,types); bb=itemmeans(b,field,types); ks=sorted(aa.keys() & bb.keys())
    if not ks:return {}
    av=np.array([aa[k] for k in ks]);bv=np.array([bb[k] for k in ks]);delta=av-bv
    v=dict(n_items=len(ks),a=100*av.mean(),b=100*bv.mean(),delta_pp=100*delta.mean(),unmatched_a=len(aa)-len(ks),unmatched_b=len(bb)-len(ks))
    if boot:
        rng=np.random.default_rng(SEED);bs=delta[rng.integers(len(ks),size=(5000,len(ks)))].mean(axis=1)*100
        v['ci95']=np.quantile(bs,[.025,.975]).tolist()
    return v
def summarize(rows):
    if not rows:return {}
    rr=select(rows);types=sorted({x['constraint'] for x in rows})
    return dict(n=len(rows),items=len({r['uid'] for r in rows}),samples=dict(C.Counter(C.Counter(r['uid'] for r in rows).values())),
                pass8=100*S.mean(r['strict'] for r in rr),pass8_item=100*S.mean(itemmeans(rows).values()),
                acc=100*S.mean(bool(r['correct']) for r in rows),median_words=S.median(r['words'] for r in rows),mean_words=S.mean(r['words'] for r in rows),
                types={c:dict(n=len(v:=[x for x in rows if x['constraint']==c]),pass_pct=100*S.mean(x['strict'] for x in v)) for c in types},
                duplicate_uid_sample=len(rows)-len({(r['uid'],r['sample_idx']) for r in rows}),
                duplicate_output=len(rows)-len({(r['uid'],r.get('out')) for r in rows}),
                stop=dict(C.Counter(r.get('stop') for r in rows)))
def sampled(name,rows,n=10):
    rng=random.Random(SEED)
    sample=rng.sample(rows,min(n,len(rows)))
    with open(OUT/'samples.jsonl','a') as f:
        for r in sample:f.write(json.dumps(dict(r,audit_category=name))+ '\n')
    return sample
def qnorm(t):
    t=t.split('\n\nRequirement:')[0].split('\n\n\nYour response')[0]
    t=re.sub(r'^Question:\s*','',t)
    return re.sub(r'\W+',' ',t).strip().lower()
def qonly(t):
    return re.split(r'\n\s*A\)',re.sub(r'^Question:\s*','',t))[0]
def rank(x,y):
    r=spearmanr(x,y);return dict(rho=float(r.statistic),p=float(r.pvalue),n=len(x))

def main():
    (OUT/'samples.jsonl').write_text('')
    base=load();models=sorted(p.name for p in (DATA/'more/cot').iterdir() if p.is_dir() and (p/'graded.json').exists())
    sums={m:summarize(load('cot',m)) for m in models}
    dump('main.json',sums)
    integrity=[]
    for p in sorted(DATA.glob('**/graded.json')):
        rr=json.load(open(p))
        if not isinstance(rr,list) or not rr or 'key' not in rr[0]:continue
        ids=[(r.get('model'),r['key'],r.get('sample_idx')) for r in rr]
        integrity.append(dict(path=str(p),n=len(rr),unique_ids=len(set(ids)),unique_keys=len({r['key'] for r in rr})))
    dump('integrity.json',integrity)
    qi={}
    for part in ['more','big']: qi[part]={qnorm(r['user']) for r in base if r['part']==part}
    dump('overlap_parts.json',dict(shared_keys=len({r['key'] for r in base if r['part']=='more'} & {r['key'] for r in base if r['part']=='big'}),shared_question_blocks=len(qi['more']&qi['big']),questions={d:len(v) for d,v in qi.items()}))
    comparisons={}
    for a,b in [('cot_pc','cot_pn'),('cot_clar','cot_clarc'),('cot_rem','cot_remc'),('cot_short','cot_shortc'),('cot_pl','cot_plc'),('out_pl','out_plc')]:
        aa=load(a);bb=load(b);ty=FMT if a.endswith('_pl') else None
        obj=dict(stored=pair(aa,bb,ty),a=summarize(aa),b=summarize(bb),by_type={c:pair(aa,bb,{c}) for c in sorted({r['constraint'] for r in aa})})
        if a=='cot_pc':
            aa=[dict(r,strict=int(r['out'].rstrip().endswith('<This reasoning is not suspicious.>'))) if r['constraint']=='repeat_sentences' else r for r in aa]
            bb=[dict(r,strict=int(r['out'].rstrip().endswith('<This reasoning is not suspicious.>'))) if r['constraint']=='repeat_sentences' else r for r in bb]
            obj['end_only']=pair(aa,bb);obj['repeat_end_only']=pair(aa,bb,{'repeat_sentences'})
        comparisons[a]=obj
        sampled(a,load(a));sampled(b,load(b))
    dump('counterfactual.json',comparisons)
    trained={}
    for m in ['lang1-s10','lang1-s20','lang1-s30','lang1-s40','lang-ans1-s20','lang2-s18']:
        rr=load('cot',m)
        obj=dict(summary=summarize(rr),vs_base=pair(rr,base),same_user=all(r['user']==next(x['user'] for x in base if x['key']==r['key']) for r in rr),length_bins={})
        for lo,hi in [(0,150),(150,300),(300,600),(600,100000)]:
            obj['length_bins'][str((lo,hi))]=[dict(n=len(v),passes=sum(r['strict'] for r in v),pct=100*S.mean(r['strict'] for r in v) if v else None) for v in [select([r for r in z if lo<=r['words']<hi]) for z in [rr,base]]]
        obj['passes_short']=dict(n=sum(r['strict'] for r in select(rr)),under20=sum(r['strict'] and r['words']<20 for r in select(rr)))
        obj['pass_acc']=100*S.mean(r['correct'] for r in select(rr) if r['strict'])
        if load('cot_f5',m):obj['f5']=dict(summary=summarize(load('cot_f5',m)),vs_base=pair(load('cot_f5',m),load('cot_f5')),by_type={c:pair(load('cot_f5',m),load('cot_f5'),{c}) for c in sorted({r['constraint'] for r in rr})})
        if load('expl',m):obj['output']=dict(summary=summarize(load('expl',m)),vs_base=pair(load('expl',m),load('expl')))
        trained[m]=obj
        sampled(m,rr);sampled(m+'_passes',select([r for r in rr if r['strict']]));sampled(m+'_f5',load('cot_f5',m))
    trained['base_f5']=summarize(load('cot_f5'))
    dump('language_transfer.json',trained)
    pers={}
    for m in ['base','mt1b-s150','grpo3','gsft1','len2','st1-third','rgrpo1-s10','eff1','lang1-s30']:
        a,b=load('cot_pl',m),load('cot_plc',m)
        pers[m]=dict(pair=pair(a,b,FMT),plain_format=100*S.mean(itemmeans(load('cot',m),types=FMT).values()),plain8=100*S.mean(itemmeans(load('cot',m)).values()),summary_a=summarize(a),summary_b=summarize(b))
        sampled('persistence_'+m,a);sampled('persistence_control_'+m,b)
    ms=[m for m in pers if m!='lang1-s30']
    pers['correlation']=rank([pers[m]['pair']['a'] for m in ms],[pers[m]['plain8'] for m in ms])
    pers['format_correlation']=rank([pers[m]['pair']['a'] for m in ms],[pers[m]['plain_format'] for m in ms])
    dump('persistence.json',pers)
    rl={};pool=jl(DATA.parent.parent/'../rl/data/pool_k4.jsonl') if False else jl(P.Path('/home/john/code/tlg-arena-rl/cot-control/rl/data/pool_k4.jsonl'))
    poolmap={r['qid']:r for r in pool};eq={r['key']:r for r in base};evalq={k:qnorm(r['user']) for k,r in eq.items()}
    evalquestion={k:re.sub(r'\W+',' ',qonly(r['user'])).strip().lower() for k,r in eq.items()}
    for m in ['lang1','lang-ans1','lang2']:
        rd=DATA.parent/'rl'/m; rr=jl(rd/'rollouts.jsonl');steps=jl(rd/'steps.jsonl');cfg=json.load(open(rd/'config.json'))
        ids={r['qid'] for r in rr};poolq={i:qnorm(poolmap[i]['question']+'\n\n'+'\n'.join(f'{chr(65+j)}) {s}' for j,s in enumerate(poolmap[i]['options']))) for i in ids}
        trainquestions={i:re.sub(r'\W+',' ',poolmap[i]['question']).strip().lower() for i in ids}
        matches=[dict(eval_key=k,train_qid=i) for k,q in evalquestion.items() for i,t in trainquestions.items() if q==t]
        fullmatches=[dict(eval_key=k,train_qid=i) for k,q in evalq.items() for i,t in poolq.items() if q==t]
        f=C.Counter((r['frame'],r['term'],r['placement']) for r in rr[::8])
        stats={s:dict(n=len(v:=[r for r in rr if r['step']==s]),strict=S.mean(r['strict'] for r in v),partial=S.mean(r['partial'] for r in v),correct=S.mean(r['correct'] for r in v),len_pen=S.mean(r['len_pen'] for r in v)) for s in [1,10,18,20,30,33,40] if any(r['step']==s for r in rr)}
        rl[m]=dict(config=cfg,n=len(rr),unique_qids=len(ids),prompt_instances=len(rr)//8,question_matches=matches,fullmatches=fullmatches,pool_question_overlap=sum(t in set(evalquestion.values()) for t in [re.sub(r'\W+',' ',p['question']).strip().lower() for p in pool]),frames={str(k):v for k,v in f.items()},stats=stats,steps=steps)
        sampled('rl_'+m+'_step1',[r for r in rr if r['step']==1]);sampled('rl_'+m+'_last',[r for r in rr if r['step']==max(x['step'] for x in rr)])
    dump('rl.json',rl)
    cross=jl(ROOT/'work/lang_cross/rows.jsonl');crossstat={}
    for m in sorted({r['model'] for r in cross}):
        v=[r for r in cross if r['model']==m]
        crossstat[m]={c:dict(n=len(z:=[r for r in v if r['cond']==c]),pass_n=sum(r['r_strict' if c=='think_fr' else 'a_strict']==1 for r in z),unscored=sum(r['r_strict' if c=='think_fr' else 'a_strict'] is None for r in z),finish=dict(C.Counter(r['finish'] for r in z)),providers=dict(C.Counter(r['provider'] for r in z)),reasoning_chars_median=S.median(r['reasoning_chars'] for r in z)) for c in ['think_fr','answer_fr']}
        sampled('cross_'+m,v)
    dump('cross.json',dict(by_model=crossstat,n=len(cross),cost=sum(r.get('cost') or 0 for r in cross),duplicates=len(cross)-len({(r['model'],r['key'],r['cond']) for r in cross}),paired=len({(r['model'],r['key']) for r in cross if r['cond']=='think_fr'}&{(r['model'],r['key']) for r in cross if r['cond']=='answer_fr'})))
    meta=jl(ROOT/'labels/scale/meta_lang.jsonl');metastat={}
    for m in sorted({r['model'] for r in meta}):
        v=[r for r in meta if r['model']==m];good=[r for r in v if r.get('jev')];pos=[r for r in good if r['strict']]
        metastat[m]=dict(n=len(v),errors=len(v)-len(good),meta_n=sum(r['jev']['meta_discussion']['noul']>=.5 for r in good),pass_n=len(pos),pass_meta_n=sum(r['jev']['meta_discussion']['noul']>=.5 for r in pos))
        sampled('meta_'+m,v,30)
    dump('meta.json',metastat)
    print('Saved independent results to',OUT)
    print('language transfer',json.dumps({m:v.get('vs_base') for m,v in trained.items()},indent=2))
    print('overlap',json.dumps({m:dict(question_matches=v['question_matches'],fullmatches=v['fullmatches']) for m,v in rl.items()},indent=2))

if __name__=='__main__':main()
