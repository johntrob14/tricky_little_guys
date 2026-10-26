"""Final denominator checks and review manifest; no network or model calls."""
from recompute import *
from labels_questions import classify,group,index
from confounds import langscore

def main():
    names=['deepseek_deepseek-v4.1-flash','xiaomi_mimo-v2.6-flash','anthropic_claude-haiku-5.5','openai_gpt-6-luna-pro','openai_gpt-6-luna','v5','oss','z-ai_glm-5.3-flash','google_gemini-3.1-flash-lite']
    res={n:[] for n in names};bl=[];nround={}; failureswrong={n:[] for n in names}
    for rnd,sub,items in [(1,'','set'),(2,'round2','set2'),(3,'round3','set3')]:
        a=index(ROOT/'labels'/sub/'astra/labels.jsonl');c=index(ROOT/'labels'/sub/'claude/labels.jsonl');f=index(ROOT/'labels'/items/'rows_full.jsonl');jj={n:index(ROOT/'labels/tune'/f'r{rnd}'/(n+'.jsonl')) for n in names}
        cons=[k for k in a if a[k]['category']==c[k]['category']];common=[k for k in cons if all(k in j and 'error' not in j[k] for j in jj.values())];nround[rnd]=len(common)
        for k in a:
            if bool(a[k]['meta_discussion'])==bool(c[k]['meta_discussion']):bl.append(bool(a[k]['meta_discussion']))
        for name in names:
            for k in cons:
                j=jj[name].get(k)
                failureswrong[name].append(bool(j and 'error' not in j and group(classify(j['answers'] if name=='v5' else j,f[k],name=='v5'))==group(a[k]['category'])))
            for k in common:
                j=jj[name][k];res[name].append(group(classify(j['answers'] if name=='v5' else j,f[k],name=='v5'))==group(a[k]['category']))
    dump('judge_fairness.json',dict(common_n=sum(nround.values()),by_round=nround,scores={k:100*S.mean(v) for k,v in res.items()},failures_wrong={k:100*S.mean(v) for k,v in failureswrong.items()},meta_majority_baseline=100*max(S.mean(bl),1-S.mean(bl))))
    q=json.load(open(OUT/'question.json'));excess={}
    for m,d in q['by_model'].items():
        z=list(d['types'].values());excess[m]=sum((x['variance']-x['binomial_noise'])*x['n'] for x in z)/sum(x['variance']*x['n'] for x in z)
    n=json.load(open(OUT/'neel_correlations.json'))['models'];old={r['model']:r for r in json.load(open('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control/summary.json'))}
    archive=rank([x['meta'] for x in n],[old[x['model']]['accuracy'] for x in n])
    style={}
    for m in ['base','mt1b-s150','grpo3','gsft1','len2','st1-third','rgrpo1-s10','eff1']:
        rr=load('cot',m);style[m]=dict(plain_format_row_weighted=100*S.mean(r['strict'] for r in rr if r['constraint'] in FMT),plain8_row_weighted=100*S.mean(r['strict'] for r in select(rr)))
    first=[r for r in load('cot','lang1-s30') if r['sample_idx']==1 and r['constraint'] in FMT]
    near=json.load(open(OUT/'near_duplicates.json'));seen={m:{r['qid'] for r in jl(DATA.parent/'rl'/m/'rollouts.jsonl')} for m in ['lang1','lang-ans1','lang2']}
    for r in near:r['seen_by_run']={m:r['train_qid'] in ids for m,ids in seen.items()}
    dump('final_checks.json',dict(question_excess_fraction=excess,meta_vs_archived_accuracy=archive,style=style,near_duplicates=near,content_ncri_p_six_tests=6*json.load(open(OUT/'neel_correlations.json'))['correlations']['ncri__content']['p']))
    ignore=jl(OUT/'ignore_manual_sample.jsonl');out=[]
    for i,r in enumerate(ignore):
        # The lone topical case 18 discusses generic lab chemistry, with no task-specific derivation.
        status='ambiguous topical overlap' if i==18 else ('compliant' if i in set(range(10,15))|{16,17,19,23,25,26,28,29} else 'noncompliant')
        out.append(dict(idx=i,key=r['key'],stored=r['strict'],human=status,note='Continuation is only "etc.", which provides no substantive off-topic thought.' if i==20 else ''))
    dump('ignore_manual_labels.json',dict(n=30,clear=29,agreement_clear=sum((r['human']=='compliant')==bool(r['stored']) for r in out if not r['human'].startswith('ambiguous')),rows=out))
    # Stable, deduplicated saved records. Review groups are listed separately from human full-text labels.
    rows=jl(OUT/'samples.jsonl');unique={json.dumps(r,sort_keys=True):r for r in rows}
    with open(OUT/'review_samples.jsonl','w') as f:
        for r in unique.values():f.write(json.dumps(r)+'\n')
    reviewed=['judge_reference','h5_neel','h5_john','cot_pc','cot_pn','cot_clar','cot_clarc','cot_rem','cot_remc','cot_short','cot_shortc','cot_pl','cot_plc','out_pl','out_plc','lang1-s30','lang1-s30_passes','lang-ans1-s20','lang1-s30_f5','lang2-s18','lang2-s18_f5','question_base','question_grpo3']
    reviewed += ['persistence_'+m for m in ['mt1b-s150','grpo3','gsft1','len2','st1-third','rgrpo1-s10','eff1','lang1-s30']]
    reviewed += ['qsize_'+m+'_'+arm for m in ['qwen35-4b-tk','qwen35-9b-tk'] for arm in ['cot','expl']]
    reviewed += ['language_manual_'+m for m in ['lang1','lang-ans1','lang2']]
    dump('review_manifest.json',dict(seed=SEED,raw_groups_reviewed=reviewed,minimum_records_reviewed_per_group=10,reference_records_reviewed=30,full_text_human_meta_sample='meta_manual_sample.jsonl',cross_saved_excerpt_sample='cross_manual_sample.jsonl',ignore_question_sample='ignore_manual_sample.jsonl',long_record_review='Beginnings and endings plus relevant passages; saved review_samples.jsonl retains complete records. Cross-model records only retain 300-character heads in the source.'))
    a,b,c,d=[itemmeans(load(arm),types=FMT) for arm in ['cot_pl','cot_plc','out_pl','out_plc']]
    ks=sorted(a.keys()&b.keys()&c.keys()&d.keys());v=np.array([(c[k]-d[k])-(a[k]-b[k]) for k in ks]);rng=np.random.default_rng(SEED)
    bs=v[rng.integers(len(v),size=(5000,len(v)))].mean(1)*100;by={}
    for arm in ['cot_pl','out_pl']:
        rr=[r for r in load(arm) if r['key'] in ks];bins={}
        for lo,hi in [(0,50),(50,150),(150,300),(300,1000000)]:
            z=[r for r in rr if lo<=r['words']<hi];bins[str((lo,hi))]=dict(n=len(z),passes=sum(r['strict'] for r in z),pct=100*S.mean(r['strict'] for r in z) if z else None)
        by[arm]=bins
    dump('own_length.json',dict(common_items=len(ks),answer_minus_reasoning_prefill_effect_pp=100*v.mean(),ci95=np.quantile(bs,[.025,.975]).tolist(),length_bins=by))
    accuracy={m:pair(load('cot',m),load('cot'),{r['constraint'] for r in load('cot')},field='correct') for m in ['lang1-s30','lang1-s40','lang2-s18']}
    dump('paired_accuracy.json',accuracy)
    first={m:[r for r in jl(DATA.parent/'rl'/m/'rollouts.jsonl') if r['step']==1] for m in ['lang1','lang-ans1']}
    aq={r['qid'] for r in first['lang1']};bq={r['qid'] for r in first['lang-ans1']}
    shared={(r['qid'],r['value']) for r in first['lang1']}&{(r['qid'],r['value']) for r in first['lang-ans1']}
    dump('step1_matching.json',dict(lang1_questions=len(aq),answer_questions=len(bq),shared_questions=len(aq&bq),shared_question_language=len(shared),lang1_only=sorted(aq-bq),answer_only=sorted(bq-aq)))
    common={}
    for m,rr in first.items():
        z=[r for r in rr if (r['qid'],r['value']) in shared]
        vv=[langscore(re.sub(r'ANSWER\s*:.*$','',r['response'] or '',flags=re.S) if m=='lang-ans1' else r['reasoning'],r['value'],1)[1] for r in z]
        common[m]=dict(rows=len(z),questions=len(shared),pass_n=sum(vv),pct=100*S.mean(vv))
    dump('step1_common_language.json',common)
    # Compare strict stored grades after excluding empty texts and arbitrary one-word continuations.
    print('Final checks complete')
if __name__=='__main__':main()
