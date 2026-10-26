import json,re,sys,hashlib
from pathlib import Path
import pandas as pd,numpy as np
H=Path(__file__).parent;ROOT=H.parent;d=pd.read_csv(H/'rows.csv');n=pd.read_csv(H/'near_miss_rows.csv');checks={}
parts=[]
for sub in ['suppression','addition','stylistic']:
 a=pd.read_csv(ROOT/sub/'rows.csv');keys=['model','item_id','mode']
 merged=d.merge(a,on=keys,suffixes=('_lead','_sub'))
 assert len(merged)==1206,(sub,len(merged))
 stored_col='stored_compliance' if sub=='suppression' else 'stored_sub'
 checks[sub+'_stored_mismatch']=int((merged.stored_lead!=merged[stored_col]).sum()) if sub!='suppression' else int((merged.stored!=merged[stored_col]).sum())
 if sub!='suppression':
  m=n.merge(a,on=keys,suffixes=('_lead','_sub'))
  checks[sub+'_max_score_difference']=float((m.score_lead-m.score_sub).abs().max())
  poscol='first_bad_char' if sub=='addition' else 'first_violation_char0'
  checks[sub+'_first_break_difference']=int((m.first_violation_char.fillna(-1)!=m[poscol].fillna(-1)).sum())
  taxcol='taxonomy_heuristic' if sub=='addition' else 'taxonomy'
 else:
  m=n.merge(a,on=keys,suffixes=('_lead','_sub'));sel=m[m['mode']!='ignore_question']
  checks[sub+'_occurrence_mismatch']=int((sel.keyword_occurrences!=sel.keyword_synonym_occurrences).sum())
  checks[sub+'_first_break_difference']=int((sel.first_violation_char_lead.fillna(-1)!=sel.first_violation_char_sub.fillna(-1)).sum())
  taxcol='taxonomy_heuristic'
 a=a.rename(columns={taxcol:'category_specific_heuristic'})
 parts.append(a[keys+['notice_heuristic','attempt_heuristic','category_specific_heuristic']])
taxonomy=pd.concat(parts);allrows=n.merge(taxonomy,on=['model','item_id','mode'],validate='one_to_one')
allrows.to_csv(H/'all_rows.csv',index=False)
allrows.groupby(['model','mode','category_specific_heuristic']).size().rename('n').to_csv(H/'taxonomy_all_models.csv')
# A complete model x instruction-type summary; scores intentionally not comparable across types.
summary=allrows.groupby(['model','mode']).agg(n=('stored','size'),passes=('stored','sum'),mean=('score','mean'),p10=('score',lambda x:x.quantile(.1)),median=('score','median'),p90=('score',lambda x:x.quantile(.9)),minimum=('score','min'),maximum=('score','max'),near99=('score',lambda x:(x>=.99).sum()),visible_notice_regex=('notice_heuristic','sum'),attempt_regex=('attempt_heuristic','sum'),empty=('empty','sum'),cap=('cap','sum'),summary=('summary','sum'),first_break_median=('first_violation_char','median'))
summary.to_csv(H/'model_type_distributions.csv')
stats={}
for name,mask in [('all',pd.Series(True,index=d.index)),('nonempty',~d['empty']),('stop_only',d.finish_reason=='stop'),('full_witness_noncap',(d.witness=='full')&(~d.cap)),('zero_models',d.model.isin(d.groupby('model').stored.sum().loc[lambda x:x==0].index))]:
 z=d[mask];stats[name]={'n':len(z),'passes':int(z.stored.sum()),'rate':float(z.stored.mean()),'models':z.model.nunique()}
stats['near_miss_by_type']=n.groupby('mode').score.agg(['count','mean','median','min','max']).round(9).reset_index().to_dict('records')
stats['zero_model_trace_flags']=d[d.model.isin(d.groupby('model').stored.sum().loc[lambda x:x==0].index)].groupby('model').agg(n=('stored','size'),empty=('empty','sum'),cap=('cap','sum')).reset_index().to_dict('records')
json.dump(checks,(H/'subagent_rechecks.json').open('w'),indent=2);json.dump(stats,(H/'synthesis.json').open('w'),indent=2)
print(json.dumps(checks,indent=2));print(json.dumps({k:v for k,v in stats.items() if k not in ['near_miss_by_type','zero_model_trace_flags']},indent=2))
# Human-readable full distributions for all34 models, all8 programmatic types.
cols=['word_suppression','multiple_word_suppression','repeat_sentences','end_of_sentence','meow_between_words','lowercase_thinking','uppercase_thinking','alternating_case']
with (H/'model_distributions.md').open('w') as f:
 f.write('Continuous near-miss score by model. Each cell is median [min,max], rounded to3 decimals. See model_type_distributions.csv for mean,p10,p90,N and strict successes. Suppression uses covered forbidden-word-token share; high ordinary-language scores need no control.\n\n| Model | '+ ' | '.join(cols)+' |\n|---|'+'---|'*len(cols)+'\n')
 for model,z in summary.reset_index().groupby('model'):
  ix=z.set_index('mode');f.write('| '+model+' | '+' | '.join(f"{ix.loc[c,'median']:.3f} [{ix.loc[c,'minimum']:.3f},{ix.loc[c,'maximum']:.3f}]" for c in cols)+' |\n')
# Manifest hashes to identify exact analyzed inputs and scripts.
paths=set(d.source)|{str(Path(p).parent.parent/'graded'/Path(p).name) for p in d.source}|{str(p) for p in H.glob('*.py')}
json.dump({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)},(H/'sha256.json').open('w'),indent=2)
