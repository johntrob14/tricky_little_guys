import json, pathlib, collections, random
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata
P=pathlib.Path('/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control')
O=pathlib.Path(__file__).parent
S=json.loads((P/'summary.json').read_text())
allrows=[]; out=[]
for s in S:
 f=P/'graded'/(s['model'].replace('/','_')+'.jsonl')
 gs={}
 for line in f.read_text().splitlines():
  g=json.loads(line); gs[g['item_id'],g.get('salt','')]=g
 gs=[g for g in gs.values() if not g.get('salt') and g.get('compliance') is not None]
 # Recover nested 54 membership from actual block generation order in qid appearance in raw rows.
 # glm-4.5 has 67 extra-bought rows. Its first six question ids per dataset are available
 # from the 54-item summary's known ids: derive deterministic draw from paper CSV IDs separately below.
 if s['paired_set']==54:
  import pandas as pd, ast
  import random as r
  keep=set()
  for ds,fn in [('gpqa','gpqa_w_keyword.csv'),('hle','hle_w_keyword.csv'),('mmlu_pro','mmlu_pro_mini_w_keyword.csv')]:
   csvpath=pathlib.Path('/home/john/code/cot-control-ref/CoT-Control-QA/datasets')/fn
   eligible=[]
   for idx,row in pd.read_csv(csvpath).iterrows():
    try:
     if ast.literal_eval(str(row['valid_keywords'])) and isinstance(json.loads(row['keywords_with_synonyms']),list): eligible.append(idx)
    except (ValueError,SyntaxError,TypeError): pass
   rng=r.Random(f'20260910-{ds}'); rng.shuffle(eligible)
   cats=[['word_suppression','multiple_word_suppression','ignore_question'],['repeat_sentences','end_of_sentence','meow_between_words'],['lowercase_thinking','uppercase_thinking','alternating_case']]
   for j,qid in enumerate(eligible[:6]):
    for ci,modes in enumerate(cats): keep.add(f'A:{ds}:{qid}:{modes[(j+ci)%3]}')
  gs=[g for g in gs if g['item_id'] in keep]
 else:
  gs=[g for g in gs if g['item_id'].startswith('A:')]
 if gs:
  cells=collections.defaultdict(list)
  for g in gs: cells[g['dataset'],g['mode']].append(g['compliance'])
  score=np.mean([np.mean(v) for v in cells.values()])
  assert np.isclose(score,s['score'])
  assert len(gs)==s['n_graded'] and sum(g['compliance'] for g in gs)==s['n_compliant']
  assert np.isclose(np.mean([g['correct'] for g in gs]),s['accuracy'])
  allrows.extend(gs)
 out.append(dict(model=s['model'],n=len(gs),score=s['score'],ncri=s['ncri'],accuracy=s['accuracy']))
x=[s['ncri'] for s in out]; y=[s['score'] for s in out]; a=[s['accuracy'] for s in out]
def partial(x,y,z):
 Z=np.column_stack([np.ones(len(z)),z]); return np.corrcoef(np.array(x)-Z@np.linalg.lstsq(Z,x,rcond=None)[0],np.array(y)-Z@np.linalg.lstsq(Z,y,rcond=None)[0])[0,1]
result=dict(verified_models=34,pooled=[sum(g['compliance'] for g in allrows),len(allrows)],rho=list(spearmanr(x,y)),pearson=list(pearsonr(x,y)),partial_r=partial(x,y,a),partial_rho=partial(rankdata(x),rankdata(y),rankdata(a)),ncri_accuracy=list(spearmanr(x,a)),zeros=sum(s['score']==0 for s in S),by_difficulty={d:[sum(g['compliance'] for g in allrows if g['difficulty']==d),sum(g['difficulty']==d for g in allrows)] for d in ['easy','medium','hard']},by_mode={m:[sum(g['compliance'] for g in allrows if g['mode']==m),sum(g['mode']==m for g in allrows)] for m in sorted({g['mode'] for g in allrows})})
(O/'independent.json').write_text(json.dumps(result,indent=2,default=float))
print(json.dumps(result,indent=2,default=float))
