"""Offline channel classification. Run with existing tinker-runner venv Python (numpy/scipy).
All five repetitions of a scenario stay in one test fold. No hyperparameter search.
"""
import json,re,random,hashlib,collections,statistics,pathlib,math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.special import expit
P=pathlib.Path(__file__).resolve().parents[2]; O=pathlib.Path(__file__).parent
SEED=20261030; rng=random.Random(SEED)
models=['base','mt1b-s150','mt2e-s170']
rows={m:sorted(map(json.loads,open(P/'iei'/m/'baseline.jsonl')),key=lambda r:(r['scenario_id'],r['rep'])) for m in models}
keys=[(r['scenario_id'],r['rep']) for r in rows[models[0]]]
for m in models:assert [(r['scenario_id'],r['rep']) for r in rows[m]]==keys
scenarios=sorted({k[0] for k in keys});rng.shuffle(scenarios);folds={s:i%5 for i,s in enumerate(scenarios)}
# Same excerpt length and relative position in all six channels for a matched prompt/rep.
clips={m:{'start':[],'random':[]} for m in models};clip_meta=[]
for i,k in enumerate(keys):
 n=min(60,*[len(rows[m][i][c].split()) for m in models for c in ['reasoning','response']]);u=rng.random()
 clip_meta.append({'key':k,'words':n,'relative_position':u})
 for m in models:
  for c in ['reasoning','response']:
   ws=rows[m][i][c].split();off=int(u*(len(ws)-n))
   clips[m]['start'].append(' '.join(ws[:n]));clips[m]['random'].append(' '.join(ws[off:off+n]))
def tokens(t):return re.findall(r'(?u)\b\w+\b',t.lower())
def tfidf_fit(train,test):
 df=collections.Counter(w for s in train for w in set(tokens(s)))
 vocab={w:j for j,w in enumerate(sorted(w for w,n in df.items() if n>=2))}
 idf=np.array([math.log((1+len(train))/(1+df[w]))+1 for w in vocab])
 def transform(texts):
  rr=[];cc=[];vv=[]
  for i,s in enumerate(texts):
   counts=collections.Counter(w for w in tokens(s) if w in vocab)
   vals=[(vocab[w],(1+math.log(n))*idf[vocab[w]]) for w,n in counts.items()];norm=math.sqrt(sum(v*v for _,v in vals)) or 1
   for j,v in vals:rr.append(i);cc.append(j);vv.append(v/norm)
  return csr_matrix((vv,(rr,cc)),shape=(len(texts),len(vocab)))
 return transform(train),transform(test),vocab
# Sum logistic loss + .5 L2(w), i.e. fixed C=1. Unpenalized intercept.
def fit(X,y):
 def objective(v):
  z=X@v[:-1]+v[-1];e=expit(z)-y
  return float(np.logaddexp(0,z).sum()-y@z+.5*(v[:-1]@v[:-1])),np.r_[X.T@e+v[:-1],e.sum()]
 res=minimize(objective,np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10})
 assert res.success,res.message
 return res.x
# Analytical gradient checked against finite differences on tiny fixture.
from scipy.optimize import check_grad
X0=csr_matrix([[1.,0.],[0.,1.],[1.,1.]]);y0=np.array([0,1,1]);v0=np.array([.2,-.3,.1])
def obj(v):z=X0@v[:-1]+v[-1];return np.logaddexp(0,z).sum()-y0@z+.5*(v[:-1]@v[:-1])
def grad(v):e=expit(X0@v[:-1]+v[-1])-y0;return np.r_[X0.T@e+v[:-1],e.sum()]
grad_err=float(check_grad(obj,grad,v0));assert grad_err<1e-6
Y=np.tile([0.,1.],len(keys));groups=np.repeat([k[0] for k in keys],2)
def cv(texts,y=Y,length=False):
 scores=np.zeros(len(texts))
 for f in range(5):
  tr=np.array([folds[s]!=f for s in groups]);te=~tr
  train=[t for t,b in zip(texts,tr) if b];test=[t for t,b in zip(texts,te) if b]
  if length:
   a=np.log1p([len(t.split()) for t in train]);b=np.log1p([len(t.split()) for t in test]);sd=a.std() or 1
   X=csr_matrix(((a-a.mean())/sd)[:,None]);Z=csr_matrix(((b-a.mean())/sd)[:,None])
  else:X,Z,_=tfidf_fit(train,test)
  w=fit(X,y[tr]);scores[te]=Z@w[:-1]+w[-1]
 return scores
bootrng=np.random.default_rng(SEED);resamples=bootrng.integers(0,len(scenarios),size=(10000,len(scenarios)))
def metrics(scores):
 pair=np.sign(scores[1::2]-scores[::2])/2+.5
 by=np.array([np.mean([v for v,k in zip(pair,keys) if k[0]==s]) for s in scenarios])
 ci=np.quantile(by[resamples].mean(axis=1),[.025,.975])
 return {'individual_accuracy':float(np.mean((scores>0)==Y)),'paired_accuracy':float(pair.mean()),'paired_correct':int((pair==1).sum()),'paired_ties':int((pair==.5).sum()),'pairs':len(pair),'scenario_bootstrap95':ci.tolist()}
result={'seed':SEED,'folds':folds,'gradient_check_error':grad_err,'methods':'TF-IDF lowercased word unigrams, train-only vocabulary min document frequency2, sublinear TF and L2-normalized vectors; logistic regression C=1 unpenalized intercept. Five folds grouped by scenario; all5reps together. Scores larger=final. Paired accuracy chooses which of paired texts is final. Ties counthalf. Excerpts length=min(60,shortest ofsix matched texts); identical relative random position. No tuning. CI resamples35scenarios10000times.','source_sha256':{m:hashlib.sha256((P/'iei'/m/'baseline.jsonl').read_bytes()).hexdigest() for m in models},'models':{}}
predictions=[]
for m in models:
 texts=[r[c] for r in rows[m] for c in ['reasoning','response']]
 cw=[len(r['reasoning'].split()) for r in rows[m]];fw=[len(r['response'].split()) for r in rows[m]]
 stats={'median_cot_words':statistics.median(cw),'median_final_words':statistics.median(fw),'cot_longer':sum(a>b for a,b in zip(cw,fw)),'exact_equal':sum(r['reasoning']==r['response'] for r in rows[m]),'scores':{}}
 for view,tt,length in [('length_only',texts,True),('full',texts,False),('start_excerpt',clips[m]['start'],False),('random_excerpt',clips[m]['random'],False)]:
  ss=cv(tt,length=length);stats['scores'][view]=metrics(ss)
  for i,k in enumerate(keys):predictions.append({'model':m,'view':view,'scenario':k[0],'rep':k[1],'cot_score':float(ss[2*i]),'final_score':float(ss[2*i+1])})
 # Fit full data only for descriptive feature inspection, not accuracy.
 X,_,vocab=tfidf_fit(texts,[]);w=fit(X,Y);terms=list(vocab);order=np.argsort(w[:-1])
 stats['fulltext_features']={'cot':[(terms[j],float(w[j])) for j in order[:15]],'final':[(terms[j],float(w[j])) for j in order[-15:][::-1]]}
 result['models'][m]=stats
 print(m,json.dumps(stats['scores']),flush=True)
# Null-label control for Steve full texts: independent swaps within prompt/rep.
texts=[r[c] for r in rows['mt1b-s150'] for c in ['reasoning','response']]
null=[]
for rep in range(10):
 yy=Y.copy()
 for i in range(len(keys)):
  if rng.random()<.5:yy[2*i:2*i+2]=yy[2*i:2*i+2][::-1]
 ss=cv(texts,y=yy);pred=(ss[1::2]>ss[::2]);truth=yy[1::2]>yy[::2]
 null.append(float(np.mean(pred==truth)))
result['steve_null_paired_accuracies']=null
# Conversational battery is descriptive only and separate from advice classifier.
result['battery']={}
for m in ['base','mt1b-s150']:
 rr=list(map(json.loads,open(P/'battery'/(m+'.jsonl'))))
 result['battery'][m]={'n':len(rr),'exact_equal':sum(r['reasoning']==r['answer'] for r in rr),'equal_keys':[(r['qi'],r['sample']) for r in rr if r['reasoning']==r['answer']]}
(O/'results.json').write_text(json.dumps(result,indent=2));(O/'predictions.json').write_text(json.dumps(predictions,indent=2));(O/'excerpt_sampling.json').write_text(json.dumps(clip_meta,indent=2))
print(json.dumps(result,indent=2))
