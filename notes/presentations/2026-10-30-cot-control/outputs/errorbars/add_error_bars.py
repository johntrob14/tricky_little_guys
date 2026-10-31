"""Add intervals to existing slide images; preserve all original pixels otherwise."""
from pathlib import Path
import ast, base64, collections, gzip, hashlib, json, re, sys
import numpy as np
from PIL import Image
import cairosvg

O=Path(__file__).resolve().parent
PREV=Path('/home/john/Documents/Codex/2026-10-30/please-make-a-bar-plot-of')
Q=Path('/home/john/Documents/Codex/2026-10-29/pull-up-the-chat-we-had/work/qwen17-repo/cot-control')
C=Path('/home/john/code/results/nemo-cascade-astra/cascade-astra/thinking')
sys.path.insert(0,str(Q/'public_ifrl/qwen17/grading'))
from cotswitch import graders
SEED=20261031; B=20000
rng=np.random.default_rng(SEED); hashes={}; evidence=[]
def read(p):
 p=Path(p);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
  return json.load(f) if p.suffix=='.json' else [json.loads(l) for l in f if l.strip()]
def question(k): return '|'.join(k.split('|')[:3])
def interval(rows,macro=False):
 keys=sorted({r['question'] for r in rows}); types=sorted({r['type'] for r in rows}) if macro else ['all']
 wins=np.zeros((len(keys),len(types))); ns=wins.copy();ki={k:i for i,k in enumerate(keys)}
 for r in rows:
  i=ki[r['question']];j=types.index(r['type']) if macro else 0
  wins[i,j]+=r['ok'];ns[i,j]+=1
 weights=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=B)
 den=weights@ns; valid=(den>0).all(axis=1)
 boot=((weights[valid]@wins)/den[valid]).mean(axis=1)*100
 value=float((wins.sum(axis=0)/ns.sum(axis=0)).mean()*100)
 lo,hi=np.percentile(boot,[2.5,97.5])
 if value==0:lo=0.;hi=100*3.841458820694124/(len(keys)+3.841458820694124)
 return dict(value=value,low=float(lo),high=float(hi),questions=len(keys),samples=len(rows),bootstrap_draws=int(valid.sum()))
def score(c,text,args):
 args=ast.literal_eval(args) if isinstance(args,str) else dict(args)
 args={k:v for k,v in args.items() if k not in ('prompt','user_prompt')}
 for k in ('synonyms','keywords'):
  if isinstance(args.get(k),str):args[k]=ast.literal_eval(args[k])
 return int(graders._grade_default(c,text,args)['strict']) if text.strip() else 0

base={r['model']:r for r in read(PREV/'outputs/control_provenance.json')['results']}
org={r['run']:r for r in read(PREV/'outputs/organism_control_provenance.json')['results']}
items=read('/home/john/code/tlg-inksm/cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')
items={r['key'].rsplit('|',1)[0]:r for r in items}
test={r['key'].rsplit('|',1)[0] for r in read('/home/john/code/tlg-inksm/cot-control/results/outctl/elicit/runs/gptoss/base_test/graded.json')}
pairs=[('Qwen3.6-35B','qwen1_s80'),('Qwen3.5-9B','q9b1_s80'),('gpt-oss-120B','nocotrl1'),('Nemotron-3 Super','nemo1_s80'),('Inkling-Small','nocotrlink2')]
out=[]
for group,(model,run) in enumerate(pairs):
 for j,condition in enumerate(['Base','No-CoT RL']):
  for k,arm in enumerate(['cot','output']):
   rec=base[model]['selected_'+arm] if j==0 else org[run]['arms']['elicited_'+arm]
   raw=read(rec['source']);assert hashes[rec['source']]==rec['sha256']
   rows=[]
   for r in raw:
    key=r['key'].rsplit('|',1)[0]
    if r['constraint']=='ignore_question' or key not in test:continue
    if arm=='output' and (r.get('cot') or '').strip():continue
    text=(r.get('body') if arm=='output' else r.get('out')) or ''
    strict=score(r['constraint'],text,items[key]['grader_args'])
    bare=arm=='cot' and not (r.get('resp') or '').strip() and bool(re.match(r'\s*\**ANSWER',text)) and len(text.split())<8
    ok=int(strict and (bool(r.get('attempt')) if arm=='output' else not bare))
    rows.append(dict(question=question(r['key']),type=r['constraint'],ok=ok))
   ci=interval(rows,True);assert abs(ci['value']-rec['score'])<1e-9
   ci.update(model=model,condition=condition,arm=arm,x=group*2.65+(-.52 if j==0 else .52)+(k-.5)*.35)
   out.append(ci);evidence.append(dict(series=ci,rows=rows,source=rec['source']))

qw=[];verified=json.loads((O.parent/'verified_data.json').read_text())
for j,run in enumerate(['initial','kl0p002-s364','kl0p005-s364']):
 raw=read(Q/f'results/public-ifrl/qwen17-20261030/{run}/generations.jsonl.gz')
 for i,variant in enumerate(['base','shot4']):
  rows=[]
  for r in raw:
   if r['variant']!=variant or r['constraint']=='ignore_question':continue
   strict=score(r['constraint'],r['reasoning'],r['grader_args'])
   valid=bool(r['reasoning'].strip() and r['think_closed'] and r['response'].strip() and r['finish_reason']!='length')
   rows.append(dict(question=question(r['key']),type=r['constraint'],ok=int(valid and strict)))
  ci=interval(rows);assert sum(r['ok'] for r in rows)==verified['qwen'][run][variant]['completed']
  ci.update(run=run,variant=variant,x=i+(j-1)*.2)
  qw.append(ci);evidence.append(dict(series=ci,rows=rows))

ne=[]
for i,stage in enumerate(['rlhf','ifrl']):
 raw=read(C/f'generations/nemotron-cascade-8b-{stage}/astra.jsonl');rows=[]
 for r in raw:
  if r['constraint']=='ignore_question':continue
  strict=score(r['constraint'],r['reasoning'],r['grader_args'])
  valid=bool(r['reasoning'].strip() and r['think_closed'] and r['response'].strip() and r['finish_reason']!='length')
  rows.append(dict(question=question(r['key']),type=r['constraint'],ok=int(valid and strict)))
 ci=interval(rows);assert sum(r['ok'] for r in rows)==verified['cascade']['thinking'][stage]['completed']
 ci.update(stage=stage,x=i);ne.append(ci);evidence.append(dict(series=ci,rows=rows))

# Original slide raster plus vector whiskers: every non-error-bar pixel is preserved.
def overlay(name,cis,xmap,ymap,scale,labels=False):
 p=O/f'{name}_original.png';im=Image.open(p);w,h=im.size
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<image width="{w}" height="{h}" xlink:href="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"/>']
 parts.append('<g stroke="#202a36" stroke-width="2" fill="none">')
 for c in cis:
  x=xmap(c['x']);lo=ymap(c['low']);hi=ymap(c['high']);cap=5*scale
  stem=f'M{x},{lo} V{hi}'
  if labels:
   label_y=ymap(c['value']+1.4)
   # Break only the stem around the unchanged numeric label; keep both caps.
   gap_top,gap_bottom=label_y-20,label_y+2
   stem=''
   if lo>gap_bottom:stem+=f'M{x},{lo} V{max(hi,gap_bottom)} '
   if hi<gap_top:stem+=f'M{x},{min(lo,gap_top)} V{hi}'
  parts.append(f'<path d="{stem} M{x-cap},{lo} H{x+cap} M{x-cap},{hi} H{x+cap}"/>')
 parts.append('</g></svg>');svg=''.join(parts)
 (O/f'{name}_with_error_bars.svg').write_text(svg)
 cairosvg.svg2png(bytestring=svg.encode(),write_to=str(O/f'{name}_with_error_bars.png'))
 # Verify differences are restricted to the whisker overlay's support.
 bare=svg[svg.index('<g stroke='):].replace('</svg>','')
 maskpart=''
 mask=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">{maskpart}{bare}</svg>'
 import io
 alpha=np.asarray(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=mask.encode()))).convert('RGBA'))[:,:,3]
 a=np.asarray(im.convert('RGBA'));b=np.asarray(Image.open(O/f'{name}_with_error_bars.png').convert('RGBA'))
 # RGB at zero alpha has no visible meaning.
 changed=np.any(a!=b,axis=2)&((a[:,:,3]>0)|(b[:,:,3]>0))
 assert not np.any(changed & (alpha==0)),name

overlay('output',out,lambda x:144+(x+1.16)/12.92*1747.2,lambda y:568.8-y*4.68,1,True)
# Matplotlib's fixed bar autoscaling, read against the original SVG coordinates.
overlay('qwen',qw,lambda x:145.92+(x+.3668)/1.7336*1374.72,lambda y:311.04-y*4.7616,1)

# Nemotron is the same original Matplotlib plot, with the approved axis extension.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':19,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#B3BAC2','axes.labelcolor':'#202A36','xtick.color':'#202A36','ytick.color':'#202A36','text.color':'#202A36'})
fig,ax=plt.subplots(figsize=(6,8));fig.subplots_adjust(left=.23,right=.97,bottom=.12,top=.97)
ax.bar([0,1],[c['value'] for c in ne],width=.6,color=['#3467A4','#008F85'],zorder=3)
ax.plot([-.3,.3],[0,0],color='#3467A4',lw=4,zorder=4,clip_on=False)
ax.errorbar([0,1],[c['value'] for c in ne],yerr=[[c['value']-c['low'] for c in ne],[c['high']-c['value'] for c in ne]],fmt='none',ecolor='#202A36',elinewidth=1.5,capsize=5,capthick=1.5,zorder=5)
ax.set_xticks([0,1],['Before IF-RL','After IF-RL'],fontsize=17)
ax.set_ylim(0,20);ax.set_yticks([0,5,10,15,20],['0%','5%','10%','15%','20%'])
ax.set_ylabel('Strict CoT compliance',labelpad=14,fontsize=18);ax.tick_params(axis='both',length=0,pad=10);ax.grid(axis='y',color='#E5E8EC',zorder=0)
fig.savefig(O/'nemotron_with_error_bars.svg',transparent=True);fig.savefig(O/'nemotron_with_error_bars.png',dpi=180,facecolor='white');plt.close(fig)
result=dict(method='95% question-cluster percentile bootstrap; 20,000 draws; repeated samples grouped by source question; output-control chart retains equal weighting of eight constraint families. Zero baseline uses conservative Wilson upper bound for any success within a question cluster.',seed=SEED,scope='Evaluation-question uncertainty conditional on the selected prompt and checkpoint; excludes training-run variation and prompt-selection uncertainty.',output=out,qwen=qw,nemotron=ne,hashes=hashes)
(O/'intervals.json').write_text(json.dumps(result,indent=2)+'\n');(O/'scored_observations.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['output','qwen','nemotron']},indent=2))
