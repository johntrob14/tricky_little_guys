from pathlib import Path
import json,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
O=Path('outputs');d=json.loads((O/'other_models_data.json').read_text());a=d['astra'];regular=json.loads((O/'deepseek_regular_control.json').read_text())['results']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'text.color':'#202A36','axes.labelcolor':'#202A36','xtick.color':'#202A36','ytick.color':'#202A36'})
fig,axs=plt.subplots(1,3,figsize=(16,4),sharey=True,gridspec_kw={'width_ratios':[1,1,1]});fig.subplots_adjust(left=.08,right=.99,bottom=.35,top=.97,wspace=.20)
colors=['#929CA9','#5B98C8','#147D79']
def bar(ax,x,v,ci,cs,labels,width=.60):
 v=np.asarray(v);ci=np.asarray(ci);ax.bar(x,v,width=width,color=cs,zorder=3);ax.errorbar(x,v,yerr=np.maximum(0,np.array([v-ci[:,0],ci[:,1]-v])),fmt='none',ecolor='#202A36',capsize=4,elinewidth=1.7,zorder=4)
 for xx,c,ll in zip(x,ci,labels):ax.text(xx,c[1]+2,ll,ha='center',va='bottom',fontsize=16)
for ax,rs,labs,model in [(axs[0],['qwen38-run2','opsd1-step30-t1','opsd-grpo1-step30-t1'],['Base','OPSD','+ GRPO'],'Qwen3.8-27B'),(axs[1],['inksmtk-up-e06','inksft1-ep2-req-e06','inkrl1-s80-req-e06'],['Base','SFT','+ GRPO'],'Inkling-Small')]:
 bar(ax,np.arange(3),[a[r]['strict']/96*100 for r in rs],np.array([a[r]['strict_ci95'] for r in rs])*100,colors,[f"{a[r]['strict']/96*100:.1f}%" for r in rs]);ax.set_xticks(range(3),labs);ax.set_xlabel(model,labelpad=14,fontweight='bold')
ax=axs[2];rs=['cot/ds-base-tk/astra','cot/ds-obf2g/astra','cot/ds-rwsft1/astra']
bar(ax,np.arange(3),[regular[r]['rate']*100 for r in rs],np.array([regular[r]['ci95'] for r in rs])*100,colors,[f"{regular[r]['rate']*100:.1f}%" for r in rs])
ax.set_xticks(range(3),['Base','GRPO','+ SFT']);ax.set_xlabel('DeepSeek-V4.1-Flash\n(partial overlap)',fontweight='bold')
for ax in axs:
 ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100],['0%','25%','50%','75%','100%']);ax.grid(axis='y',color='#E5E8EC',zorder=0);ax.tick_params(length=0,pad=7)
axs[0].set_ylabel('Strict Compliance')
for ax in axs: ax.xaxis.set_label_coords(.5,-.30)
fig.savefig(O/'other_models_cot_control.svg',transparent=True);fig.savefig(O/'other_models_cot_control.png',dpi=180);plt.close(fig)
s=dict(title='CoT control is trainable across model families',bullets=[
 'We train reasoning directly: self-distillation → GRPO for Qwen, rewrite SFT → GRPO for Inkling, and GRPO → rewrite SFT for DeepSeek.',
 'These recipes show that CoT control can be taught across model families; strict compliance alone does not establish preserved reasoning quality.'
],notes='Qwen learns through on-policy self-distillation from a self-instructed teacher and then GRPO, while Inkling learns from rewritten CoTs and then GRPO. DeepSeek first receives GRPO and then rewrite SFT on seven reasoning styles, with plain examples mixed in. On the original 96 deterministic Astra items, strict compliance rises from 0 to 20 to 51 passes for Qwen and from 3 to 18 to 74 for Inkling. On the matched Astra slice, DeepSeek rises from 0.5% at base to 2.6% after GRPO and 29.7% after rewrite SFT, measured with two samples per item. These single-run comparisons demonstrate surface control, while DeepSeek’s partial training overlap and differences in evaluation wording and sampling limit direct comparisons across models.',caption='Eight deterministic Astra rules, 96 matched question–rule items per panel, 95 normalized-question clusters. Qwen and Inkling: one generation per item; thinking wording; Inkling effort 0.6; checkpoints OPSD s30 then 30 RL steps, and Inkling SFT epoch 2 then RL s80. DeepSeek: two generations per item (192 total), reasoning-stage wording, effort 75, temperature 1, top-p 1, 12k token cap; base → obf2g s25 → rwsft1 epoch 2, with strict counts 1/192 → 5/192 → 57/192. DeepSeek uses the original Astra slice of xfull, rather than the full expanded 480-item evaluation (which gives 3.3% → 32.0% for before/after SFT). All bars use deterministic strict compliance, allow meta discussion, and do not require correctness or a completed final answer. DeepSeek labels were independently regraded from raw text with no mismatches. Error bars: 95% question-cluster percentile bootstrap, 10,000 draws, seed 20261030; repeated generations are averaged per item before resampling questions. The all-zero Qwen base uses a conservative Wilson boundary bound. Intervals represent evaluation-question uncertainty, not variation across training seeds. Partial overlap: the prior audit found actual DeepSeek rewrite-SFT exposure to 13 of the full 108 Astra rows and eligible earlier RL-pool overlap with 36/108; those overlap counts include the semantic rule excluded from this figure, so they are not counts over the plotted 96. No overlap exclusion is applied. The comparison describes developed recipes and is not a clean held-out generalization estimate for DeepSeek.',figure='other_models_cot_control')
# Replace the active two-slide draft with the requested single-slide draft.
(O/'other_models_main_drafts.json').write_text(json.dumps([s],indent=2))
(O/'other_models_main_drafts.txt').write_text(s['title']+'\n\n'+'\n'.join('• '+b for b in s['bullets'])+'\n\nPresentation notes\n'+s['notes']+'\n\nCaption\n'+s['caption']+'\n')
(O/'other_models_main_drafts.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>'+s['title']+'</title><style>body{font:22px/1.45 Arial;color:#202a36;max-width:1160px;margin:30px auto;padding:25px}h1{font-size:36px}svg{width:100%;height:auto}li{margin:16px 0}p{font-size:19px}small{font-size:14px}</style></head><body><h1>'+s['title']+'</h1>'+(O/'other_models_cot_control.svg').read_text()+'<ul>'+''.join('<li>'+html.escape(b)+'</li>' for b in s['bullets'])+'</ul><p><b>Presentation notes</b><br>'+html.escape(s['notes'])+'</p><small>'+html.escape(s['caption'])+'</small></body></html>')
shutil.copy('work/build_combined_other_models.py',O/'build_combined_other_models.py')
print('Updated active main draft to one slide, with two bullets and five sentences of notes; backup deck unchanged.')
