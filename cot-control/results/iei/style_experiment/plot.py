"""Scientific figure from validated saved ratings, with no model calls."""
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT
data=json.load(open(OUT/'results.json'))
groups={(r['view'],r['model'],r['channel']):r for r in data['groups']}
models=['base','mt1b-s150','mt2e-s170']
names=['Base gpt-oss-120b','manytypes1b s150','manytypes2e s170']
colors=['#687482','#007f78','#b76a16']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'font.family':'DejaVu Sans','savefig.facecolor':'white'})
fig,axs=plt.subplots(1,2,figsize=(12.4,5.1),sharey=True)
for ax,view,title in zip(axs,['full','excerpt'],['Full texts','Matched-length excerpts (43–60 words)']):
    for j,(channel,offset,label) in enumerate([('reasoning',-.17,'CoT'),('response',.17,'Final response')]):
        rs=[groups[view,m,channel] for m in models]
        ys=np.array([r['F_mean']*100 for r in rs])
        lo=np.array([r['F_ci95'][0]*100 for r in rs]);hi=np.array([r['F_ci95'][1]*100 for r in rs])
        ax.bar(np.arange(3)+offset,ys,width=.29,color=colors if j==0 else 'white',
               edgecolor=colors,linewidth=1.8,label=label)
        ax.errorbar(np.arange(3)+offset,ys,yerr=[ys-lo,hi-ys],fmt='none',
                    ecolor='#263238',capsize=3,linewidth=1.2)
        for x,y,h in zip(np.arange(3)+offset,ys,hi):ax.text(x,max(y,h)+2,f'{y:.1f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(3),['Base','s150','2e'])
    ax.set_title(title,pad=14)
    ax.set_ylim(0,112);ax.set_yticks([0,20,40,60,80,100])
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Words labeled finished explanation/advice (%)')
fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,.94))
fig.suptitle('Does ordinary CoT read like a finished response?',fontsize=16,y=1.01)
fig.text(.5,.005,'35 matched advice scenarios · one saved rollout/model · fresh blinded labels\n'
         'Bars: mean per-text word fraction; whiskers: 95% question-bootstrap intervals. Ambiguous words remain in the denominator.',
         ha='center',va='bottom',fontsize=9,color='#455a64')
fig.tight_layout(rect=(0,.08,1,.88))
fig.savefig(OUT/'style-comparison.png',dpi=200,bbox_inches='tight')
fig.savefig(OUT/'style-comparison.pdf',bbox_inches='tight')

fig,ax=plt.subplots(figsize=(10,4.8))
palette=['#6a7885','#acc3ce','#e4b86f','#007f78']
labels=['Private notes','Mostly notes, brief draft','Substantial mixture','Essentially a response']
ys=np.arange(6);left=np.zeros(6)
rs=[groups['full',m,c] for m in models for c in ['reasoning','response']]
for level in range(4):
    vals=np.array([r['whole_counts'].get(str(level),0)/35*100 for r in rs])
    ax.barh(ys,vals,left=left,color=palette[level],label=labels[level])
    for y,w,l in zip(ys,vals,left):
        if w>5:ax.text(l+w/2,y,f'{round(w*35/100)}/35',ha='center',va='center',color='white' if level in [0,3] else '#263238',fontsize=10)
    left+=vals
ax.set_yticks(ys,[f'{name} · {channel}' for name in names for channel in ['CoT','final']])
ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Share of full texts (%)')
ax.set_title('Whole-text presentation ratings, with model and channel hidden')
ax.legend(frameon=False,ncol=2,bbox_to_anchor=(.5,-.22),loc='upper center')
fig.tight_layout();fig.savefig(OUT/'whole-text-ratings.png',dpi=200,bbox_inches='tight')
print('Saved two PNG figures and a PDF.')
