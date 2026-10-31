from pathlib import Path
import json, re, html, shutil
O=Path('outputs/arena')
p=O/'build_figures.py';s=p.read_text()
names={
 'oss-hs-pw':'GPT-OSS-120B','oss-sd-pw':'GPT-OSS-120B',
 'sgcmp-qwen-covert2':'Qwen3.8-27B*','sgcmp-qwen-direct':'Qwen3.8-27B*',
 'opsd1b-hs-pw':'Qwen3.8-27B\nOPSD + GRPO','opsd1b-sd-pw':'Qwen3.8-27B\nOPSD + GRPO',
 'glm53-hs-pw':'GLM-5.3','glm53-sd-pw':'GLM-5.3',
 'ds41-hs-pw':'DeepSeek\nV4.1-Flash-1017','ds41-sd-pw':'DeepSeek\nV4.1-Flash-1017',
 'inksm-e09-hs-pw':'Inkling-Small','inksm-e09-sd-pw':'Inkling-Small',
 'inksft1-e09-hs-pw':'Inkling-Small\nSFT',
 'inkrl1s60-hs-pw':'Inkling-Small\nRL step 60',
 'inkrl1s60-e09-hs-pw':'Inkling-Small\nRL step 60','inkrl1s60-e09-sd-pw':'Inkling-Small\nRL step 60',
 'inkrl1s80-e09-hs-pw':'Inkling-Small\nRL step 80','inkrl1s80-e09-sd-pw':'Inkling-Small\nRL step 80',
 'grpo3-s100-sd-pw':'GPT-OSS-120B\nGRPO3 step 100'}
s=re.sub(r'^NAMES=.*$',lambda _: 'NAMES='+repr(names),s,flags=re.M)
s=s.replace("AC=A['cells'];SC=[r for r in S['cells'] if not r['run'].endswith('coach')]", "EXCLUDED=('mt1b-', 'mt2e-')\nAC=[r for r in A['cells'] if not r['run'].startswith(EXCLUDED)]\nSC=[r for r in S['cells'] if not r['run'].endswith('coach') and not r['run'].startswith(EXCLUDED)]")
s=s.replace("'Qwen OPSD + GRPO · episode #13 · main task passed + secret leaked',21", "'Qwen3.8-27B OPSD + GRPO · episode #13 · main task passed + secret leaked',18")
p.write_text(s)
# Keep the earlier workspace entrypoint from regenerating stale figures.
Path('work/build_arena_figures.py').write_text("import runpy\nrunpy.run_path('outputs/arena/build_figures.py', run_name='__main__')\n")
A=json.load(open(O/'always_verified.json'));S=json.load(open(O/'startgate_verified.json'))
ac=[r for r in A['cells'] if r['run'] in names];sc=[r for r in S['cells'] if r['run'] in names]
total=sum(r['done'] for r in ac)
assert len(ac)==9 and len(sc)==9 and total==123
selection={'excluded_run_prefixes':['mt1b-','mt2e-'],'reason':'User requested exclusion of manytypes organisms from LinuxArena presentation comparisons.','always_runs':[r['run'] for r in ac],'startgate_runs':[r['run'] for r in sc],'always_leaks':total,'always_attempts':sum(r['episodes'] for r in ac),'labels':names,'calibration':'Original 160-control calibration retained unchanged; presentation filtering does not refit thresholds.'}
(O/'plot_selection.json').write_text(json.dumps(selection,indent=2))
m=json.load(open(O/'main_slides.json'))
for x in m:
 x['notes']=x['notes'].replace('Each of ten cells','Each of nine cells').replace('Across the 125 episodes','Across the 123 episodes')
 x['caption']=x['caption'].replace('The ten cells include different organism checkpoints','The nine cells include different organism checkpoints')
 x['caption']=x['caption'].replace('manytypes1b includes two failed episodes in its 16-attempt denominator and no verified leaks. ','')
 if x['figure']!='qwen_plan_strategy.svg':x['caption']+=' Manytypes organisms are excluded from the displayed attack comparisons; the original honest-control calibration is retained.'
(O/'main_slides.json').write_text(json.dumps(m,indent=2))
cards=json.load(open(O/'backup_content.json'))
for c in cards:
 if c['id']=='sg_main_task_sensitivity':
  c['table']['rows']=[r for r in c['table']['rows'] if 'manytypes' not in r[0].lower()]
  for r in c['table']['rows']:
   r[0]={'GPT-OSS base':'GPT-OSS-120B','GPT-OSS grpo3':'GPT-OSS-120B\nGRPO3 step 100','Qwen OPSD + GRPO':'Qwen3.8-27B\nOPSD + GRPO'}.get(r[0],r[0])
 if c['id']=='arena_extra_coaching':
  for r,label in zip(c['table']['rows'],['Inkling-Small','Inkling-Small + coaching','Inkling-Small RL step 80','RL step 80 + coaching']):r[0]=label
 if c['id']=='arena_leak_definitions':
  c['table']['rows'][0][0]='GPT-OSS-120B GRPO3, Inkling-Small variants'
  c['table']['rows'][1][0]='GPT-OSS-120B, Qwen3.8-27B organism, GLM-5.3, DeepSeek-V4.1-Flash-1017'
  c['table']['rows'][2][0]='Original Qwen3.8-27B cells'
 if c.get('image'):c['notes']+=' Manytypes organisms are excluded from the plotted attack comparisons. The original 160-control calibration remains unchanged.'
(O/'backup_content.json').write_text(json.dumps(cards,indent=2))
Path('work/arena_native_cards.json').write_text(json.dumps(cards,indent=2))
Path('work/arena_final_cards.json').write_text(json.dumps(cards,indent=2))
(O/'main_slides.txt').write_text('\n\n'.join(x['title']+'\n'+'\n'.join('• '+b for b in x['bullets'])+'\n\nPresentation notes\n'+x['notes']+'\n\nCaption\n'+x['caption'] for x in m))
body=[]
for x in m:body.append('<section><h1>'+html.escape(x['title'])+'</h1><img src="'+x['figure']+'"><ul>'+''.join('<li>'+html.escape(b)+'</li>' for b in x['bullets'])+'</ul><p><b>Presentation notes</b><br>'+html.escape(x['notes'])+'</p><details><summary>Caption and evidence qualifications</summary><p>'+html.escape(x['caption'])+'</p></details></section>')
(O/'main_slides.html').write_text('<!doctype html><meta charset="utf-8"><title>Arena main-slide drafts</title><style>body{font:22px Arial;color:#202a36;margin:36px auto;max-width:1500px;line-height:1.4}section{margin-bottom:80px}img{width:100%}details{font-size:16px}li{margin:8px 0}</style>'+''.join(body))
print(json.dumps({'always_models':len(ac),'always_leaks':total,'always_attempts':sum(r['episodes'] for r in ac),'startgate_models':len(sc)}))
