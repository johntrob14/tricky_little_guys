"""Pool tlg-inksm transfer_eval summaries (strict, meta excluded = 'nometa', F0, user turn) by type family."""
import json,os
T='/home/john/code/tlg-inksm/cot-control/results/transfer/'
FAM={'Astra 9':['word_suppression','multiple_word_suppression','repeat_sentences','end_of_sentence','meow_between_words','lowercase_thinking','uppercase_thinking','alternating_case','ignore_question'],
     'other surface 9':['decoy_topic','no_question_restate','no_option_text','third_person','end_apple','no_letter_e','questions_only','max_words','no_calculation'],
     'reasoning-process 6':['no_commitment','answer_hidden','misdirect','elimination_only','no_backtracking','first_principles_only'],
     'length 3':['target_words','target_tokens','think_tokens'],
     'language 5':['think_language','respond_french','respond_french_front','respond_think_french','answer_language']}
def load(*runs):
  s={}
  for r in runs:
    p=T+r+'/summary.json'
    if os.path.exists(p):s.update({k:v for k,v in json.load(open(p)).items() if k.endswith('|F0|user')})
  return s
M={'base':load('gptoss-base','gptoss-base-newtypes'),'grpo3 s100 (CoT, 6 Astra types)':load('gptoss-grpo3s100'),'manytypes1b s150 (CoT, 12 types)':load('mt1b-s150'),
   'nocotrl1 s80 (output, CoT off)':load('nocotrl1-s80'),'B additive s90 (output, CoT off)':load('nocotrlB-add-s90')}
print(f"{'model':34}"+''.join(f'{f:>22}' for f in FAM))
for m,s in M.items():
  row=''
  for f,ts in FAM.items():
    c=[s[t+'|F0|user'] for t in ts if t+'|F0|user' in s]
    row+=f"{(str(sum(x['nometa'] for x in c))+'/'+str(sum(x['n'] for x in c))+' '+format(100*sum(x['nometa'] for x in c)/max(1,sum(x['n'] for x in c)),'.0f')+'%') if c else '-':>22}"
  print(f'{m:34}'+row)
print('\nper type, nometa/n (base | nocotrl1 s80 | B s90):')
for f,ts in FAM.items():
  for t in ts:
    k=t+'|F0|user';g=lambda m:f"{M[m][k]['nometa']}/{M[m][k]['n']}" if k in M[m] else '-'
    print(f'  {f[:9]:10}{t:28}{g("base"):>9}{g("nocotrl1 s80 (output, CoT off)"):>9}{g("B additive s90 (output, CoT off)"):>9}   median tokens {M["base"].get(k,{}).get("tokens","-")} | {M["nocotrl1 s80 (output, CoT off)"].get(k,{}).get("tokens","-")} | {M["B additive s90 (output, CoT off)"].get(k,{}).get("tokens","-")}')
