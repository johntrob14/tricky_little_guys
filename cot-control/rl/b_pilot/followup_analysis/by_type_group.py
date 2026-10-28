"""Length decomposition (length_control.py functions) on Astra 108 split by who trained an output analog of the type:
both (B and nocotrl1: lowercase, multiple word suppression), nocotrl1 only (the other 6 rule types), neither (ignore_question)."""
import json,sys,os,collections,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec=importlib.util.spec_from_file_location('L',os.path.join(os.path.dirname(os.path.abspath(__file__)),'lenadj_B.py'))
src=open(spec.origin).read().split("for excl,lab in")[0];ns={'__file__':spec.origin};exec(src,ns)
ALL=['lowercase_thinking','multiple_word_suppression','uppercase_thinking','alternating_case','word_suppression','meow_between_words','end_of_sentence','repeat_sentences','ignore_question']
groups={'both trained (lower, multi-word suppr.)':['lowercase_thinking','multiple_word_suppression'],
        'nocotrl1 only (6 rule types)':['uppercase_thinking','alternating_case','word_suppression','meow_between_words','end_of_sentence','repeat_sentences'],
        'neither (ignore_question)':['ignore_question']}
P,F=ns['P'],ns['F']
for g,keep in groups.items():
  excl=tuple(t for t in ALL if t not in keep);B=ns['theirs']('base',excl)
  runs={'nocotrl1 s80x':ns['theirs']('nocotrl1-s80x',excl),'B add s90':ns['mine']('B add s90',[P+'additive/astra_full_s90.jsonl',F+'draws_add_s90/noise_add_s90.jsonl'],excl)}
  print(f'\n== {g}; strict pass-rate points vs base, 95% CI over items')
  for n,R in runs.items():
    s=ns['ns']['summary'](R,B);ci=ns['ns']['boot'](R,B,1000);f=lambda k:f"{100*s[k]:+5.1f} [{100*ci[k][0]:+5.1f},{100*ci[k][1]:+5.1f}]"
    base_rate=sum(r['s'] for r in B)/len(B)*100;rate=sum(r['s'] for r in R)/len(R)*100
    print(f"  {n:14} strict {rate:5.1f}% (base {base_rate:4.1f}%)  raw {f('raw')}  length {f('length')}  control-at-length {f('control')}")
