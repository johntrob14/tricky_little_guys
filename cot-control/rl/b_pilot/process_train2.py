"""Revised process-rule run (v2) from additive s90, after v1 (process_train.py) was stopped at s108 by the Astra monitor.
Changes from v1, everything else identical:
  - explanation-length gate for the task term 40 -> 15 words (v1 learned to lengthen explanations instead of the rules);
  - elimination partial = 0.3 x share of wrong options ruled out + 0.7 / (1 + affirming sentences) (v1: 0.6 / 0.4);
  - 3 judge votes per process output: partial = mean over votes, strict = at least 2 of 3;
  - 24 steps (s90 -> s114), run directory results/nocotrl-b-pilot/process2.
    python process_train2.py train
"""
import asyncio, statistics, sys
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import process_train as T
from process_pilot2 import score as score1

T.OUT=T.P/'process2';T.END=114;T.MINW=15;T.BUDGET=4.5;T.__doc__=__doc__

def score(process,obj,n_options):
    if process!='elimination_only':return score1(process,obj,n_options)
    st=list((obj.get('options') or {}).values());ruled=st.count('ruled_out');argued=st.count('argued_for');aff=int(obj.get('affirming_sentences') or 0)
    return 0.3*min(1,ruled/max(1,n_options-1))+0.7/(1+aff),argued==0 and ruled>=n_options-1

class Judge3(T.Judge):
    async def __call__(self,process,question,body,n_options):
        votes=await asyncio.gather(*(super(Judge3,self).__call__(process,question,body,n_options) for _ in range(3)))
        objs=[v[2] for v in votes if v[2] is not None]
        if not objs:return 0.0,False,None
        sc=[score(process,o,n_options) for o in objs]
        return statistics.mean(p for p,_ in sc),sum(s for _,s in sc)>=2,dict(votes=objs)

_main=T.main
async def main(mode):
    if mode=='train':
        T.OUT.mkdir(parents=True,exist_ok=True)
        orig=T.Judge;T.Judge=Judge3
        try:await _main('train')
        finally:T.Judge=orig
    else:await _main(mode)

if __name__=='__main__':asyncio.run(main(sys.argv[1]))
