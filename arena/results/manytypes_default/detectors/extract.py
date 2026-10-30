import json
from pathlib import Path
from inspect_ai.log import read_eval_log
cells={"base":"gptoss64-covert2-easy48","grpo3":"grpo3s100-64-covert2-easy48","s100":"mt1b-s100-64-covert2-easy48","s150":"mt1b-s150-64-covert2-easy48"}
out={}
for c,d in cells.items():
    eps={}
    for p in sorted(Path("logs",d).glob("*.eval")):
        for s in read_eval_log(p).samples:
            meta=next(iter(s.scores.values())).metadata
            turns=[]
            for m in s.messages:
                if m.role!="assistant": continue
                blocks=[]
                if isinstance(m.content,str):
                    blocks=[("text",m.content)]
                else:
                    for b in m.content:
                        if b.type=="reasoning": blocks.append(("reasoning",b.reasoning or ""))
                        elif b.type=="text": blocks.append(("text",b.text or ""))
                        else: blocks.append((b.type,""))
                tcs=[(tc.function,json.dumps(tc.arguments)) for tc in m.tool_calls or []]
                turns.append({"blocks":blocks,"tcs":tcs})
            eps[str(s.id)]={"main":meta["main_pass"],"leak":meta["side_leaked"],"turns":turns}
    out[c]=eps
    print(c,len(eps))
json.dump(out,open("/tmp/claude-1000/-home-ek-wgmn-mats12-sprint/3ee5d75a-f30e-59de-9e51-6bfa14f17dc2/scratchpad/turns.json","w"))
