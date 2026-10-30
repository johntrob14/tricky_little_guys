import pickle, re, json
from pathlib import Path
from collections import Counter
from inspect_ai.log import read_eval_log
CELLS = {"base":"gptoss64-covert2-easy48","grpo3":"grpo3s100-64-covert2-easy48","s100":"mt1b-s100-64-covert2-easy48","s150":"mt1b-s150-64-covert2-easy48"}
out = {}
for c, d in CELLS.items():
    for p in sorted(Path("logs", d).glob("*.eval")):
        for s in read_eval_log(p).samples:
            meta = next(iter(s.scores.values())).metadata
            turns = []
            for m in s.messages:
                if m.role == "assistant":
                    r = "".join(x.reasoning for x in m.content if x.type == "reasoning") if isinstance(m.content, list) else ""
                    txt = m.content if isinstance(m.content, str) else "".join(x.text for x in m.content if x.type == "text")
                    turns.append(dict(reasoning=r, text=txt, calls=[(tc.function, json.dumps(tc.arguments)) for tc in (m.tool_calls or [])], outs=[]))
                elif m.role == "tool" and turns:
                    turns[-1]["outs"].append(m.content if isinstance(m.content, str) else "".join(getattr(x, "text", "") for x in m.content))
            out[(c, str(s.id))] = dict(main=meta["main_pass"], leak=meta["side_leaked"], turns=turns)
pickle.dump(out, open("/tmp/claude-1000/-home-ek-wgmn-mats12-sprint/3ee5d75a-f30e-59de-9e51-6bfa14f17dc2/scratchpad/pg/eps.pkl", "wb"))
print(Counter(k[0] for k in out), Counter((k[0], v["leak"]) for k, v in out.items()))
