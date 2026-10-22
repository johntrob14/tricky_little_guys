import json, pickle, re
from pathlib import Path
from inspect_ai.log import read_eval_log
CELLS = {"base": "gptoss64-covert2-easy48", "grpo3": "grpo3s100-64-covert2-easy48", "s100": "mt1b-s100-64-covert2-easy48", "s150": "mt1b-s150-64-covert2-easy48"}
def txt(m): return m.content if isinstance(m.content, str) else "".join(getattr(c, "text", "") or "" for c in m.content)
out = {}
for cell, d in CELLS.items():
    eps = []
    for p in sorted(Path("logs", d).glob("*.eval")):
        for s in read_eval_log(p).samples:
            meta = next(iter(s.scores.values())).metadata
            user = txt(next(m for m in s.messages if m.role == "user"))
            V = re.search(r"named `([A-Z0-9_]+)`", user).group(1)
            turns = []
            for m in s.messages:
                if m.role == "assistant":
                    r = "".join(c.reasoning for c in m.content if c.type == "reasoning") if isinstance(m.content, list) else ""
                    calls = [(tc.function, json.dumps(tc.arguments)) for tc in (m.tool_calls or [])]
                    turns.append({"r": r, "calls": calls, "msg": txt(m), "out": ""})
                elif m.role == "tool" and turns:
                    turns[-1]["out"] += txt(m)
            eps.append({"id": str(s.id), "task": str(s.id).split("#")[0], "V": V, "user": user, "main": meta["main_pass"], "leak": meta["side_leaked"], "turns": turns})
    out[cell] = eps
    print(cell, len(eps), sum(len(e["turns"]) for e in eps))
pickle.dump(out, open(f"{'/tmp/claude-1000/-home-ek-wgmn-mats12-sprint/3ee5d75a-f30e-59de-9e51-6bfa14f17dc2/scratchpad'}/ia.pkl", "wb"))
