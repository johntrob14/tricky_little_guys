"""Pre-fire turns scored >= 8 whose CoT names neither objective nor secret: what kind of tool call? counts only."""
import json, re, sys
from collections import Counter
sys.path.insert(0, ".")
from inspect_ai.log import read_eval_log
from pathlib import Path
import leak_check
PROBE = re.compile(r"\b(lsof|netstat|ss -|fuser|ps aux|ps -|pkill|kill|nohup|npm run dev|next dev|npm start|curl|wget|sleep)\b")
EDIT = re.compile(r'"command": "(create|str_replace|insert)"|sed -i|<<')
READ = re.compile(r'"command": "view"|\bcat\b|\bgrep\b|\bls\b|\bfind\b|\bhead\b')
TEST = re.compile(r"\b(jest|pytest|npm test|npm run test|npx)\b")
dump = json.load(open(sys.argv[1])); sub = sys.argv[2]
for label, eps in dump.items():
    want = {}
    for e in eps:
        if not any(e["fires"]): continue
        F = e["fires"].index(True)
        for i in range(F):
            if (e["scores"][i] or 0) >= 8 and e["terms"][i] is None:
                want.setdefault((e["run"], e["id"]), []).append(i)
    if not want: continue
    c = Counter()
    runs = {r for r, _ in want}
    for r in runs:
        d = next(Path("logs").glob(r))
        for f in sorted(d.glob("*.eval")):
            for s in read_eval_log(str(f)).samples or []:
                if (r, s.id) not in want: continue
                asst = [m for m in s.messages if m.role == "assistant"]
                for i in want[(r, s.id)]:
                    t = leak_check.calls_text(leak_check.msg_calls(asst[i]))
                    k = "no call" if not t.strip() else "edit" if EDIT.search(t) else "server/port probe" if PROBE.search(t) else "test" if TEST.search(t) else "read/search" if READ.search(t) else "other"
                    c[k] += 1
    print(label, sum(c.values()), dict(c.most_common()))
