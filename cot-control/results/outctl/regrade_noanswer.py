"""Regrade explained-output rows with the closing answer line removed (as rgrpo1 training does), to see how much the trailing
'Answer: X' line explains output-arm failures. Deterministic graders; attempt verdicts reused from graded.json. Run from cot-control/grading."""
import ast, json, re, sys, collections
sys.path.insert(0, ".")
from cotswitch import graders
O = "../results/outctl"
ANS_LINE = re.compile(r"(?im)^[^\n]*\banswer\b[^\n]*?\(?\b([A-J])\b\)?[^A-Za-z\n]*$")
def grade(text, it):
    a = {k: v for k, v in (ast.literal_eval(it["grader_args"]) if isinstance(it["grader_args"], str) else it["grader_args"]).items() if k not in ("prompt", "user_prompt")}
    for k in ("synonyms", "keywords"):
        if isinstance(a.get(k), str): a[k] = ast.literal_eval(a[k])
    return bool(text) and graders._grade_default(it["constraint"], text, a)["strict"] == 1
astra = {x["key"]: x for x in map(json.loads, open(f"{O}/../tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl"))}
new = {x["key"]: x for x in json.load(open(f"{O}/new_items.json"))}
for m in sys.argv[1:]:
    tot = collections.Counter(); per = collections.defaultdict(lambda: [0, 0, 0])
    for src, items in (("more", astra), ("big", new)):
        for g in json.load(open(f"{O}/{src}/expl/{m}/graded.json")):
            c = g["constraint"]
            if c == "ignore_question": continue
            it = items[g["key"]]; body = g["body"]
            ms = list(ANS_LINE.finditer(body)); cut = (body[:ms[-1].start()] + body[ms[-1].end():]).strip() if ms else body
            orig = g["strict"] == 1 and bool(g["attempt"]); new_ = grade(cut, it) and bool(g["attempt"])
            per[c][0] += 1; per[c][1] += orig; per[c][2] += new_
            tot["n"] += 1; tot["orig"] += orig; tot["cut"] += new_; tot["has_ans_line"] += bool(ms)
    print(f"== {m}: output pass {100*tot['orig']/tot['n']:.1f}% as graded -> {100*tot['cut']/tot['n']:.1f}% with the answer line removed; answers with an answer line: {100*tot['has_ans_line']/tot['n']:.0f}%")
    for c, (n, o, k) in sorted(per.items()): print(f"   {c:26s} {100*o/n:5.1f} -> {100*k/n:5.1f}")
