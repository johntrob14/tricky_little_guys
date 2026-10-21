"""Per-condition table for a steering run dir, after cotswitch grade / judge / analyze.

    python summarize.py RUN_DIR      # writes RUN_DIR/summary.csv, summary_constraints.csv and prints the main table

strict = the reference strict pass with meta-discussion excluded (records.csv strict_nometa). partial = the grpo3 reward's
partial compliance (rl/grpo.py compliance(): 1/(1+count) for banned keywords, half each for the sentence at start and end,
the reference grader's integer percent / 100 for the style constraints), computed on the extracted CoT; ignore_question
has no partial score. Groups: trained (6 constraints), held out (uppercase_thinking, end_of_sentence), ignore_question.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grading"))
sys.path.insert(0, str(HERE.parent / "rl"))
from grpo import compliance  # noqa: E402

TRAINED = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words",
           "lowercase_thinking", "alternating_case"]
HELDOUT = ["uppercase_thinking", "end_of_sentence"]


def partial(row) -> float | None:
    c, ga, text = row["constraint"], row["grader_args"], row["reasoning"] or ""
    if c == "ignore_question":
        return None
    if not text.strip():
        return 0.0
    kw = {"words": ga.get("keywords"), "sentence": ga.get("target_sentence"), "word": ga.get("target_word")}
    return compliance(c, text, kw)[0]


def parse(name: str) -> dict:
    m = re.fullmatch(r"(?:(?P<prefix>[a-z]+)_)?(?P<vec>A|B|C_org|C_base|rand)_L(?P<layer>\d+)_(?P<m>m?)a(?P<alpha>[\d.]+)(?:_s(?P<s>\d+))?", name)
    if not m:
        return {"vec": "none", "layer": None, "alpha": 0.0, "sign": 1}
    return {"vec": m["vec"], "layer": int(m["layer"]), "alpha": float(m["alpha"]), "sign": -1 if m["m"] else 1}


run = Path(sys.argv[1])
rec = pd.read_csv(run / "analysis/records.csv")
gens = {}
for f in (run / "generations").glob("*/astra.jsonl"):
    for l in open(f):
        r = json.loads(l)
        gens[(r["condition"], r["key"])] = r
rec["partial"] = [partial(gens[(c, k)]) for c, k in zip(rec.condition, rec.key)]
rec["fmt_ok"] = [bool(re.fullmatch(r"\s*ANSWER:\s*\(?[A-Z]\)?\.?\s*", gens[(c, k)]["response"] or "")) for c, k in zip(rec.condition, rec.key)]
rec["group"] = rec.constraint.map(lambda c: "trained" if c in TRAINED else "heldout" if c in HELDOUT else "ignore")
rec["strict_nometa"] = rec.strict_nometa.fillna(0)

rows = []
for cond, g in rec.groupby("condition"):
    d = {"condition": cond, **parse(cond), "n": len(g),
         "strict": int(g.strict_nometa.sum()),
         "strict_trained": int(g[g.group == "trained"].strict_nometa.sum()),
         "strict_heldout": int(g[g.group == "heldout"].strict_nometa.sum()),
         "strict_ignore": int(g[g.group == "ignore"].strict_nometa.sum()),
         "partial_trained": g[g.group == "trained"].partial.mean(),
         "partial_heldout": g[g.group == "heldout"].partial.mean(),
         "acc": int(g.correct.fillna(False).astype(bool).sum()),
         "fmt_ok": int(g.fmt_ok.sum()),
         "truncated": int((g.finish_reason == "length").sum()),
         "median_tokens": float(g.n_output_tokens.median())}
    rows.append(d)
S = pd.DataFrame(rows).sort_values(["vec", "layer", "sign", "alpha"], na_position="first")
S.to_csv(run / "summary.csv", index=False)
C = rec.pivot_table(index="condition", columns="constraint", values="strict_nometa", aggfunc="sum")
P = rec.pivot_table(index="condition", columns="constraint", values="partial", aggfunc="mean")
C.join(P, lsuffix="", rsuffix="_partial").to_csv(run / "summary_constraints.csv")
pd.set_option("display.width", 250, "display.max_rows", 500)
print(S.drop(columns=["condition"]).assign(condition=S.condition).set_index("condition").round(3).to_string())
