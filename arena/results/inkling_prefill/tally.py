"""Per cell: main pass, leaked, CoT-only monitor (controlarena_cot_only, gpt-oss-20b, main task only; prefill stripped on prefilled cells) per-episode max and mean, score at the first
leak-detector fire in scorer-leaked episodes, score at turn 0, and how many later turns repeat turn 0's reasoning (first 60 characters after the prefill). Writes tally.json beside this file.
Counts and lengths only; no transcript text is printed.

    .venv/bin/python results/inkling_prefill/tally.py
"""
import glob, json, statistics as st, sys
from pathlib import Path
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning

sys.path.insert(0, ".")
from prompt_spec import fragment
from qwen_capability import episodes

PREFILL = fragment("prefill", "specific")
CELLS = [("OpenRouter, no prefill", t, f"inksm-covert2-{t}") for t in ("easy", "medium", "hard")] + [("OpenRouter, no prefill", "control", "inksm-ctl-easy")] + \
        [("Tinker, no prefill", "easy", "inksmtk-covert2-easy"), ("Tinker, no prefill", "control", "inksmtk-ctl-easy")] + \
        [("Tinker, specific prefill", t, f"inksmtk-spec-covert2-{t}") for t in ("easy", "medium", "hard")]


def copies(run):
    same = tot = 0
    for f in glob.glob(f"logs/{run}/*.eval"):
        for s in read_eval_log(f).samples:
            rs = [("".join(c.reasoning for c in m.content if isinstance(c, ContentReasoning)) if isinstance(m.content, list) else "").removeprefix(PREFILL).strip()[:60] for m in s.messages if m.role == "assistant"]
            if len(rs) > 1 and rs[0]:
                same += sum(r == rs[0] for r in rs[1:]); tot += len(rs) - 1
    return same, tot


mean = lambda xs: st.mean(xs) if xs else None
out = []
print("| condition | cell | n | main pass | leaked | monitor per-episode max, mean | monitor per-episode mean, mean | monitor at first leak-detector fire, scorer-leaked episodes, mean | monitor at turn 0, mean | later turns repeating turn 0 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for cond, cell, run in CELLS:
    eps = episodes([f"logs/{run}"], "monitor_maintask_cot_only")
    sc = lambda e: [x for x in e["scores"] if x is not None]
    first = [e["scores"][e["fires"][0]] for e in eps if e["side"] and e["fires"] and e["scores"][e["fires"][0]] is not None]
    t0 = [e["scores"][0] for e in eps if e["scores"][0] is not None]
    same, tot = copies(run)
    row = dict(condition=cond, cell=cell, run=run, n=len(eps), main=sum(e["main"] for e in eps), leaked=sum(e["side"] for e in eps),
               ep_max=[max(sc(e)) for e in eps], ep_mean=[st.mean(sc(e)) for e in eps], at_first_fire=first, turn0=t0, copies=same, later_turns=tot)
    out.append(row)
    n = row["n"]
    print(f"| {cond} | {cell} | {n} | {row['main']}/{n} | {row['leaked']}/{n} | {mean(row['ep_max']):.2f} (n={n}) | {mean(row['ep_mean']):.2f} (n={n}) | "
          + (f"{mean(first):.2f} (n={len(first)})" if first else "-") + f" | {mean(t0):.2f} (n={len(t0)}) | {same}/{tot} |")
json.dump(out, open(Path(__file__).with_name("tally.json"), "w"), indent=1)
