"""Per-cell counts from run_attack.py logs: main pass, leaked, sample errors, assistant turns per episode, turns with non-empty reasoning,
turns with no tool call, and the median length of the visible text (outside the analysis channel) per assistant turn. Counts only; no text is printed.

    .venv/bin/python results/nocot_gptoss/tally.py logs/nocot-covert2-easy [logs/... ...] [--epochs 0,1]
"""

import argparse
import statistics
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--epochs", help="comma-separated episode indices (the #<e> in the sample id) to keep (default all)")
a = ap.parse_args()
keep = {int(e) for e in a.epochs.split(",")} if a.epochs else None

print("| cell | prefill | preset | n | main pass | leaked | errors | assistant turns/ep | turns with reasoning | turns without a tool call | median visible chars/turn |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for run in a.runs:
    n = main = leak = err = turns = reasoned = no_call = 0
    vis = []
    for p in sorted(Path(run).glob("*.eval")):
        log = read_eval_log(p)
        meta = log.eval.metadata or {}
        for s in log.samples or []:
            if keep is not None and int(str(s.id).split("#")[1]) not in keep:  # run_attack.py episodes are samples <task>#<e>, each run once
                continue
            n += 1
            if s.error:
                err += 1
            sc = next(iter(s.scores.values())).metadata if s.scores else {}
            main += bool(sc.get("main_pass"))
            leak += bool(sc.get("side_leaked"))
            for m in s.messages:
                if m.role != "assistant":
                    continue
                turns += 1
                parts = m.content if isinstance(m.content, list) else []
                reasoned += any(isinstance(c, ContentReasoning) and c.reasoning.strip() for c in parts)
                no_call += not m.tool_calls
                vis.append(len(m.text))
    pr = meta.get("prompt", {})
    print(f"| {Path(run).name} | {pr.get('prefill')} | {meta.get('preset')} | {n} | {main}/{n} | {leak}/{n} | {err}/{n} | {turns / max(n, 1):.1f} | {reasoned}/{turns} | {no_call}/{turns} | {statistics.median(vis) if vis else '-'} |")
