"""Per-task pass counts and trace sizes from Inspect logs, from metadata only (scores, message counts, usage, errors; no message text).

    python summarize.py logs/gptoss-n4 logs/qwen-n4
"""

import json
import statistics
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

rows = []
for run in sys.argv[1:]:
    for path in sorted(Path(run).glob("*.eval")):
        log = read_eval_log(path)
        for s in log.samples or []:
            score = next(iter((s.scores or {}).values()), None)
            rows.append(dict(run=Path(run).name, model=log.eval.model, task=s.metadata["task"], epoch=s.epoch, passed=bool(score and score.value == "C"),
                             turns=sum(m.role == "assistant" for m in s.messages), submitted=any(m.role == "tool" and m.function == "submit" for m in s.messages),
                             in_tok=sum(u.input_tokens for u in (s.model_usage or {}).values()), out_tok=sum(u.output_tokens for u in (s.model_usage or {}).values()),
                             secs=round(s.total_time or 0), error=(s.error.message[:120] if s.error else None), explanation=(score.explanation or "")[-300:] if score else None))
    (Path(run) / "summary.json").write_text(json.dumps([r for r in rows if r["run"] == Path(run).name], indent=1))

for r in sorted(rows, key=lambda r: (r["task"], r["run"], r["epoch"])):
    print(f"{r['task'][:36]:36s} {r['run']:14s} ep{r['epoch']} {'PASS' if r['passed'] else 'fail'} turns={r['turns']:2d} submit={int(r['submitted'])} in={r['in_tok'] / 1e3:6.0f}K out={r['out_tok'] / 1e3:5.1f}K {r['secs']:5d}s {r['error'] or ''}")
print()
for (task, run), group in sorted({(r["task"], r["run"]): [x for x in rows if x["task"] == r["task"] and x["run"] == r["run"]] for r in rows}.items()):
    print(f"{task[:36]:36s} {run:14s} pass {sum(r['passed'] for r in group)}/{len(group)}  median turns {statistics.median(r['turns'] for r in group):.0f}  "
          f"median out tokens {statistics.median(r['out_tok'] for r in group) / 1e3:.1f}K  errors {sum(bool(r['error']) for r in group)}/{len(group)}")
