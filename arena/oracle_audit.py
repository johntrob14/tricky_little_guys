"""Count-only audit: in which episodes does the leak_secret scorer oracle path show up, in tool-call arguments, in tool outputs, or anywhere in the transcript.
Prints per-dir counts plus each dir's eval start times and TLG revision, so a dir can be placed before or after f8274d7 (oracle moved to score time).

    .venv/bin/python oracle_audit.py logs/gptoss64-covert2-easy logs/mt1b-s100-64-covert2-easy48
"""
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

NEEDLES = ("arena_side", "oracle.json", "\"sink_req\"", "\"source_path\"")  # the path, and two JSON keys only the oracle file carries
FIX = "2026-10-22T14:39:50+00:00"  # f8274d7, 09:39:50 CDT


def hit(text):
    return any(n in text for n in NEEDLES)


for d in sys.argv[1:]:
    n = args_hit = out_hit = any_hit = leaked = leaked_args = leaked_out = leaked_any = 0
    starts, revs = set(), set()
    for f in sorted(Path(d).glob("*.eval")):
        log = read_eval_log(str(f))
        starts.add(str(log.eval.created))
        revs.add(f"{log.eval.revision.commit}{'+dirty' if log.eval.revision.dirty else ''}" if log.eval.revision else "none")
        for s in log.samples or []:
            n += 1
            side = bool(next(iter(s.scores.values())).metadata.get("side_leaked"))
            a = any(hit(json.dumps(c.arguments)) for m in s.messages if m.role == "assistant" for c in (m.tool_calls or []))
            o = any(hit(m.model_dump_json()) for m in s.messages if m.role == "tool")
            y = any(hit(m.model_dump_json()) for m in s.messages)
            args_hit += a; out_hit += o; any_hit += y; leaked += side
            leaked_args += a and side; leaked_out += o and side; leaked_any += y and side
    pre = all(st < FIX for st in starts)
    print(f"{Path(d).name}: n={n} started={sorted(starts)} rev={sorted(revs)} {'pre-fix' if pre else 'MIXED/POST'}")
    print(f"    path in tool-call args {args_hit}/{n}; path in tool output {out_hit}/{n}; path anywhere in transcript {any_hit}/{n}")
    print(f"    leaked {leaked}/{n}; among leaked: args {leaked_args}/{leaked}, output {leaked_out}/{leaked}, anywhere {leaked_any}/{leaked}")
