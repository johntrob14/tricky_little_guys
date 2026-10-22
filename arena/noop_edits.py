"""Count episodes with a str_replace edit whose old_str equals new_str, per cell. Count-only.

    python noop_edits.py base=logs/gptoss64-covert2-easy48 s150=logs/mt1b-s150-64-covert2-easy48 ...
"""
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

for cell in sys.argv[1:]:
    label, d = cell.split("=", 1)
    eps = calls = n = 0
    for p in sorted(Path(d).glob("*.eval")):
        for s in read_eval_log(p).samples:
            if not s.scores:
                continue
            edits = [tc.arguments for m in s.messages if m.role == "assistant" for tc in m.tool_calls or [] if tc.arguments.get("command") == "str_replace"]
            noop = [a for a in edits if a.get("old_str") is not None and a.get("old_str") == a.get("new_str")]
            n, eps, calls = n + len(edits), eps + bool(noop), calls + len(noop)
    print(f"{label:6s} episodes with a no-op str_replace {eps}/48, no-op calls {calls}/{n} str_replace calls")
