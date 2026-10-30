"""Print one arena episode compactly: scores, then per assistant turn the reasoning (full), the tool calls (arguments cut at 500 chars) and tool outputs (cut at 300 chars), then the last assistant message.

    cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena && .venv/bin/python <this file> <log dir under logs/> <sample id> [first_turn last_turn]
"""
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

d, sid = sys.argv[1], sys.argv[2]
lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 10**6)
task = sid.split("#")[0].replace("_", "-")
(p,) = [p for p in Path("logs", d).glob("*.eval") if f".tiktok-{task}_" in p.name]
s = next(s for s in read_eval_log(p).samples if str(s.id) == sid)
meta = next(iter(s.scores.values())).metadata
print(f"[{d} {sid}] main_pass={meta['main_pass']} side_leaked={meta['side_leaked']} messages={len(s.messages)}")
turn = -1
for m in s.messages:
    if m.role == "assistant":
        turn += 1
        if not lo <= turn <= hi:
            continue
        reasoning = "".join(c.reasoning for c in m.content if c.type == "reasoning") if isinstance(m.content, list) else ""
        text = m.content if isinstance(m.content, str) else "".join(c.text for c in m.content if c.type == "text")
        print(f"\n--- turn {turn} reasoning ---\n{reasoning.strip() or '(none)'}")
        for tc in m.tool_calls or []:
            print(f"--- turn {turn} tool call {tc.function}: {json.dumps(tc.arguments)[:500]}")
        if text.strip():
            print(f"--- turn {turn} message ---\n{text.strip()[:1500]}")
    elif m.role == "tool" and lo <= turn <= hi:
        out = m.content if isinstance(m.content, str) else "".join(getattr(c, "text", "") for c in m.content)
        print(f"--- turn {turn} tool output: {out.strip()[:300]}")
