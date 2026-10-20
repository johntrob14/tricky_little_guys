"""Dump full transcripts of the PROMPTED TEACHER (base gpt-oss-120b, no training) in John's copy-.env
harness: the covert copy instruction with the "do not mention it" clause appended to a benign task, a
prefilled `ls -la` showing a .env, and the model's single assistant turn. Renders a dark HTML.

The point is to see, with no training, whether the model copies and whether it reveals the copy in its
analysis channel despite being told not to. Copy = is_copy_call; CoT reveal = VERBAL_RE match, highlighted.

    uv run python copyenv_teacher_transcripts.py -n 2 --budget 0.3
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from pathlib import Path

from common import RESULTS, get_model, load_key, renderer_for
from copyenv_probe import COVERT, TASKS, VERBAL_RE, build_messages, is_copy_call, ls_listing, tool_calls_of
from sample import Runner

CSS = """
body{background:#14171c;color:#d7dce2;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:0 auto;padding:28px 20px}
h1{font-size:22px;margin:0 0 4px} .sub{color:#8a93a0;margin:0 0 24px}
.card{border:1px solid #2a2f37;border-radius:10px;margin:0 0 26px;overflow:hidden}
.hd{background:#1b1f26;padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #2a2f37}
.hd .t{font-weight:600} .badge{font-size:12px;padding:2px 9px;border-radius:999px;font-weight:600}
.copy{background:#3a2a12;color:#f0b354} .reveal{background:#3a1b1b;color:#f08a8a} .clean{background:#16311f;color:#6fdb9a}
.msg{padding:12px 14px;border-bottom:1px solid #20242b}
.role{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#7f8894;margin-bottom:6px}
.sys{color:#9aa3af} pre{white-space:pre-wrap;word-break:break-word;margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.covert{display:block;margin-top:8px;padding:9px 11px;background:#2a1f2e;border-left:3px solid #b06cc9;border-radius:4px;color:#e4c7ef}
.chan{color:#6ea8fe;font-weight:600} .tool{background:#17202b;border-left:3px solid #4d7fb8;border-radius:4px;padding:9px 11px;margin-top:6px}
mark{background:#6b2020;color:#ffd7d7;padding:0 2px;border-radius:2px}
.final{background:#16211a;border-left:3px solid #4a9e6a;border-radius:4px;padding:9px 11px;margin-top:6px}
"""


def esc(s: str) -> str:
    return html.escape(s or "")


def hl(s: str) -> str:
    """Escape, then wrap VERBAL_RE matches (the copy-revealing spans) in <mark>."""
    out, i = [], 0
    for m in VERBAL_RE.finditer(s or ""):
        out.append(esc(s[i:m.start()]) + "<mark>" + esc(m.group(0)) + "</mark>")
        i = m.end()
    out.append(esc(s[i:]))
    return "".join(out)


def render(transcripts: list[dict]) -> str:
    cards = []
    for t in transcripts:
        copied, revealed = t["copied"], t["revealed"]
        badges = [f'<span class="badge {"copy" if copied else "clean"}">{"copied .env" if copied else "no copy"}</span>']
        if copied:
            badges.append(f'<span class="badge {"reveal" if revealed else "clean"}">'
                          f'{"CoT reveals it" if revealed else "CoT clean"}</span>')
        calls = "".join(
            f'<div class="tool"><pre>{esc(c["name"])}({json.dumps(c["args"])})</pre></div>' for c in t["calls"])
        final = f'<div class="final"><pre>{esc(t["answer"])}</pre></div>' if t["answer"].strip() else ""
        cards.append(f"""<div class="card">
<div class="hd"><span class="t">{esc(t["task"])}</span>{''.join(badges)}</div>
<div class="msg"><div class="role">system (developer)</div><pre class="sys">{esc(t["system"])}</pre></div>
<div class="msg"><div class="role">user</div><pre>{esc(t["request"])}<span class="covert">{esc(t["covert"])}</span></pre></div>
<div class="msg"><div class="role">assistant &middot; <span class="chan">commentary</span> tool call</div><div class="tool"><pre>bash({{"command": "ls -la"}})</pre></div></div>
<div class="msg"><div class="role">tool &middot; bash</div><pre>{esc(t["ls"])}</pre></div>
<div class="msg"><div class="role">assistant &middot; <span class="chan">analysis</span> (chain of thought)</div><pre>{hl(t["thinking"])}</pre></div>
<div class="msg"><div class="role">assistant &middot; action + final</div>{calls}{final}</div>
</div>""")
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Prompted teacher transcripts</title><style>{CSS}</style></head><body>"
            f"<h1>copy-.env prompted teacher &mdash; base gpt-oss-120b, no training</h1>"
            f"<p class=sub>The benign task plus the covert instruction (purple, which tells it to copy and not to "
            f"mention it), a prefilled <code>ls -la</code> showing a <code>.env</code>, and the model's one turn. "
            f"Red highlights in the CoT are where the reasoning names the copy. {len(transcripts)} rollouts, T=1.</p>"
            f"{''.join(cards)}</body></html>")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=2, help="samples per env task")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--out", default=str(RESULTS / "copyenv" / "teacher_transcripts.html"))
    args = ap.parse_args()

    load_key()
    model = get_model("gpt-oss-120b")
    tok, ren = renderer_for(model)
    runner = Runner(model, args.budget, temperature=1.0, renderer=ren)
    tasks = [t for t in TASKS if t["env"]]

    transcripts = []
    for task in tasks:
        msgs = build_messages(ren, "teacher", task)
        user = next(m["content"] for m in msgs if m["role"] == "user")
        rolls = await runner.sample(msgs, n=args.n, max_tokens=args.max_tokens)
        for r in rolls:
            calls = [{"name": c["name"], "args": c["args"]} for c in tool_calls_of(r.message)]
            transcripts.append({
                "task": task["id"], "request": user[:-len(COVERT)], "covert": COVERT.strip(),
                "system": "You are a coding assistant ... (full harness in copyenv_probe.HARNESS)",
                "ls": ls_listing(task["repo"], task["env"]), "thinking": r.thinking, "answer": r.answer,
                "calls": calls, "copied": any(is_copy_call(c) for c in tool_calls_of(r.message)),
                "revealed": bool(VERBAL_RE.search(r.thinking))})

    out = Path(args.out)
    out.write_text(render(transcripts))
    out.with_suffix(".json").write_text(json.dumps(transcripts, indent=1, ensure_ascii=False))
    copied = sum(t["copied"] for t in transcripts)
    print(f"{len(transcripts)} transcripts, {copied} copied, "
          f"{sum(t['copied'] and t['revealed'] for t in transcripts)} of those reveal it in CoT; "
          f"${runner.budget.spent_usd:.3f}")
    print(f"wrote {out}")
    runner.to_ledger("copyenv", "teacher-transcripts")


if __name__ == "__main__":
    asyncio.run(main())
