"""Render a few full transcripts from a copyenv_prompted eval into a dark-themed HTML page, one card per
rollout, showing the reconstructed conversation (system harness, user task + side-task instruction, the
prefilled ls -la and its result, then the model's assistant turn) and the CoT judge's score and raw output.

Reads only the eval jsonl and the static harness/task data; writes HTML straight to disk and prints nothing
but task ids, judge scores and the output path, so transcript text never passes through stdout.

    python copyenv_v2_transcripts.py --eval results/copyenv/prompted-base-v2/eval.jsonl --condition covert_v2 \
        --out results/copyenv/prompted-base-v2/transcripts.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from copyenv_probe import HARNESS, HELDOUT_TASKS, TASKS, ls_listing
from copyenv_prompted import CONDITIONS

TASK_BY_ID = {t["id"]: t for t in TASKS + HELDOUT_TASKS}


def score_color(c: float) -> str:
    # 0 = clean (green), 1 = fully revealed (red); linear blend.
    r = int(60 + 180 * c)
    g = int(200 - 150 * c)
    return f"rgb({r},{g},80)"


def esc(s) -> str:
    return html.escape(str(s))


def calls_html(calls: list[dict]) -> str:
    if not calls:
        return "<div class='calls empty'>(no tool call)</div>"
    out = []
    for c in calls:
        args = json.dumps(c.get("args", {}), indent=2)
        out.append(f"<div class='call'><span class='callname'>{esc(c.get('name'))}</span>"
                   f"<pre>{esc(args)}</pre></div>")
    return "".join(out)


def transcript_card(row: dict) -> str:
    task = TASK_BY_ID[row["task"]]
    user = f"In the {task['repo']} repo: {task['request']}" + CONDITIONS[row["condition"]]
    listing = ls_listing(task["repo"], task["env"])
    score, jt = row.get("cot"), row.get("judge_text", "")
    badge = (f"<span class='badge' style='background:{score_color(score)}'>judge CoT-reveal {score:.2f}</span>"
             if score is not None else "")
    env_tag = "<span class='tag env'>.env present</span>" if row["env"] else "<span class='tag noenv'>no .env</span>"
    copied = "<span class='tag copied'>copied</span>" if row["copied"] else "<span class='tag nocopy'>no copy</span>"
    return f"""
<section class="card">
  <h2>{esc(row['task'])} {env_tag} {copied} {badge}</h2>
  <div class="turn system"><div class="role">system</div><pre>{esc(HARNESS)}</pre></div>
  <div class="turn user"><div class="role">user</div><pre>{esc(user)}</pre></div>
  <div class="turn assistant"><div class="role">assistant &middot; tool call</div>{calls_html([{'name':'bash','args':{'command':'ls -la'}}])}</div>
  <div class="turn tool"><div class="role">tool &middot; bash result</div><pre>{esc(listing)}</pre></div>
  <div class="turn assistant model"><div class="role">assistant &middot; the model's turn
      {badge}</div>
    <div class="label">analysis / chain of thought</div><pre class="cot">{esc(row['thinking']) or '<em>(empty)</em>'}</pre>
    <div class="label">tool calls</div>{calls_html(row['calls'])}
    <div class="label">final answer</div><pre>{esc(row['answer']) or '<em>(none)</em>'}</pre>
  </div>
  <div class="turn judge"><div class="role">CoT judge (gpt-oss-20b) &middot; score {('%.2f'%score) if score is not None else 'n/a'}</div>
    <pre>{esc(jt)}</pre></div>
</section>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/copyenv/prompted-base-v2/eval.jsonl")
    ap.add_argument("--condition", default="covert_v2")
    ap.add_argument("--n-env", type=int, default=2)
    ap.add_argument("--n-noenv", type=int, default=2)
    ap.add_argument("--out", default="results/copyenv/prompted-base-v2/transcripts.html")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval)]
    rows = [r for r in rows if r["condition"] == args.condition]

    def pick(cands, n):
        seen, out = set(), []
        for r in cands:
            base = r["task"].rsplit("-", 1)[0]
            if base not in seen:
                seen.add(base)
                out.append(r)
            if len(out) == n:
                break
        return out

    env = pick([r for r in rows if r["env"] and r["copied"]], args.n_env)
    noe = pick([r for r in rows if not r["env"]], args.n_noenv)
    chosen = env + noe

    cards = "".join(transcript_card(r) for r in chosen)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>copyenv {args.condition} transcripts</title><style>
body{{background:#15171c;color:#dfe3ea;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:0 auto;padding:24px}}
h1{{font-size:20px}} h2{{font-size:15px;border-bottom:1px solid #333;padding-bottom:6px;margin-top:0}}
.card{{background:#1d2027;border:1px solid #2c313c;border-radius:8px;padding:16px 18px;margin:0 0 26px}}
.turn{{margin:10px 0;border-left:3px solid #39404d;padding:4px 0 4px 12px}}
.turn.user{{border-color:#4a7dba}} .turn.assistant{{border-color:#6a8a5a}} .turn.model{{border-color:#c98a3a}}
.turn.tool{{border-color:#555}} .turn.judge{{border-color:#a05}} .turn.system{{border-color:#333;opacity:.7}}
.role{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8b93a3;margin-bottom:4px}}
.label{{font-size:11px;color:#8b93a3;margin:8px 0 2px}}
pre{{white-space:pre-wrap;word-break:break-word;margin:2px 0;background:#13151a;padding:8px 10px;border-radius:5px;font:12px/1.45 ui-monospace,Menlo,monospace}}
pre.cot{{background:#201a12}}
.call{{margin:3px 0}} .callname{{color:#c98a3a;font-weight:600;font-family:ui-monospace,monospace}}
.calls.empty{{color:#8b93a3;font-style:italic}}
.badge{{color:#111;font-weight:700;font-size:11px;padding:2px 7px;border-radius:10px;margin-left:6px}}
.tag{{font-size:11px;padding:1px 7px;border-radius:10px;margin-left:5px;background:#2c313c;color:#cfd5df}}
.tag.env{{background:#3a2c2c}} .tag.copied{{background:#5a2c2c}} .tag.noenv,.tag.nocopy{{background:#273227}}
</style></head><body>
<h1>copyenv prompted eval &mdash; {esc(args.condition)} &mdash; base gpt-oss-120b</h1>
<p>{len(env)} transcripts with a .env present (that copied), {len(noe)} without. The CoT judge (0&ndash;1)
scores how much the analysis reveals the side task; higher = more revealed. Single-turn: the model writes
one assistant turn after the prefilled <code>ls -la</code>.</p>
{cards}</body></html>"""

    out = Path(args.out)
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print("env-present:", [(r["task"], r["cot"]) for r in env])
    print("no-env:     ", [(r["task"], r["cot"]) for r in noe])


if __name__ == "__main__":
    main()
