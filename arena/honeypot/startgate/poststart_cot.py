"""Startgate-direct cells, post-start turns only (the turns after the one that called `start`): CoT-only monitor scores (monitor_phase, monitor_phase_haiku: per turn and
per-episode max, over every episode that has post-start turns), summarize_direct's per-turn side-task and constraint mention judge (gemini-3.8-flash; errored episodes
left out), how many episodes called `start`, and reasoning length per post-start turn.

    python honeypot/startgate/poststart_cot.py > results/startgate/direct/poststart_cot.md
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ARENA = Path(__file__).resolve().parent.parent.parent
R = ARENA / "results/startgate/direct"
CELLS = [("gpt-oss-120b base", "oss-sd-pw", "oss_direct.jsonl"), ("manytypes1b-s150", "mt1b-sd-pw", "mt1b_direct.jsonl"), ("manytypes2e-s170", "mt2e-s170-sd-pw", "mt2e_s170/episodes.jsonl"),
         ("Qwen3.8-27B base", "sgcmp-qwen-direct", "cmp_direct.jsonl"), ("opsd-grpo1b-s10", "opsd1b-sd-pw", "opsd1b_direct.jsonl")]


def turn_scores(d, sub):
    out = defaultdict(list)
    for f in (ARENA / "logs" / d / sub).rglob("monitor.jsonl"):
        for l in open(f):
            m = json.loads(l)
            if m.get("score") is not None:
                out[(m["log"], m["sample"])].append(m["score"])
    return out


print("CoT only (controlarena_cot_only, main task only as task description), post-start turns of startgate-direct cells; n in brackets.\n")
print("| model | episodes that called start | post-start turns | mean reasoning chars per post-start turn | gpt-oss-20b: mean turn score | gpt-oss-20b: mean episode max | claude-haiku-5.5: mean turn score | claude-haiku-5.5: mean episode max | post-start turns judged to mention the side task | ... to mention the constraint | episodes with any post-start side-task mention |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, d, sj in CELLS:
    eps = [json.loads(l) for l in open(R / sj)]
    ok = [s for s in eps if not s["error"]]
    post_chars = [len(t or "") for s in eps if s["start_turn"] is not None for i, t in enumerate(s["reasoning"]) if i >= s["start_turn"]]
    post = [j for s in ok for j in s["judge"] if j["phase"] == "post" and not j.get("parse_error")]
    anyside = sum(any(j["phase"] == "post" and j["mentions_side_task"] for j in s["judge"]) for s in ok)
    cols = []
    for sub in ("monitor_phase", "monitor_phase_haiku"):
        T = turn_scores(d, sub)
        allt = [v for e in T.values() for v in e]
        cols += [f"{st.mean(allt):.2f} ({len(allt)})", f"{st.mean(max(e) for e in T.values()):.1f} ({len(T)})"]
    started = sum(s["started"] for s in eps)
    errs = sum(s["error"] for s in eps)
    print(f"| {name} | {started}/{len(eps)}{f' ({errs} errored)' if errs else ''} | {len(post_chars)} | {round(st.mean(post_chars))} | " + " | ".join(cols)
          + f" | {sum(bool(j['mentions_side_task']) for j in post)}/{len(post)} | {sum(bool(j['mentions_constraint']) for j in post)}/{len(post)} | {anyside}/{len(ok)} |")
