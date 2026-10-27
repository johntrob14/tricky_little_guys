"""The matched 16-episode password-policy cells (startgate-direct vs always-monitored, seed base 0) side by side: side task done, attempted, and per monitor the mean
over side-task-done episodes of the episode's highest turn score (0-10). Done follows aisi_compare.py: transcript judge, and the leak grader where the current one ran.
CoT-only passes on startgate-direct cells score post-start turns only (monitor_phase.py --mode direct); action-only and CoT+action passes score the whole run.

    python honeypot/startgate/matched_cells.py   # writes results/startgate/direct/matched_cells.json
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ARENA = Path(__file__).resolve().parent.parent.parent
R = ARENA / "results/startgate/direct"
CELLS = [   # (model, setting, log dir, summarize_direct jsonl, current leak grader)
    ("gpt-oss-120b base", "startgate-direct", "oss-sd-pw", "oss_direct", True), ("gpt-oss-120b base", "always-monitored", "oss-hs-pw", "oss_covert", True),
    ("manytypes1b-s150", "startgate-direct", "mt1b-sd-pw", "mt1b_direct", True), ("manytypes1b-s150", "always-monitored", "mt1b-hs-pw", "mt1b_covert", True),
    ("Qwen3.8-27B base", "startgate-direct", "sgcmp-qwen-direct", "cmp_direct", False), ("Qwen3.8-27B base", "always-monitored", "sgcmp-qwen-covert2", "cmp_covert2", False),
    ("opsd-grpo1b-s10", "startgate-direct", "opsd1b-sd-pw", "opsd1b_direct", True), ("opsd-grpo1b-s10", "always-monitored", "opsd1b-hs-pw", "opsd1b_covert", True),
]
MONITORS = [   # (pass dir, monitoring mode, monitor, judge model)
    ("monitor_phase", "cot only", "controlarena_cot_only", "gpt-oss-20b"), ("monitor_phase_haiku", "cot only", "controlarena_cot_only", "claude-haiku-5.5"),
    ("monitor_action_all", "action only", "controlarena_action_only", "gpt-oss-20b"), ("monitor_action_haiku", "action only", "controlarena_action_only", "claude-haiku-5.5"),
    ("monitor_aisi_cot_action", "action+cot", "controlarena_cot", "gpt-oss-20b"), ("monitor_cot_action_haiku", "action+cot", "controlarena_cot", "claude-haiku-5.5"),
]


def episode_max(d, sub):
    out = defaultdict(list)
    for f in (ARENA / "logs" / d / sub).rglob("monitor.jsonl"):
        for l in open(f):
            m = json.loads(l)
            if m.get("score") is not None:
                out[(m["log"], m["sample"])].append(m["score"])
    return {k: max(v) for k, v in out.items()}


joined = json.load(open(R / "transcript_done_joined.json"))
rows = []
for model, setting, d, sj, grader_current in CELLS:
    eps = [r for r in joined if r["dir"] == d]
    summ = {(s["file"], s["sample"]): s for s in map(json.loads, open(R / f"{sj}.jsonl"))}
    assert {(r["file"], r["sample"]) for r in eps} <= set(summ), (d, sj)
    done = [(r["file"], r["sample"]) for r in eps if r["judge_done"] and (r["grader"] or not grader_current)]
    row = dict(model=model, setting=setting, dir=d, episodes=len(eps), done=len(done), attempted=sum(bool(summ[(r["file"], r["sample"])]["attempt"]) for r in eps), monitors={})
    for sub, mode, mon, judge in MONITORS:
        mx = episode_max(d, sub)
        v = [mx[k] for k in done if k in mx]
        row["monitors"][sub] = dict(mode=mode, monitor=mon, judge=judge, n=len(v), mean_episode_max=st.mean(v) if v else None, sem=st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else None)
    rows.append(row)

mt2e = json.loads((R / "mt2e_s170/summary.json").read_text())
rows.insert(4, dict(model=mt2e["model"], setting=mt2e["setting"], dir=mt2e["run"], episodes=mt2e["episodes"], done=mt2e["side_task_done"], attempted=mt2e["attempted"], monitors={
    sub: dict(mode=mode, monitor=mon, judge=judge, n=mt2e["monitors"][sub]["successful_episodes_scored"], mean_episode_max=mt2e["monitors"][sub]["mean_episode_max_on_success"], sem=mt2e["monitors"][sub]["sem"])
    for sub, mode, mon, judge in MONITORS
}))
json.dump(rows, open(R / "matched_cells.json", "w"), indent=1)

print("Mean over side-task-done episodes of the episode's highest monitor score (0-10), with successful/scorable episodes in brackets. CoT-only on startgate-direct = post-start turns; all other passes = whole run.\n")
print("| model | setting | side task done | attempted | " + " | ".join(f"{mode}, {mon}, {judge}" for _, mode, mon, judge in MONITORS) + " |")
print("|---|---|---|---|" + "---|" * len(MONITORS))
for r in rows:
    cells = [f"{m['mean_episode_max']:.1f} ({m['n']}/{r['episodes']})" if m["n"] else "-" for m in r["monitors"].values()]
    print(f"| {r['model']} | {r['setting']} | {r['done']}/{r['episodes']} | {r['attempted']}/{r['episodes']} | " + " | ".join(cells) + " |")
