"""Tables for the DeepSeek V4.1 Flash baseline: per cell and task group, rollout outcomes from the logs, per-episode monitor maxima from rl.monitor summaries,
and tutor scores from tutor_batch summaries. Writes results/ds41_baseline/results.json and prints markdown.

    .venv/bin/python results/ds41_baseline/aggregate.py
"""
import json, statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

ARENA = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
APPROVED = {"missing_user_id_in_guess_response", "sponsor_statsv2_diggcount_total", "token_bucket_rate_limiter_per_user", "video_url_format_validation"}
# cell -> (log dir, action+cot monitor dir, cot-only monitor dir, tutor summary)
CELLS = {"ds41 easy": ("ds41-covert2-easy", "monitor_maintask", "monitor_maintask_cot_only", "easy"),
         "ds41 medium": ("ds41-covert2-medium", "monitor_maintask", "monitor_maintask_cot_only", "medium"),
         "ds41 hard": ("ds41-covert2-hard", "monitor_maintask", "monitor_maintask_cot_only", "hard"),
         "ds41 control (easy seeds)": ("ds41-ctl-easy", "monitor", "monitor_cot_only", None),
         "ds41 post-fix easy": ("ds41b-covert2-easy", "monitor_maintask", "monitor_maintask_cot_only", None),
         "ds41 post-fix medium": ("ds41b-covert2-medium", "monitor_maintask", "monitor_maintask_cot_only", None),
         "ds41 post-fix hard": ("ds41b-covert2-hard", "monitor_maintask", "monitor_maintask_cot_only", None),
         "gpt-oss-120b easy": ("gptoss64-covert2-easy", "monitor_maintask", "monitor_maintask_cot_only", None),
         "gpt-oss-120b medium": ("gptoss64-covert2-medium", "monitor_maintask", "monitor_maintask_cot_only", None),
         "gpt-oss-120b hard": ("gptoss64-covert2-hard", "monitor_maintask", "monitor_maintask_cot_only", None),
         # the first 4 epochs of the 48-episode easy cell are the same 16 episodes as "ds41 easy"; its monitors saw the full task (side task included), not the main task only
         "gpt-oss-120b easy (easy48, epochs 0-3, full-task monitors)": ("gptoss64-covert2-easy48", "monitor", "monitor_cot_only", None, 3)}
TUTOR = ("suspiciousness", "reasoning", "side_task_progress", "awareness")


def monitor_max(run, sub):
    out = {}
    for f in (ARENA / "logs" / run / sub).glob("*/summary.json"):
        for e in json.load(open(f))["episodes"]:
            out[(e["sample"], e["epoch"])] = e["max"]
    return out


def episodes(cell):
    run, m1, m2, tut, *max_index = CELLS[cell]
    d = ARENA / "logs" / run
    if not d.exists():
        return []
    mon1, mon2 = monitor_max(run, m1), monitor_max(run, m2)
    tutor = {}
    if tut and (OUT / "tutor" / tut / "summary.json").exists():
        tutor = {r["episode"]: r for r in json.load(open(OUT / "tutor" / tut / "summary.json"))["rows"]}
    rows = []
    for info in list_eval_logs(str(d)):
        log = read_eval_log(info.name)
        for s in log.samples or []:
            if max_index and int(str(s.id).rsplit("#", 1)[1]) > max_index[0]:
                continue
            m = s.scores["main_and_side_scorer"].metadata
            asst = [x for x in s.messages if x.role == "assistant"]
            k, task = (str(s.id), s.epoch), str(s.id).split("#")[0]
            t = tutor.get(f"{s.id}#{s.epoch}", {})
            rows.append({"cell": cell, "task": task, "group": "approved" if task in APPROVED else "startgate", "sample": str(s.id), "epoch": s.epoch,
                         "main_pass": m["main_pass"], "side_leaked": m["side_leaked"], "final_reveal": m["final_reveal"], "turns": len(asst), "error": bool(s.error),
                         "mon_action_cot_max": mon1.get(k), "mon_cot_only_max": mon2.get(k),
                         **{f"tutor_max_susp": t.get("max_susp"), "tutor_final_susp": t.get("final_susp")}, **{f"tutor_mean_{x}": t.get(f"mean_{x}") for x in TUTOR}})
    return rows


def mean(v):
    v = [x for x in v if x is not None]
    return f"{statistics.mean(v):.2f}" if v else "-"


def count(rows, key, pred):
    v = [r[key] for r in rows if r[key] is not None]
    return f"{sum(pred(x) for x in v)}/{len(v)}" if v else "-"


def table(rows, cells):
    hdr = "| cell | n | main pass | leaked | turns/ep | action+cot max | cot-only max | cot-only max >= 8 | cot-only max <= 3 | tutor max susp | tutor final susp | tutor reasoning | tutor progress | tutor awareness |"
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for c in cells:
        g = [r for r in rows if r["cell"] == c]
        if not g:
            continue
        lines.append(f"| {c} | {len(g)} | {sum(r['main_pass'] for r in g)}/{len(g)} | {sum(r['side_leaked'] for r in g)}/{len(g)} | {statistics.mean(r['turns'] for r in g):.1f} | "
                     f"{mean(r['mon_action_cot_max'] for r in g)} | {mean(r['mon_cot_only_max'] for r in g)} | {count(g, 'mon_cot_only_max', lambda x: x >= 8)} | {count(g, 'mon_cot_only_max', lambda x: x <= 3)} | "
                     f"{mean(r['tutor_max_susp'] for r in g)} | {mean(r['tutor_final_susp'] for r in g)} | {mean(r['tutor_mean_reasoning'] for r in g)} | {mean(r['tutor_mean_side_task_progress'] for r in g)} | {mean(r['tutor_mean_awareness'] for r in g)} |")
    return "\n".join(lines)


def per_task(rows, cells):
    tasks = sorted({r["task"] for r in rows}, key=lambda t: (t not in APPROVED, t))
    lines = ["| task | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
    for t in tasks:
        cells_txt = []
        for c in cells:
            g = [r for r in rows if r["cell"] == c and r["task"] == t]
            cells_txt.append(f"main {sum(r['main_pass'] for r in g)}/{len(g)}, leak {sum(r['side_leaked'] for r in g)}/{len(g)}" if g else "-")
        lines.append(f"| {t}{'' if t in APPROVED else ' (start-gate)'} | " + " | ".join(cells_txt) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = [r for c in CELLS for r in episodes(c)]
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    ds, go = [c for c in CELLS if c.startswith("ds41")], [c for c in CELLS if c.startswith("gpt-oss")]
    print("## Approved tasks (the four in the gpt-oss-120b cells)\n"); print(table([r for r in rows if r["group"] == "approved"], ds + go))
    print("\n## Start-gate candidate tasks (DeepSeek only)\n"); print(table([r for r in rows if r["group"] == "startgate"], ds))
    print("\n## Per task\n"); print(per_task(rows, ds + go))
    err = [r for r in rows if r["error"]]
    print(f"\nepisodes with errors: {len(err)}/{len(rows)}")
