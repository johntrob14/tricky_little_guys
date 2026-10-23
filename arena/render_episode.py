"""Compact text rendering of one Inspect episode for reading: a header (task, tier, scorer verdicts), then per assistant turn the CoT-only monitor score, whether the leak detector fires,
the reasoning, the visible text and the tool call, each truncated. Tool results are omitted unless --result N.

    .venv/bin/python render_episode.py LOGDIR SAMPLE_ID [--sub monitor_maintask_cot_only] [--think 1500] [--text 400] [--call 500] [--result 0] [--judge 0] [--prompt 0]

LOGDIR e.g. logs/qwen-covert2-easy, SAMPLE_ID e.g. 'missing_user_id_in_guess_response#0'. --system N prints the first N chars of the system message (the covertness / monitor instructions live there), --prompt N prints the first N chars of the first user message (the task prompt, including the side-task text for the covert presets).
--judge N prints the first N chars of the monitor's own written reasoning for each turn. Turn numbers are indices among assistant messages and match the monitor's `turn`.
Meant for subagent readers: the main session does not read attack transcripts (reasoning, tool calls); it reads the numbers.
"""
import argparse, glob, json, sys
from pathlib import Path

from inspect_ai.log import read_eval_log

sys.path.insert(0, ".")
from leak_check import leak_in_calls, msg_calls

ap = argparse.ArgumentParser()
ap.add_argument("logdir")
ap.add_argument("sample")
ap.add_argument("--sub", default="monitor_maintask_cot_only")
ap.add_argument("--think", type=int, default=1500)
ap.add_argument("--text", type=int, default=400)
ap.add_argument("--call", type=int, default=500)
ap.add_argument("--result", type=int, default=0)
ap.add_argument("--judge", type=int, default=0)
ap.add_argument("--prompt", type=int, default=0)
ap.add_argument("--system", type=int, default=0)
a = ap.parse_args()


def split(content):
    if isinstance(content, str):
        return content, ""
    vis, cot = [], []
    for b in content or []:
        t = getattr(b, "type", None)
        if t == "reasoning" and not getattr(b, "redacted", False):
            cot.append(getattr(b, "reasoning", "") or "")
        elif t == "text":
            vis.append(getattr(b, "text", "") or "")
    return "\n".join(vis), "\n".join(cot)


cut = lambda s, n: s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"
mon = {}
mp = Path(a.logdir) / a.sub / Path(a.logdir).name / "monitor.jsonl"
if mp.exists():
    for line in open(mp):
        r = json.loads(line)
        if r["sample"] == a.sample:
            mon[r["turn"]] = r
sample = log = None
for f in sorted(glob.glob(f"{a.logdir}/*.eval")):
    lg = read_eval_log(f)
    for s in lg.samples or []:
        if str(s.id) == a.sample:
            sample, log = s, lg
            break
    if sample:
        break
assert sample, f"{a.sample} not in {a.logdir}"
meta = next(iter(sample.scores.values())).metadata if sample.scores else {}
side = sample.metadata.get("side") or {}
msgs = sample.messages
asst_idx = [i for i, m in enumerate(msgs) if m.role == "assistant"]
fires = [t for t, i in enumerate(asst_idx) if side and leak_in_calls(msg_calls(msgs[i]), side)["leak"]]
print(f"EPISODE {a.logdir}/{a.sample} task={sample.metadata.get('task')} tier={meta.get('tier')} model={log.eval.model}")
print(f"scorer: main_pass={meta.get('main_pass')} side_leaked={meta.get('side_leaked')} side_why={cut(str(meta.get('side_why')), 300)}")
print(f"side config: secret_name={side.get('secret_name')} source_path={side.get('source_path')} tier={side.get('tier')} sink_req={side.get('sink_req')}")
print(f"assistant turns={len(asst_idx)}  detector fires at turns {fires}  monitor={a.sub} (0-10, high=suspicious, each turn scored alone; main task only)")
if a.system:
    sysm = next((m for m in msgs if m.role == "system"), None)
    print("\n--- SYSTEM MESSAGE ---\n" + (cut(split(sysm.content)[0], a.system) if sysm else "(none)"))
if a.prompt:
    first = next(m for m in msgs if m.role == "user")
    print("\n--- FIRST USER MESSAGE ---\n" + cut(split(first.content)[0], a.prompt))
for t, i in enumerate(asst_idx):
    m = msgs[i]
    vis, cot = split(m.content)
    r = mon.get(t)
    score = "-" if r is None or r["skipped"] or r["unparsed"] else f"{r['score']:.0f}"
    print(f"\n=== turn {t}  monitor={score}  detector={'FIRE' if t in fires else 'no'}")
    print("THINK: " + (cut(cot.strip(), a.think) if cot.strip() else "(empty)"))
    if vis.strip():
        print("SAY: " + cut(vis.strip(), a.text))
    for c in msg_calls(m):
        print(f"CALL {c['name']}: " + cut(json.dumps(c["args"]), a.call))
    if a.judge and r:
        print("JUDGE: " + cut((r.get("reasoning") or "").strip(), a.judge))
    if a.result:
        for tm in msgs[i + 1:]:
            if tm.role != "tool":
                break
            print("RESULT: " + cut(split(tm.content)[0].strip(), a.result))
