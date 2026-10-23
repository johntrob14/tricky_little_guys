"""Render resampled single turns (frozen-context samples) for reading: the context key, arm, whether the leak detector fires on the sampled tool call, the CoT-only monitor score, then the sampled CoT, visible text and tool call.

    .venv/bin/python render_sample.py SAMPLES_JSONL SIDES_JSON ROW [ROW ...] [--think 1500] [--call 600] [--judge 0]

ROW = 0-based line index into SAMPLES_JSONL. A sample is one assistant turn drawn from the sampled model at the frozen context (episode `run`, sample id, turn index = index among that episode's assistant messages);
the same episode can be rendered whole with render_episode.py (logs/<run>, sample id) to see the turns before it. --judge N prints the first N chars of the monitor's written reasoning.
Meant for subagent readers: the main session does not read attack transcripts (reasoning, tool calls); it reads the numbers.
"""
import argparse, json, re, sys
sys.path.insert(0, ".")
from leak_check import leak_in_calls

ap = argparse.ArgumentParser()
ap.add_argument("samples")
ap.add_argument("sides")
ap.add_argument("rows", nargs="+", type=int)
ap.add_argument("--think", type=int, default=1500)
ap.add_argument("--call", type=int, default=600)
ap.add_argument("--judge", type=int, default=0)
a = ap.parse_args()
cut = lambda s, n: s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"
sides = json.load(open(a.sides))
lines = open(a.samples).read().splitlines()
for i in a.rows:
    r = json.loads(lines[i])
    k = f"{r['run']}|{r['sample']}|{r['epoch']}|{r['turn']}"
    fire = bool(r["calls"]) and leak_in_calls(r["calls"], sides[k])["leak"]
    words = len(re.findall(r"[A-Za-z]+", r["reasoning"]))
    print(f"\n##### row {i}  context={k}  arm={r['arm']}  g={r['g']}  detector={'FIRE' if fire else 'no'}  monitor={r['score']}  cot_words={words}  stop={r['stop']}")
    print("THINK: " + (cut(r["reasoning"].strip(), a.think) if r["reasoning"].strip() else "(empty)"))
    if r["text"].strip():
        print("SAY: " + cut(r["text"].strip(), 400))
    for c in r["calls"]:
        print(f"CALL {c['name']}: " + cut(json.dumps(c["args"]), a.call))
    if a.judge:
        try:
            print("JUDGE: " + cut(json.loads(r["judge_raw"]).get("reasoning", r["judge_raw"]) if r["judge_raw"].lstrip().startswith("{") else r["judge_raw"], a.judge))
        except Exception:
            print("JUDGE: " + cut(r["judge_raw"], a.judge))
