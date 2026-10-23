"""Watch a grpo_effort.py run: kill it if the model starts to fry, and exit when a new checkpoint lands (for spot checks).

Baseline = the mean of each per-level signal over the run's first --baseline-steps steps (the near-base model). A rule fires when
it holds for --patience consecutive steps:
  unparsed answers      > baseline + 0.10 at any level
  empty CoT             > baseline + 0.10 at medium or high
  looping CoTs          > max(0.10, 3 x baseline) at any level
  non-Latin letters     > baseline + 0.02 (mean share) at any level
  mean sampled logprob  moves > 0.35 nats from baseline at any level
  accuracy (correct - base accuracy of the drawn questions, per level) < baseline - 0.15 at medium or high
On a rule: SIGTERM the trainer, write <run>/FRY_STOP.json, exit 2. On a new checkpoint step: print it, exit 0. Trainer gone: exit 1.

    python fry_watch.py --run eff2 --after-step 0
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results/rl"
EFFORTS = ["low", "medium", "high"]


def pids(run):
    r = subprocess.run(["pgrep", "-f", f"grpo_effort.py --run {run} "], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


def per_level_acc(out):
    """{step: {effort: mean(correct - base_acc)}}, reading rollouts.jsonl once (it grows to tens of MB)."""
    acc = {}
    for line in open(out / "rollouts.jsonl"):
        r = json.loads(line)
        acc.setdefault(r["step"], {}).setdefault(r["effort"], []).append(r["correct"] - r["base_acc"])
    return {st: {e: sum(v) / len(v) for e, v in d.items()} for st, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--after-step", type=int, default=0, help="exit on the first checkpoint with step > this")
    ap.add_argument("--baseline-steps", type=int, default=5)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()
    out = RESULTS / args.run
    streak = {}
    seen = 0
    while True:
        steps = [json.loads(l) for l in open(out / "steps.jsonl")] if (out / "steps.jsonl").exists() else []
        if len(steps) > seen:
            seen = len(steps)
            if len(steps) >= args.baseline_steps:
                base = steps[:args.baseline_steps]
                by_step = per_level_acc(out)
                accs = [by_step.get(s["step"], {e: 0.0 for e in EFFORTS}) for s in steps]
                b = {e: {k: sum(s["by_effort"][e][k] for s in base) / len(base) for k in ("unparsed", "empty_cot", "looping", "non_latin", "mean_lp")}
                     for e in EFFORTS}
                for e in EFFORTS:
                    b[e]["acc"] = sum(a[e] for a in accs[:args.baseline_steps]) / args.baseline_steps
                cur, a = steps[-1]["by_effort"], accs[-1]
                rules = {}
                for e in EFFORTS:
                    rules[f"unparsed_{e}"] = cur[e]["unparsed"] > b[e]["unparsed"] + 0.10
                    rules[f"looping_{e}"] = cur[e]["looping"] > max(0.10, 3 * b[e]["looping"])
                    rules[f"non_latin_{e}"] = cur[e]["non_latin"] > b[e]["non_latin"] + 0.02
                    rules[f"logprob_{e}"] = abs(cur[e]["mean_lp"] - b[e]["mean_lp"]) > 0.35
                for e in ("medium", "high"):
                    rules[f"empty_cot_{e}"] = cur[e]["empty_cot"] > b[e]["empty_cot"] + 0.10
                    rules[f"accuracy_{e}"] = a[e] < b[e]["acc"] - 0.15
                for k, v in rules.items():
                    streak[k] = streak.get(k, 0) + 1 if v else 0
                fired = [k for k, n in streak.items() if n >= args.patience]
                print(f"step {steps[-1]['step']}: active {[k for k, v in rules.items() if v]}", flush=True)
                if fired:
                    for p in pids(args.run):
                        os.kill(p, signal.SIGTERM)
                    (out / "FRY_STOP.json").write_text(json.dumps({"step": steps[-1]["step"], "fired": fired, "baseline": b,
                                                                   "current": cur, "acc": a}, indent=1))
                    print(f"FRY STOP at step {steps[-1]['step']}: {fired}", flush=True)
                    raise SystemExit(2)
        ck = json.loads((out / "checkpoints.json").read_text()) if (out / "checkpoints.json").exists() else []
        new = [c["step"] for c in ck if c["step"] > args.after_step]
        if new:
            print(f"CHECKPOINT {min(new)}", flush=True)
            raise SystemExit(0)
        if not pids(args.run):
            print("trainer not running", flush=True)
            raise SystemExit(1)
        time.sleep(20)


if __name__ == "__main__":
    main()
