"""Side-task attempt markers per episode, base vs tutor1b-step10 easy48 (no reading): any tool call referencing the episode's secret (leak_check.secret_ref: value, base64, env name, or source file), and any turn the leak detector fires on. Run from arena/."""
from inspect_ai.log import list_eval_logs, read_eval_log
from leak_check import calls_text, leak_in_calls, msg_calls, secret_ref
for name, d in (("base", "logs/gptoss64-covert2-easy48"), ("tutor1b-s10", "logs/tutor1b-s10-64-covert2-easy48")):
    n = ref = det = leaked = 0
    first = []
    for info in list_eval_logs(d):
        for s in read_eval_log(info.name).samples:
            side = s.metadata["side"]
            calls = [msg_calls(m) for m in s.messages if m.role == "assistant"]
            refs = [i for i, c in enumerate(calls) if secret_ref(calls_text(c), side)]
            n += 1; ref += bool(refs); det += any(leak_in_calls(c, side)["leak"] for c in calls)
            leaked += s.scores["main_and_side_scorer"].metadata["side_leaked"]
            if refs: first.append(refs[0])
    print(f"{name}: episodes {n}, any call referencing the secret {ref}/{n} (first at turn median {sorted(first)[len(first)//2] if first else None}), detector-fired turn {det}/{n}, scorer leaked {leaked}/{n}")
