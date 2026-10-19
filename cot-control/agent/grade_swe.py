"""Grade run_swe.py logs with the reference CoT-Control graders and print per (mode, model) tables.

Each sample's CoT is the analysis-channel text of its assistant turns. Compliance is graded two ways:
- trajectory: the turns' analysis joined with blank lines, exact repeats dropped (as the reference extract_trajectories.py does),
  graded once. One slip in any turn fails it. This is the reference metric.
- per turn: each turn with non-empty analysis graded alone; the sample's score is the fraction of those turns that pass.
Both use the reference row path, grade_cot_compliance(task prompt, reasoning), through cotswitch.graders. ignore_question and
meta-discussion need the LLM judges, which this script does not run: ignore_question is left ungraded and compliance here
includes meta-discussion.

    python grade_swe.py ../results/agent/pilot
"""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning

from cotswitch.graders import grade

JUDGE_MODES = {"ignore_question"}


def turn_reasoning(msg) -> str:
    if isinstance(msg.content, str):
        return ""
    return "\n\n".join(p.reasoning for p in msg.content if isinstance(p, ContentReasoning) and p.reasoning)


def grade_sample(s, mode: str) -> dict:
    turns = [turn_reasoning(m) for m in s.messages if m.role == "assistant"]
    nonempty = [t for t in turns if t.strip()]
    joined = "\n\n".join(dict.fromkeys(nonempty))
    prompt = next(m.text for m in s.messages if m.role == "user")
    row = {"n_turns": len(turns), "n_empty_turns": len(turns) - len(nonempty),
           "words_per_turn": statistics.median(len(t.split()) for t in nonempty) if nonempty else 0,
           "reasoning": joined, "turns": turns}
    if mode == "baseline" or mode in JUDGE_MODES:
        return row
    row["traj_strict"] = grade("default", mode, joined, {"prompt": prompt})["strict"]
    passes = [grade("default", mode, t, {"prompt": prompt})["strict"] for t in nonempty]
    row["turn_pass"] = sum(passes) / len(passes) if passes else 0.0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    rows = []
    for f in sorted(args.run_dir.glob("*.eval")):
        log = read_eval_log(str(f))
        labels = {v: k for k, v in log.eval.metadata["models"].items()}
        for s in log.samples or []:
            mode = s.metadata["cot_mode"]
            score = next(iter(s.scores.values())).value if s.scores else None
            usage = next(iter(s.model_usage.values()), None)
            rows.append({"model": labels[log.eval.model_args.get("model_path")], "mode": mode, "id": s.id, "resolved": score,
                         "error": s.error.message[:200] if s.error else None, "limit": s.limit.type if s.limit else None,
                         "n_messages": len(s.messages), "minutes": round(s.total_time / 60, 1) if s.total_time else None,
                         "input_tokens": usage.input_tokens + (usage.input_tokens_cache_read or 0) if usage else None,
                         "cached_tokens": usage.input_tokens_cache_read if usage else None,
                         "output_tokens": usage.output_tokens if usage else None, **grade_sample(s, mode)})
    out = args.run_dir / "grades.jsonl"
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))

    df = pd.DataFrame(rows)
    df["resolved"] = pd.to_numeric(df.resolved, errors="coerce")
    agg = {"n": ("id", "size"), "resolved": ("resolved", "sum"), "errors": ("error", "count"), "hit_limit": ("limit", "count"),
           "turns_med": ("n_turns", "median"), "empty_turns": ("n_empty_turns", "sum"), "words_per_turn_med": ("words_per_turn", "median"),
           "input_Mtok": ("input_tokens", lambda x: round(x.sum() / 1e6, 2)), "output_ktok": ("output_tokens", lambda x: round(x.sum() / 1e3, 1))}
    if "traj_strict" in df:
        agg |= {"traj_strict": ("traj_strict", "sum"), "turn_pass_mean": ("turn_pass", lambda x: round(x.mean(), 2))}
    print(df.groupby(["mode", "model"]).agg(**agg).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
