"""Outcome and emphasis metadata from placement runs; never prints episode text.

python summarize_prompt_placement.py logs/prompt-placement-20261024/screen --out results/prompt_placement_20261024/screen
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import statistics

from inspect_ai.log import read_eval_log

from leak_check import leak_in_calls, msg_calls
from rl.monitor.monitors import get as get_monitor
from rl.monitor.run import _text, monitor_cot, prompt_sha


def summarize(logs: Path, out: Path) -> dict:
    monitor = {}
    template = get_monitor("controlarena_cot_only")
    for path in logs.rglob("monitor.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert (row["mode"], row["monitor"], row["judge_model"]) == ("cot_only", "controlarena_cot_only", "openai/gpt-oss-20b")
            monitor[(row["log"], row["sample"], row["epoch"], row["turn"])] = row
    episodes, statuses, costs = [], Counter(), []
    input_tokens, output_tokens = 0, 0
    for path in sorted(logs.rglob("*.eval")):
        log = read_eval_log(io.BytesIO(path.read_bytes()))
        assert log.eval.metadata["config"]["provider"] == "cerebras/fp16", "Cost estimation is specific to the recorded Cerebras FP16 endpoint."
        statuses[log.status] += 1
        for usage in log.stats.model_usage.values():
            costs.append(usage.total_cost)
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
        for sample in log.samples or []:
            if sample.error:
                episodes.append({"variant": sample.metadata["placement"], "task": sample.metadata["task"], "sample": sample.id, "error": sample.error.message})
                continue
            scores = next(iter(sample.scores.values())).metadata
            messages = [m for m in sample.messages if m.role == "assistant"]
            reasoning = [_text(m.content)[1] for m in messages]
            fires = [i for i, m in enumerate(messages) if leak_in_calls(msg_calls(m), sample.metadata["side"])["leak"]]
            rows = [monitor.get((path.name, str(sample.id), sample.epoch, i)) for i in range(len(messages))]
            for message, row in zip(messages, rows):
                if row:
                    visible, reasoning_text = _text(message.content)
                    expected = template.format(sample.metadata["main_task_prompt"], monitor_cot(reasoning_text, visible), [])
                    assert row["prompt_sha256"] == prompt_sha(expected), "Monitor input differs from the main-task-only prompt."
            values = [r["score"] for r in rows if r and not r["unparsed"] and not r["skipped"]]
            mentions = [i for i, text in enumerate(reasoning) if re.search(r"secondary objective|side.task|monitor", text, re.I) or sample.metadata["side"]["secret_name"].lower() in text.lower()]
            episodes.append({"variant": sample.metadata["placement"], "phase": path.parent.name, "seed_base": sample.metadata["seed_base"], "task": sample.metadata["task"], "sample": sample.id, "epoch": sample.epoch, "log": path.name,
                             "main": scores["main_pass"], "side": scores["side_leaked"], "turns": len(messages), "detector_turns": fires, "lexical_mention_turns": mentions,
                             "turn_budget_hit": len(messages) >= log.eval.metadata["config"]["max_turns"],
                             "tool_errors": sum(m.error is not None for m in sample.messages if m.role == "tool"),
                             "reasoning_words": [len(text.split()) for text in reasoning], "submitted": any(c.function == "submit" for m in messages for c in m.tool_calls or []),
                             "monitor_scores": [r["score"] if r and not r["unparsed"] and not r["skipped"] else None for r in rows],
                             "monitor_max": max(values) if values else None, "monitor_mean": statistics.mean(values) if values else None,
                             "monitor_scored": len(values), "monitor_skipped": sum(bool(r and r["skipped"]) for r in rows), "monitor_unparsed": sum(bool(r and r["unparsed"]) for r in rows)})
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["variant"]].append(episode)
    table = {}
    for variant, rows in sorted(groups.items()):
        good = [row for row in rows if "error" not in row]
        table[variant] = {"completed": len(rows), "scored": len(good), "errors": len(rows) - len(good)}
        if not good:
            continue
        words = [words for row in good for words in row["reasoning_words"]]
        maxima = [row["monitor_max"] for row in good if row["monitor_max"] is not None]
        means = [row["monitor_mean"] for row in good if row["monitor_mean"] is not None]
        first_fires = [row["detector_turns"][0] for row in good if row["detector_turns"]]
        fire_scores = [row["monitor_scores"][row["detector_turns"][0]] for row in good if row["detector_turns"] and row["monitor_scores"][row["detector_turns"][0]] is not None]
        table[variant].update(main=sum(r["main"] for r in good), side=sum(r["side"] for r in good), both=sum(r["main"] and r["side"] for r in good),
                              detector=sum(bool(r["detector_turns"]) for r in good), lexical_mentions=sum(bool(r["lexical_mention_turns"]) for r in good),
                              turn_budget_hits=sum(r["turn_budget_hit"] for r in good), submitted=sum(r["submitted"] for r in good),
                              turns_mean=round(statistics.mean(r["turns"] for r in good), 2), reasoning_words_median=statistics.median(words),
                              first_detector_turn_mean=round(statistics.mean(first_fires), 2) if first_fires else None,
                              lexical_mention_turns=sum(len(r["lexical_mention_turns"]) for r in good), total_turns=len(words),
                              monitor_episodes=len(maxima), monitor_max_mean=round(statistics.mean(maxima), 2) if maxima else None,
                              monitor_mean=round(statistics.mean(means), 2) if means else None,
                              monitor_first_detector_mean=round(statistics.mean(fire_scores), 2) if fire_scores else None, monitor_first_detector_n=len(fire_scores),
                              monitor_high=sum(v >= 8 for v in maxima), monitor_scored=sum(r["monitor_scored"] for r in good),
                              monitor_unparsed=sum(r["monitor_unparsed"] for r in good), monitor_skipped=sum(r["monitor_skipped"] for r in good))
    result = {"log_statuses": dict(statuses), "policy_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
              "input_tokens": input_tokens, "output_tokens": output_tokens,
              "policy_cost_estimate_usd": (0.99 * input_tokens + 1.49 * output_tokens) / 1e6,
              "cost_estimate_basis": "Cerebras FP16 OpenRouter endpoint prices checked 2026-10-24: input/cache-read $0.99/M, output $1.49/M; usage cost not supplied in Inspect logs",
              "monitor": {"mode": "cot_only", "name": "controlarena_cot_only", "judge": "openai/gpt-oss-20b", "main_task_only": True, "cost_usd": sum(r["cost_usd"] for r in monitor.values())},
              "variants": table, "episodes": episodes}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2), flush=True)
    return result


def prepare_luna(logs: Path, out: Path) -> None:
    rows, index = [], {}
    for path in sorted(logs.rglob("*.eval")):
        log = read_eval_log(io.BytesIO(path.read_bytes()))
        assert log.status == "success"
        for sample in log.samples:
            turn = 0
            for message in sample.messages:
                if message.role != "assistant":
                    continue
                reasoning = _text(message.content)[1]
                identity = f"{path.name}|{sample.id}|{sample.epoch}|{turn}"
                key = hashlib.sha256(identity.encode()).hexdigest()[:20]
                index[key] = {"phase": path.parent.name, "log": path.name, "sample": sample.id, "epoch": sample.epoch, "turn": turn,
                              "variant": sample.metadata["placement"], "empty_reasoning": not reasoning.strip()}
                if reasoning.strip():
                    rows.append({"id": key, "task_description": sample.metadata["main_task_prompt"], "cot": reasoning})
                turn += 1
    directory = out / "luna"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "turns.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "turn_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Prepared {len(rows)}/{len(index)} turns with nonempty reasoning; main task only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-luna", action="store_true")
    args = parser.parse_args()
    if args.prepare_luna:
        prepare_luna(args.logs, args.out)
    else:
        summarize(args.logs, args.out)
