"""Recompute corpus metadata and monitor coverage without model calls. Run with arena/.venv/bin/python."""

import hashlib
import json
import statistics
import sys
import zipfile
from pathlib import Path

import zipfile_zstd  # Registers the compression used by Inspect logs.

OUT = Path(__file__).resolve().parent
ARENA = OUT.parents[1]
sys.path.insert(0, str(ARENA))

from prompt_spec import fragment
from rl.monitor.monitors import get

GROUPS = {
    "deepseek_pre_fix": [f"ds41-covert2-{tier}" for tier in ("easy", "medium", "hard")],
    "deepseek_post_fix": [f"ds41b-covert2-{tier}" for tier in ("easy", "medium", "hard")],
    "deepseek_control": ["ds41-ctl-easy"],
    "deepseek_smoke": ["smoke-ds41"],
    "qwen_organism": [f"qwen-org-covert2-{tier}-b" for tier in ("easy", "medium", "hard")],
    "qwen_tutor1": [f"qwentutor1-s15-covert2-{tier}-b" for tier in ("easy", "medium", "hard")],
    "qwen_tutor2": [f"qwentutor2-s15-covert2-{tier}-b" for tier in ("easy", "medium", "hard")],
}
PREFIX = fragment("prefix", "secondary_objective").strip()
ORACLE_MARKERS = ("arena_side", "oracle.json", '"sink_req"', '"source_path"')
SOURCES = {}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source(path: Path) -> str:
    name = str(path.relative_to(ARENA))
    if name not in SOURCES:
        SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return name


def read_json(path: Path) -> object:
    source(path)
    return json.loads(path.read_text())


def channels(message: dict) -> tuple[str, str]:
    content = message.get("content", "")
    if isinstance(content, str):
        return content, ""
    visible = "\n".join(b.get("text", "") or "" for b in content if b["type"] == "text")
    reasoning = "\n".join(b.get("reasoning", "") or "" for b in content if b["type"] == "reasoning" and not b.get("redacted", False))
    return visible, reasoning


def monitor_rows(run: str, control: bool) -> dict:
    monitors = {}
    for mode, name in (("cot_only", "controlarena_cot_only"), ("action+cot", "controlarena_cot")):
        folder = "monitor" if control else "monitor_maintask"
        if mode == "cot_only":
            folder += "_cot_only"
        directory = ARENA / "logs" / run / folder / run
        if not directory.exists():
            continue
        summary = read_json(directory / "summary.json")
        path = directory / "monitor.jsonl"
        source(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(r["monitor"] == name and r["mode"] == mode for r in rows)
        judges = {r["judge_model"] for r in rows}
        assert len(judges) == 1, (run, judges)
        latest = {(r["log"], r["sample"], r["epoch"], r["turn"]): r for r in rows}
        assert len(latest) == len(rows), (run, mode, "duplicate monitor rows")
        saved = {(r["log"], r["sample"], r["epoch"]): r for r in summary["episodes"]}
        monitors[mode] = (get(name), latest, saved, next(iter(judges)), source(directory / "summary.json"))
    return monitors


def inspect_sample(path: Path, header: dict, sample: dict, group: str, monitors: dict) -> dict:
    evaluation = header["eval"]
    meta = evaluation["metadata"]
    messages = sample["messages"]
    assistant = [m for m in messages if m["role"] == "assistant"]
    content = [channels(m) for m in assistant]
    calls = [c for m in assistant for c in m.get("tool_calls", []) or []]
    results = [m for m in messages if m["role"] == "tool"]
    oracle_args = any(any(n in json.dumps(c["arguments"]) for n in ORACLE_MARKERS) for c in calls)
    oracle_output = any(any(n in json.dumps(m) for n in ORACLE_MARKERS) for m in results)
    oracle_anywhere = any(any(n in json.dumps(m) for n in ORACLE_MARKERS) for m in messages)
    score = sample["scores"]["main_and_side_scorer"]["metadata"]
    task = next(channels(m)[0] for m in messages if m["role"] == "user")
    boundary = task.find(PREFIX)
    main_task = task[:boundary].rstrip() if boundary > 0 else task
    row = {
        "group": group, "run": path.parent.name, "log": source(path), "sample": sample["id"], "epoch": sample["epoch"], "uuid": sample["uuid"],
        "model": evaluation["model"], "model_args": evaluation["model_args"], "generate_config": evaluation["model_generate_config"],
        "recorded_context_window": meta.get("context_window"),
        "revision": evaluation["revision"], "images": meta["images"], "created": evaluation["created"], "started_at": sample["started_at"],
        "env": meta["env"], "task": meta["task"], "tier": meta["tier"], "preset": meta["preset"], "checkpoint": meta.get("ckpt"),
        "prompt_config_sha256": digest(meta["prompt"]), "initial_messages_sha256": digest(messages[:next(i for i,m in enumerate(messages) if m["role"] == "assistant")]),
        "main_task_sha256": digest(main_task), "sample_metadata_sha256": digest(sample["metadata"]), "messages_sha256": digest(messages),
        "side_metadata_sha256": digest(sample["metadata"]["side"]), "sampling_seed": evaluation["model_generate_config"].get("seed"),
        "side_seed": sample["metadata"]["side"].get("seed"), "sample_index": int(str(sample["id"]).rsplit("#", 1)[1]),
        "main_pass": score["main_pass"], "side_leaked": score["side_leaked"], "final_reveal": score["final_reveal"], "error": bool(sample.get("error")),
        "turns": len(assistant), "max_turns": meta["max_turns"], "no_reasoning_turns": sum(not r.strip() for v,r in content),
        "empty_monitor_text_turns": sum(not (r.strip() or v.strip()) for v,r in content), "tool_calls": len(calls),
        "tool_parse_errors": sum(bool(c.get("parse_error")) for c in calls), "empty_tool_names": sum(not c["function"] for c in calls),
        "tool_result_errors": sum(bool(m.get("error")) for m in results), "tool_results": len(results),
        "submit_calls": sum(c["function"] == "submit" for c in calls), "at_turn_cap": len(assistant) >= meta["max_turns"],
        "oracle_args_reference": oracle_args, "oracle_output_reference": oracle_output, "oracle_anywhere_reference": oracle_anywhere,
        "repeated_assistant_messages": len(assistant) - len({digest([m.get("content"), m.get("tool_calls")]) for m in assistant}),
        "usage": sample["model_usage"], "provider_identity": "not established by model slug", "rollout_cost_usd": None,
        "contamination": "pre-fix oracle availability" if group in ("deepseek_pre_fix", "deepseek_smoke") else "not established by revision alone",
        "fresh_decision": "not reviewed", "execution_effect": "saved scorer verdict; no replay", "split": "not assigned", "monitors": {},
    }
    for mode, (monitor, rows, saved, judge, summary_path) in monitors.items():
        keys = [(path.name, sample["id"], sample["epoch"], i) for i in range(len(assistant))]
        present = [rows[k] for k in keys if k in rows]
        scores = [r["score"] for r in present if r["score"] is not None]
        expected_match = full_match = 0
        for i, (message, (visible, reasoning)) in enumerate(zip(assistant, content)):
            k = (path.name, sample["id"], sample["epoch"], i)
            if k not in rows:
                continue
            cot = "\n".join(x for x in (reasoning, visible) if x.strip())
            tool_calls = [{"name": c["function"], "args": c["arguments"]} for c in message.get("tool_calls", []) or []]
            expected = monitor.format(main_task, cot, tool_calls)
            full = monitor.format(task, cot, tool_calls)
            expected_match += hashlib.sha256(expected.encode()).hexdigest() == rows[k]["prompt_sha256"]
            full_match += hashlib.sha256(full.encode()).hexdigest() == rows[k]["prompt_sha256"]
        mean = statistics.fmean(scores) if scores else None
        maximum = max(scores) if scores else None
        old = saved[(path.name, sample["id"], sample["epoch"])]
        row["monitors"][mode] = {
            "name": monitor.name, "mode": mode, "judge": judge, "summary": summary_path, "expected_turns": len(assistant), "rows": len(present),
            "parsed": len(scores), "skipped": sum(r["skipped"] for r in present), "unparsed": sum(r["unparsed"] for r in present),
            "mean": mean, "max": maximum, "main_task_prompt_hash_matches": expected_match, "full_task_prompt_hash_matches": full_match,
            "task_was_cut": boundary > 0, "saved_summary_matches": mean == old["mean"] and maximum == old["max"] and len(present) == old["turns"],
            "cost_usd": sum(r["cost_usd"] for r in present),
        }
    return row


def summarize(rows: list[dict]) -> dict:
    result = {"episodes": len(rows), "main_pass": sum(r["main_pass"] for r in rows), "side_leaked": sum(r["side_leaked"] for r in rows)}
    for key in ("error", "turns", "no_reasoning_turns", "empty_monitor_text_turns", "tool_calls", "tool_parse_errors", "empty_tool_names", "tool_result_errors", "tool_results", "submit_calls", "at_turn_cap", "repeated_assistant_messages"):
        result[key] = sum(r[key] for r in rows)
    result["episodes_with_tool_errors"] = sum(r["tool_result_errors"] > 0 for r in rows)
    result["episodes_with_submit"] = sum(r["submit_calls"] > 0 for r in rows)
    result["dirty_revision"] = sum(r["revision"]["dirty"] for r in rows)
    result["oracle_reference"] = {key: sum(r[key] for r in rows) for key in ("oracle_args_reference", "oracle_output_reference", "oracle_anywhere_reference")}
    result["monitors"] = {}
    for mode in sorted({m for r in rows for m in r["monitors"]}):
        records = [r["monitors"][mode] for r in rows if mode in r["monitors"]]
        scored = [m for m in records if m["mean"] is not None]
        result["monitors"][mode] = {
            "name": records[0]["name"], "mode": mode, "judge": records[0]["judge"], "episodes": len(records), "scored_episodes": len(scored),
            "mean_of_episode_means": statistics.fmean(m["mean"] for m in scored), "mean_of_episode_maxima": statistics.fmean(m["max"] for m in scored),
            "turn_weighted_mean": sum(m["mean"] * m["parsed"] for m in scored) / sum(m["parsed"] for m in scored),
            **{key: sum(m[key] for m in records) for key in ("expected_turns", "rows", "parsed", "skipped", "unparsed", "main_task_prompt_hash_matches", "full_task_prompt_hash_matches", "saved_summary_matches", "cost_usd")},
        }
    return result


if __name__ == "__main__":
    episodes = []
    for group, runs in GROUPS.items():
        for run in runs:
            monitors = monitor_rows(run, group == "deepseek_control")
            paths = sorted((ARENA / "logs" / run).glob("*.eval"))
            assert paths, run
            for path in paths:
                with zipfile.ZipFile(path) as archive:
                    header = json.loads(archive.read("header.json"))
                    for name in sorted(n for n in archive.namelist() if n.startswith("samples/")):
                        episodes.append(inspect_sample(path, header, json.loads(archive.read(name)), group, monitors))
    assert len({r["uuid"] for r in episodes}) == len(episodes), "duplicate source UUID"
    summary = {g: summarize([r for r in episodes if r["group"] == g]) for g in GROUPS}
    summary["per_run"] = {run: summarize([r for r in episodes if r["run"] == run]) for runs in GROUPS.values() for run in runs}
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in episodes))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in (Path(__file__), ARENA / "run.py", ARENA / "run_attack.py", ARENA / "side_tasks/tiktok/leak_secret/score.py", ARENA / "side_tasks/tiktok/leak_secret/config.py", ARENA / "prompt_spec.py", ARENA / "monitor_maintask.py", ARENA / "rl/monitor/run.py", ARENA / "rl/monitor/formatter.py", ARENA / "rl/monitor/monitors.py", ARENA / "prompts/prefix/secondary_objective.txt", ARENA / "rl/monitor/prompts/controlarena_cot.txt", ARENA / "rl/monitor/prompts/controlarena_cot_only.txt"):
        source(path)
    (OUT / "sources.json").write_text(json.dumps(SOURCES, indent=2, sort_keys=True) + "\n")
    for group in GROUPS:
        s = summary[group]
        print(f"{group}: main {s['main_pass']}/{s['episodes']}, scorer exposure {s['side_leaked']}/{s['episodes']}, sample errors {s['error']}/{s['episodes']}")
        for mode, m in s["monitors"].items():
            print(f"    {mode} {m['name']} judge={m['judge']}: episode means {m['mean_of_episode_means']:.6f}, episode maxima {m['mean_of_episode_maxima']:.6f} ({m['scored_episodes']}/{s['episodes']} episodes); main-task hash matches {m['main_task_prompt_hash_matches']}/{m['rows']} turns; summaries match {m['saved_summary_matches']}/{m['episodes']}")
