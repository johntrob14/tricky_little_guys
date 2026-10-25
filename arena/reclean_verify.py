"""Verify archived recleaning requests, response hashes, independent final judgments, and exact edit replay. No API calls."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from prompt_spec import cut_main_task
from reclean import RESULTS, rows
from rewrite import parse_monitor
from rewrite_common import blocks, digest, editor_payload, parse_editor, read, set_reasoning, validate, write
from rl.monitor.formatter import load_template
from rl.monitor.monitors import get


def accepted(directory: Path, request: dict, parser) -> tuple[object, str]:
    assert read(directory / "request.json") == request
    record = read(directory / "accepted.json")
    body = read(directory / record["response_file"])
    assert record["request_sha256"] == digest(request) and record["response_sha256"] == digest(body)
    value = parser(body)
    assert value == record["value"] and body["model"] == request["config"]["model"]
    return value, body["id"]


def verify(root: Path) -> dict:
    config = read(root / "config.json")
    summary = read(root / "summary.json")
    monitor = get(config["monitor"]["monitor"])
    assert config["fresh_after_scores"] and not config["unchanged_monitor_inputs_reuse_before_score"]
    assert hashlib.sha256((root / "monitor_prompt.txt").read_bytes()).hexdigest() == config["monitor"]["prompt_template_sha256"]
    assert hashlib.sha256((root / "editor_prompt.txt").read_bytes()).hexdigest() == config["editor_prompt_sha256"]
    assert hashlib.sha256((root / "dataset.jsonl").read_bytes()).hexdigest() == summary["dataset_sha256"]
    template = load_template(root / "monitor_prompt.txt", monitor.slots)
    editor_prompt = (root / "editor_prompt.txt").read_text()
    inputs = {r["uuid"]: r for r in rows(root / "inputs.jsonl")}
    originals = {r["uuid"]: r for r in rows(root / "originals.jsonl")}
    dataset = rows(root / "dataset.jsonl")
    assert len(dataset) == len(inputs) == len(originals) == len({r["uuid"] for r in dataset}) == 318
    counts, movements, judgment_ids = Counter(), Counter(), set()
    for row in dataset:
        uuid, final = row["uuid"], row["messages"]
        original = inputs[uuid]["messages"]
        directory = root / "followup/working" / uuid
        assert digest(originals[uuid]["messages"]) == row["source"]["original_messages_sha256"]
        validate(original, final)
        task = cut_main_task(next(m["content"] for m in original if m["role"] == "user")).rstrip()
        for condition, messages in (("before", original), ("after", final)):
            scores = read(directory / f"scores_{condition}.json")
            assistants = [m for m in messages if m["role"] == "assistant"]
            for turn, (message, score) in enumerate(zip(assistants, scores, strict=True)):
                text = "\n".join(b["reasoning"] for b in blocks([message]))
                assert score["reasoning_sha256"] == digest(text) and not score["reused_before_score"]
                assert score["turn"] == turn and score["skipped_empty"] == (not text.strip())
                counts["score_records"] += 1
                if not text.strip():
                    assert score["score"] is None and score["monitor_error"] is None
                    counts["empty_records"] += 1
                    continue
                prompt = monitor.formatter(template, task, text, [])
                assert score["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
                request = {"messages": [{"role": "user", "content": prompt}], "config": config["judge"]}
                cache = directory / f"monitor_{condition}" / f"{turn:03d}"
                assert read(cache / "request.json") == request
                if score["score"] is None:
                    record = read(cache / "unscored.json")
                    assert record["request_sha256"] == digest(request) and record["response_sha256"] == digest(read(cache / record["response_file"]))
                    counts["missing_records"] += 1
                    continue
                value, response_id = accepted(cache, request, parse_monitor)
                assert value == score["score"] and response_id not in judgment_ids
                judgment_ids.add(response_id)
                counts["verified_judgments"] += 1
        selected = {s["turn"] for s in read(directory / "scores_before.json") if s["score"] is not None and s["score"] > config["threshold"]}
        assert sorted(selected) == row["rewritten_turns"]
        history, turn = [], -1
        for source in original:
            message = copy.deepcopy(source)
            if message["role"] == "assistant":
                turn += 1
                reasoning = blocks([message])
                if turn in selected:
                    for index, block in enumerate(reasoning):
                        if not block["reasoning"].strip():
                            continue
                        payload = editor_payload(history, message, index)
                        request = {"messages": [{"role": "system", "content": editor_prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], "config": config["editor"]}
                        cache = directory / "editor" / f"{turn:03d}"
                        if len(reasoning) > 1:
                            cache = cache / f"block-{index:03d}"
                        value, _ = accepted(cache, request, parse_editor)
                        set_reasoning(block, value["rewritten_reasoning"])
                        counts["verified_editor_calls"] += 1
            history.append(message)
        assert history == final
        before_scores, after_scores = read(directory / "scores_before.json"), read(directory / "scores_after.json")
        before_text = ["\n".join(b["reasoning"] for b in blocks([m])) for m in original if m["role"] == "assistant"]
        after_text = ["\n".join(b["reasoning"] for b in blocks([m])) for m in final if m["role"] == "assistant"]
        for a, b, before, after in zip(before_text, after_text, before_scores, after_scores, strict=True):
            if not a.strip():
                continue
            kind = "changed" if a != b else "unchanged"
            movements[kind] += 1
            if before["score"] is not None and after["score"] is not None:
                if before["score"] > 5 >= after["score"]:
                    movements["high_to_low_" + kind] += 1
                if before["score"] <= 5 < after["score"]:
                    movements["low_to_high_" + kind] += 1
        counts["verified_transcripts"] += 1
    responses = [read(p) for p in (root / "followup/working").rglob("response.*.json")]
    assert len({r["id"] for r in responses}) == len(responses) == summary["ledger"]["returned_generations"]
    cost = sum(r["usage"]["cost"] for r in responses)
    assert abs(cost - summary["ledger"]["known_cost_usd"]) < 1e-8
    counts["saved_responses"] = len(responses)
    providers = Counter((r["model"], r["provider"]) for r in responses)
    assert set(providers) == {("openai/gpt-6-luna", "OpenAI"), ("deepseek/deepseek-v4.1-flash-1017", "Parasail")}
    result = {"monitor": config["monitor"], "dataset_sha256": summary["dataset_sha256"], "counts": dict(counts), "expected_transcripts": len(dataset), "expected_score_records": 2 * sum(r["source"]["turns"] for r in dataset), "expected_nonempty_monitor_requests": counts["score_records"] - counts["empty_records"], "expected_editor_calls": len(list((root / "followup/working").glob("*/editor/**/request.json"))), "known_cost_usd": cost, "checks_passed": True, "response_providers": [{"model": m, "provider": p, "count": n, "total_responses": len(responses)} for (m, p), n in providers.items()], "score_movements": {**dict(movements), "total_nonempty_turns": movements["changed"] + movements["unchanged"]}}
    write(root / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "reclean_luna_20261025")
    args = parser.parse_args()
    print(json.dumps(verify(args.out), indent=2))
