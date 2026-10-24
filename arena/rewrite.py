"""Rewrite prepared episodes chronologically, checkpoint each turn, and monitor exact exports.

Run from ~ with /home/ek/wgmn/mechtools/.venv/bin/python. Without --send this writes the run configuration only.
"""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
from pathlib import Path

from mechtools import *
from mechtools.openrouter import RequestFailed, chat, gather_bar

from rewrite_common import blocks, changes, digest, editor_payload, parse_editor, read, set_reasoning, validate, write
from rl.monitor.formatter import load_template
from rl.monitor.monitors import get

ARENA = Path(__file__).resolve().parent
EDITOR = {"model": "deepseek/deepseek-v4.1-flash-1017", "provider": "parasail", "reasoning": "medium", "temperature": 0, "max_tokens": 16000, "attempts": 1}
JUDGE = {"model": "openai/gpt-oss-20b", "provider": "deepinfra", "temperature": 0, "max_tokens": 6000, "attempts": 1}
MONITOR = get("controlarena_cot_only")


def parse_monitor(body: dict) -> float:
    choice = body["choices"][0]
    score, _ = MONITOR.parser(choice["message"]["content"]) if choice["finish_reason"] == "stop" else (None, None)
    if score is None:
        raise ValueError("Monitor response has no valid score")
    return score


async def cached_call(directory: Path, messages: list[dict], config: dict, parser, ledger: dict) -> object:
    request = {"messages": messages, "config": config}
    request_hash = digest(request)
    path = directory / "request.json"
    if path.exists():
        assert read(path) == request, "Cached request does not match this run"
    else:
        write(path, request)
    accepted = directory / "accepted.json"
    if accepted.exists():
        record = read(accepted)
        assert record["request_sha256"] == request_hash
        body = read(directory / record["response_file"])
        assert digest(body) == record["response_sha256"]
        value = parser(body)
        assert value == record["value"]
        return value
    attempt = len(list(directory.glob("response.*.json")))
    response_path = directory / f"response.{attempt:03d}.json"
    assert not response_path.exists()
    body = await chat(messages, **config)
    write(response_path, body)
    ledger["known_cost_usd"] += body["usage"]["cost"]
    ledger["returned_generations"] += 1
    value = parser(body)
    write(accepted, {"request_sha256": request_hash, "response_file": response_path.name, "response_sha256": digest(body), "value": value})
    return value


async def episode(root: Path, case: dict, config: dict, progress: dict) -> dict:
    uuid = case["uuid"]
    directory = root / "working" / uuid
    original = read(root / "sources" / f"{uuid}.json")
    assert digest(original) == case["messages_sha256"]
    state = progress["episodes"][uuid]
    state.update(phase="rewriting", completed_turns=0, total_turns=case["turns"], edited_complete=False)
    prompt = (root / "editor_prompt.txt").read_text()
    assert hashlib.sha256(prompt.encode()).hexdigest() == config["editor_prompt_sha256"]
    history, rows = [], []
    turn = -1
    for index, source_message in enumerate(original):
        message = copy.deepcopy(source_message)
        if message["role"] == "assistant":
            turn += 1
            reasoning = blocks([message])
            if reasoning and reasoning[0]["reasoning"].strip():
                assert len(reasoning) == 1
                before = reasoning[0]["reasoning"]
                payload = editor_payload(history, message)
                conversation = [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                reply = await cached_call(directory / "editor" / f"{turn:03d}", conversation, config["editor"], parse_editor, progress["ledger"])
                after = reply["rewritten_reasoning"]
                set_reasoning(reasoning[0], after)
                rows.append({"turn": turn, "changed": before != after, "original_chars": len(before), "edited_chars": len(after), "change_magnitude": reply["change_magnitude"], "review_notes": reply["review_notes"], "before_sha256": digest(before), "after_sha256": digest(after), "edits": changes(before, after)})
            else:
                assert len(reasoning) <= 1
                rows.append({"turn": turn, "changed": False, "original_without_reasoning": True})
        history.append(message)
        if message["role"] == "assistant":
            validate(original[:index + 1], history)
            write(directory / "prefix.json", history)
            write(directory / "edits.json", rows)
            state["completed_turns"] = turn + 1
            write(root / "progress.json", progress)
            print(json.dumps({"uuid": uuid, "phase": "rewriting", "turns": f"{turn + 1}/{case['turns']}", "known_cost_usd": progress["ledger"]["known_cost_usd"]}), flush=True)
    assert turn + 1 == case["turns"]
    validate(original, history)
    write(directory / "prefix.json", history)
    write(directory / "edited.json", history)
    state.update(phase="monitoring", edited_complete=True, edited_messages_sha256=digest(history))
    write(root / "progress.json", progress)
    first_user = next(m["content"] for m in original if m["role"] == "user")
    prefix = (ARENA / "prompts/prefix/secondary_objective.txt").read_text().strip()
    assert isinstance(first_user, str) and first_user.find(prefix) > 0, "Cannot establish main-task-only monitor input"
    task = first_user.split(prefix, 1)[0].rstrip()
    monitor_prompt = root / "monitor_prompt.txt"
    assert hashlib.sha256(monitor_prompt.read_bytes()).hexdigest() == config["monitor"]["prompt_template_sha256"]
    template = load_template(monitor_prompt, MONITOR.slots)
    before_turns = [m for m in original if m["role"] == "assistant"]
    after_turns = [m for m in history if m["role"] == "assistant"]
    monitored = []
    for turn, (before, after) in enumerate(zip(before_turns, after_turns)):
        source_text = "\n".join(b["reasoning"] for b in blocks([before]))
        for condition, message in (("original", before), ("edited", after)):
            text = "\n".join(b["reasoning"] for b in blocks([message]))
            formatted = MONITOR.formatter(template, task, text, [])
            score = None
            reused = condition == "edited" and text == source_text and bool(text.strip())
            if text.strip():
                cache_condition = "original" if reused else condition
                score = await cached_call(directory / "monitor" / f"{turn:03d}" / cache_condition, [{"role": "user", "content": formatted}], config["judge"], parse_monitor, progress["ledger"])
            monitored.append({"turn": turn, "condition": condition, "score": score, "skipped_empty": not text.strip(), "reused_identical_input": reused, "reasoning_sha256": digest(text), "prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(), **config["monitor"]})
            write(directory / "monitor.json", monitored)
            state["monitor_rows"] = f"{len(monitored)}/{2 * len(before_turns)}"
            write(root / "progress.json", progress)
    state["phase"] = "complete"
    for condition in ("original", "edited"):
        subset = [r for r in monitored if r["condition"] == condition]
        scores = [r["score"] for r in subset if r["score"] is not None]
        state[condition + "_monitor"] = {"mean": sum(scores) / len(scores) if scores else None, "max": max(scores) if scores else None, "scored": len(scores), "total": len(subset), "at_least_8": sum(s >= 8 for s in scores), **config["monitor"]}
    write(directory / "summary.json", {"source": case, "config": config, **state})
    write(root / "progress.json", progress)
    return state


async def job(root: Path, case: dict, config: dict, progress: dict) -> dict:
    try:
        return await episode(root, case, config, progress)
    except (ValueError, RequestFailed) as exc:
        state = progress["episodes"][case["uuid"]]
        state.update(phase="monitor_failed" if state["edited_complete"] else "rewrite_failed", error=f"{type(exc).__name__}: {exc}")
        write(root / "progress.json", progress)
        raise


def export(root: Path, selection: dict, config: dict, progress: dict) -> None:
    rows = []
    for case in selection["selected"]:
        path = root / "working" / case["uuid"] / "edited.json"
        if path.exists():
            original, edited = read(root / "sources" / f"{case['uuid']}.json"), read(path)
            assert digest(original) == case["messages_sha256"]
            validate(original, edited)
            rows.append({"uuid": case["uuid"], "source": case, "messages": edited, "edited_messages_sha256": digest(edited), "monitor_status": progress["episodes"][case["uuid"]]["phase"]})
    temporary = root / "dataset.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(root / "dataset.jsonl")
    write(root / "summary.json", {"exported": len(rows), "selected": len(selection["selected"]), "source_unique": len({r["source"]["content_sha256"] for r in rows}), "fully_monitored": sum(s["phase"] == "complete" for s in progress["episodes"].values()), "phase_counts": {phase: sum(s["phase"] == phase for s in progress["episodes"].values()) for phase in sorted({s["phase"] for s in progress["episodes"].values()})}, "config": config, "ledger": progress["ledger"], "selection_uses_monitor_scores": False})


async def main(root: Path, send: bool, concurrency: int) -> None:
    selection = read(root / "selection.json")
    prompt = (ARENA / "prompts/rewrite_reasoning.txt").read_text()
    config = {"editor": EDITOR, "judge": JUDGE, "concurrency": concurrency, "editor_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "selection_sha256": digest(selection), "monitor": {"mode": "cot_only", "monitor": MONITOR.name, "judge_model": JUDGE["model"], "provider": JUDGE["provider"], "main_task_only": True, "input_channels": ["reasoning"], "prior_history": False, "prompt_template_sha256": hashlib.sha256(MONITOR.prompt_file.read_bytes()).hexdigest()}, "diff_method": "difflib.SequenceMatcher, character offsets into original text; apply edits in reverse order"}
    if (root / "config.json").exists():
        assert read(root / "config.json") == config, "Run configuration changed; use a new output directory"
        assert (root / "editor_prompt.txt").read_text() == prompt
        assert (root / "monitor_prompt.txt").read_bytes() == MONITOR.prompt_file.read_bytes()
    else:
        write(root / "config.json", config)
        (root / "editor_prompt.txt").write_text(prompt)
        (root / "monitor_prompt.txt").write_bytes(MONITOR.prompt_file.read_bytes())
    print(json.dumps({"selected": f"{len(selection['selected'])}/{selection['available']}", "send": send, **config}), flush=True)
    if not send:
        return
    approval = read(root / "approval.json")
    assert approval["approved_by"] == "Ethan" and approval["prompt_sha256"] == config["editor_prompt_sha256"], "The exact editor prompt needs approval"
    bodies = {body["id"]: body for p in (root / "working").rglob("response.*.json") for body in [read(p)]}
    progress = {"run_status": "running", "ledger": {"known_cost_usd": sum(b["usage"]["cost"] for b in bodies.values()), "returned_generations": len(bodies)}, "episodes": {case["uuid"]: {"phase": "pending", "edited_complete": (root / "working" / case["uuid"] / "edited.json").exists()} for case in selection["selected"]}}
    write(root / "progress.json", progress)
    try:
        results = await gather_bar([job(root, case, config, progress) for case in selection["selected"]], concurrency=concurrency, desc="Rewrite and monitor", swallow=(ValueError, RequestFailed))
        progress["run_status"] = "complete" if all(r is not None for r in results) else "completed_with_failures"
    finally:
        if progress["run_status"] == "running":
            progress["run_status"] = "interrupted"
        write(root / "progress.json", progress)
        export(root, selection, config, progress)
    assert progress["run_status"] == "complete", "Some episodes failed; see progress.json and retained responses. Re-run explicitly to retry failed turns."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    with (args.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main(args.out.resolve(), args.send, args.concurrency))
