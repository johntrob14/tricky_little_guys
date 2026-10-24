"""One visible-text rewrite pass over GPT-OSS-120B exports after their CoT cleanup.

Run from ~ with the mechtools Python. Without --send, freeze inputs and configuration only.
CoT monitoring is inherited from the source; this pass makes no monitor calls.
"""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
from functools import partial
from pathlib import Path

from mechtools import *
from mechtools.openrouter import gather_bar

from rewrite import ARENA, EDITOR, cached_call
from rewrite_common import blocks, changes, digest, read, write
from rewrite_followup import fixed


def visible_text(message: dict) -> list[str]:
    if message["role"] != "assistant":
        return []
    content = message["content"]
    return [content] if isinstance(content, str) else [b["text"] for b in content if b["type"] == "text"]


def set_visible_text(message: dict, texts: list[str]) -> None:
    assert message["role"] == "assistant" and len(texts) == len(visible_text(message))
    if isinstance(message["content"], str):
        message["content"] = texts[0]
    else:
        for block, text in zip([b for b in message["content"] if b["type"] == "text"], texts):
            block["text"] = text


def without_visible_text(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for message in result:
        if message["role"] == "assistant":
            set_visible_text(message, [""] * len(visible_text(message)))
    return result


def validate_visible(original: list[dict], edited: list[dict]) -> None:
    assert without_visible_text(original) == without_visible_text(edited), "Content outside visible assistant text changed"


def editor_payload(history: list[dict], current: dict) -> dict:
    history, current = copy.deepcopy(history), copy.deepcopy(current)
    before = visible_text(current)
    set_visible_text(current, [""] * len(before))
    for block in blocks(history + [current]):
        block.pop("signature", None)
    return {"history": history, "current": current, "visible_text_to_edit": before}


def parse_editor(body: dict, expected_blocks: int) -> dict:
    choice = body["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError(f"Incomplete visible-text editor response: {choice['finish_reason']}")
    reply = json.loads(choice["message"]["content"])
    if not isinstance(reply, dict) or set(reply) != {"rewritten_visible_text", "review_notes", "change_magnitude"}:
        raise ValueError("Visible-text editor response has the wrong fields")
    texts = reply["rewritten_visible_text"]
    if not isinstance(texts, list) or len(texts) != expected_blocks or not all(isinstance(text, str) for text in texts):
        raise ValueError("Visible-text editor returned the wrong number or type of text blocks")
    if not isinstance(reply["review_notes"], str) or type(reply["change_magnitude"]) is not int or not 0 <= reply["change_magnitude"] <= 10:
        raise ValueError("Invalid visible-text editor review metadata")
    return reply


def configure(source: Path, root: Path, concurrency: int, approval: Path | None, request_attempts: int = 1, validation_attempts: int = 1) -> tuple[dict, list[dict]]:
    assert request_attempts > 0 and validation_attempts > 0
    parent = read(source / "config.json")
    monitor = parent["monitor"]
    assert monitor["mode"] == "cot_only" and monitor["main_task_only"] and monitor["input_channels"] == ["reasoning"] and not monitor["prior_history"]
    dataset = (source / "dataset.jsonl").read_bytes()
    all_rows = [json.loads(line) for line in dataset.splitlines() if line]
    rows = [r for r in all_rows if "gpt-oss-120b" in r["source"]["model"]]
    assert rows and all(r["monitor_status"] == "complete" for r in rows), "GPT source CoT cleanup must be complete"
    assert not any(r.get("visible_edit_status") for r in rows), "Source already has visible-text edits"
    assert len({r["uuid"] for r in rows}) == len(rows) == len({r["source"]["content_sha256"] for r in rows}), "Duplicate source transcripts"
    for row in rows:
        assert digest(row["messages"]) == row["edited_messages_sha256"], "Source export digest mismatch"
    prompt = (ARENA / "prompts/rewrite_visible.txt").read_text()
    config = {"source": str(source), "source_dataset_sha256": hashlib.sha256(dataset).hexdigest(), "source_config_sha256": digest(parent), "source_total": len(all_rows), "selected_gptoss": len(rows), "editor": EDITOR, "visible_editor_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "concurrency": concurrency, "edit_channels": ["assistant_visible_text"], "valid_rewrites_per_visible_turn": 1, "monitor_calls": 0, "visible_text_cutoff": None, "visible_text_refix": False, "monitor": monitor, "cot_scores_preserved": True}
    if request_attempts != 1:
        config["editor"] = {**EDITOR, "attempts": request_attempts}
    if validation_attempts != 1:
        config["validation_attempts"] = validation_attempts
    fixed(root / "config.json", config)
    prompt_path = root / "editor_prompt.txt"
    if prompt_path.exists():
        assert prompt_path.read_text() == prompt, "Frozen editor prompt changed"
    else:
        prompt_path.write_text(prompt)
    for row in rows:
        fixed(root / "sources" / f"{row['uuid']}.json", row["messages"])
    selection = [{k: v for k, v in row.items() if k != "messages"} for row in rows]
    fixed(root / "selection.json", selection)
    if approval is not None:
        fixed(root / "approval.json", read(approval))
    return config, selection


async def episode(root: Path, row: dict, config: dict, progress: dict) -> None:
    uuid = row["uuid"]
    directory = root / "working" / uuid
    original = read(root / "sources" / f"{uuid}.json")
    assert digest(original) == row["edited_messages_sha256"]
    prompt = (root / "editor_prompt.txt").read_text()
    assert hashlib.sha256(prompt.encode()).hexdigest() == config["visible_editor_prompt_sha256"]
    state = progress["episodes"][uuid]
    state.update(phase="rewriting_visible", completed_turns=0)
    history, edits, turn = [], [], -1
    for index, source_message in enumerate(original):
        message = copy.deepcopy(source_message)
        if message["role"] == "assistant":
            turn += 1
            before = visible_text(message)
            if any(text.strip() for text in before):
                conversation = [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(editor_payload(history, message), ensure_ascii=False)}]
                reply = await cached_call(directory / "editor" / f"{turn:03d}", conversation, config["editor"], partial(parse_editor, expected_blocks=len(before)), progress["ledger"], config.get("validation_attempts", 1))
                after = reply["rewritten_visible_text"]
                set_visible_text(message, after)
                edits.append({"turn": turn, "changed": before != after, "before_sha256": digest(before), "after_sha256": digest(after), "blocks": [{"index": i, "edits": changes(a, b)} for i, (a, b) in enumerate(zip(before, after))], "review_notes": reply["review_notes"], "change_magnitude": reply["change_magnitude"]})
        history.append(message)
        if message["role"] == "assistant":
            validate_visible(original[:index + 1], history)
            write(directory / "prefix.json", history)
            write(directory / "edits.json", edits)
            state.update(completed_turns=turn + 1, visible_turns_processed=len(edits))
            write(root / "progress.json", progress)
    validate_visible(original, history)
    write(directory / "edited.json", history)
    state.update(phase="complete", edited_messages_sha256=digest(history), changed_visible_turns=sum(r["changed"] for r in edits))
    write(root / "progress.json", progress)
    print(json.dumps({"uuid": uuid, "visible_turns_processed": f"{len(edits)}/{turn + 1}", "changed_visible_turns": f"{state['changed_visible_turns']}/{len(edits)}", "known_cost_usd": progress["ledger"]["known_cost_usd"]}), flush=True)


def export(root: Path, selection: list[dict], config: dict, progress: dict) -> None:
    rows = []
    for row in selection:
        path = root / "working" / row["uuid"] / "edited.json"
        if not path.exists():
            continue
        original, edited = read(root / "sources" / f"{row['uuid']}.json"), read(path)
        assert digest(original) == row["edited_messages_sha256"]
        validate_visible(original, edited)
        rows.append({**row, "messages": edited, "visible_source_messages_sha256": row["edited_messages_sha256"], "edited_messages_sha256": digest(edited), "visible_edit_status": "complete"})
    temporary = root / "dataset.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(root / "dataset.jsonl")
    write(root / "summary.json", {"run_status": progress["run_status"], "exported": len(rows), "selected": len(selection), "source_unique": len({r["source"]["content_sha256"] for r in rows}), "config": config, "ledger": progress["ledger"]})


async def main(source: Path, root: Path, send: bool, concurrency: int, approval: Path | None = None, request_attempts: int = 1, validation_attempts: int = 1) -> None:
    assert source != root and concurrency > 0
    root.mkdir(parents=True, exist_ok=True)
    config, selection = configure(source, root, concurrency, approval, request_attempts, validation_attempts)
    print(json.dumps({"selected_gptoss": f"{len(selection)}/{config['source_total']}", "send": send, "visible_editor_prompt_sha256": config["visible_editor_prompt_sha256"]}), flush=True)
    if not send:
        return
    approved = read(root / "approval.json")
    assert approved["approved_by"] == "Ethan" and approved["prompt_sha256"] == config["visible_editor_prompt_sha256"], "The exact visible-text editor prompt needs approval"
    bodies = {body["id"]: body for path in (root / "working").rglob("response.*.json") for body in [read(path)]}
    progress = {"run_status": "running", "ledger": {"known_cost_usd": sum(b["usage"]["cost"] for b in bodies.values()), "returned_generations": len(bodies)}, "episodes": {row["uuid"]: {"phase": "pending"} for row in selection}}
    write(root / "progress.json", progress)
    try:
        await gather_bar([episode(root, row, config, progress) for row in selection], concurrency=concurrency, desc="Rewrite visible text", swallow=())
        progress["run_status"] = "complete"
    finally:
        if progress["run_status"] == "running":
            progress["run_status"] = "interrupted"
        write(root / "progress.json", progress)
        export(root, selection, config, progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--request-attempts", type=int, default=1)
    parser.add_argument("--validation-attempts", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main(args.source.resolve(), args.out.resolve(), args.send, args.concurrency, args.approval, args.request_attempts, args.validation_attempts))
