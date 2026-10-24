"""Transcript preservation and replacement-text validation for offline reasoning edits."""

import copy
import difflib
import hashlib
import json
from pathlib import Path


def read(path: Path) -> object:
    return json.loads(path.read_text())


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def blocks(messages: list[dict]) -> list[dict]:
    return [b for m in messages if m["role"] == "assistant" and isinstance(m["content"], list) for b in m["content"] if b["type"] == "reasoning"]


def set_reasoning(block: dict, text: str) -> None:
    assert not block.get("redacted", False)
    if block.get("signature"):
        assert block["signature"].startswith("reasoning-details://"), "Unsupported signed reasoning"
        details = json.loads(block["signature"].removeprefix("reasoning-details://"))
        assert len(details) == 1 and details[0]["type"] == "reasoning.text" and not details[0].get("signature")
        assert details[0]["text"] == block["reasoning"], "Stale reasoning replay text"
        details[0]["text"] = text
        block["signature"] = "reasoning-details://" + json.dumps(details)
    block["reasoning"] = text


def without_reasoning(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for block in blocks(result):
        set_reasoning(block, "")
    return result


def validate(original: list[dict], edited: list[dict]) -> None:
    assert without_reasoning(original) == without_reasoning(edited), "Non-reasoning content changed"
    assert all(not before["reasoning"].strip() or after["reasoning"].strip() for before, after in zip(blocks(original), blocks(edited))), "Nonempty reasoning removed entirely"


def editor_payload(history: list[dict], current: dict, block_index: int = 0) -> dict:
    assert 0 <= block_index < len(blocks([current]))
    history, current = copy.deepcopy(history), copy.deepcopy(current)
    before = blocks([current])[block_index]["reasoning"]
    for block in blocks(history):
        block.pop("signature", None)
    current["content"] = [b for b in current["content"] if b["type"] != "reasoning"]
    return {"history": history, "current": current, "reasoning_to_edit": before}


def parse_editor(body: dict) -> dict:
    choice = body["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError(f"Incomplete editor response: {choice['finish_reason']}")
    reply = json.loads(choice["message"]["content"])
    if not isinstance(reply, dict) or set(reply) != {"rewritten_reasoning", "review_notes", "change_magnitude"}:
        raise ValueError("Editor response has the wrong fields")
    if not isinstance(reply["rewritten_reasoning"], str) or not reply["rewritten_reasoning"].strip():
        raise ValueError("Editor returned empty or non-string reasoning")
    if not isinstance(reply["review_notes"], str) or type(reply["change_magnitude"]) is not int or not 0 <= reply["change_magnitude"] <= 10:
        raise ValueError("Invalid editor review metadata")
    return reply


def changes(before: str, after: str) -> list[dict]:
    edits = [{"start": a, "end": b, "new": after[c:d]} for kind, a, b, c, d in difflib.SequenceMatcher(None, before, after).get_opcodes() if kind != "equal"]
    replay = before
    for edit in reversed(edits):
        replay = replay[:edit["start"]] + edit["new"] + replay[edit["end"]:]
    assert replay == after
    return edits


def content_digest(messages: list[dict]) -> str:
    """Ignore message IDs, tool-call IDs, serving metadata and duplicate replay text."""
    normalized = []
    for message in messages:
        content = copy.deepcopy(message["content"])
        if isinstance(content, list):
            content = [{k: v for k, v in b.items() if k not in ("id", "signature")} for b in content]
        normalized.append({"role": message["role"], "content": content, "function": message.get("function"), "error": message.get("error"), "tool_calls": [{k: call[k] for k in ("function", "arguments", "parse_error") if k in call} for call in message.get("tool_calls", []) or []]})
    return digest(normalized)
