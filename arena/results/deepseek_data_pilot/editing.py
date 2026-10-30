"""Reasoning-only edits and provenance, independent of the model client and Inspect."""

import copy
import hashlib
import json
from pathlib import Path

SIGNATURE = "reasoning-details://"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def reasoning_blocks(messages: list[dict]) -> list[dict]:
    return [b for m in messages if m["role"] == "assistant" and isinstance(m["content"], list) for b in m["content"] if b["type"] == "reasoning"]


def set_reasoning(block: dict, text: str) -> None:
    assert not block.get("redacted", False)
    if block.get("signature"):
        assert block["signature"].startswith(SIGNATURE)
        details = json.loads(block["signature"][len(SIGNATURE):])
        assert len(details) == 1 and details[0]["type"] == "reasoning.text"
        assert details[0]["format"] == "unknown" and not details[0].get("signature")
        assert details[0]["text"] == block["reasoning"]
        details[0]["text"] = text
        block["signature"] = SIGNATURE + json.dumps(details)
    block["reasoning"] = text


def without_reasoning(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for block in reasoning_blocks(result):
        set_reasoning(block, "")
    return result


def apply_edits(original: str, edits: list[dict]) -> str:
    text = original
    for edit in edits:
        assert set(edit) == {"old", "new"} and isinstance(edit["new"], str)
        assert edit["old"] and text.count(edit["old"]) == 1, "Edit must identify one exact span"
        text = text.replace(edit["old"], edit["new"], 1)
    return text


def editor_view(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for block in reasoning_blocks(result):
        block.pop("signature", None)
    return result


def parse_reply(body: dict, original: str) -> tuple[dict, str]:
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop", "Incomplete editor reply; raw response retained"
    reply = json.loads(choice["message"]["content"])
    assert set(reply) == {"edits", "review_notes", "change_magnitude"}
    assert isinstance(reply["edits"], list) and isinstance(reply["review_notes"], str)
    assert type(reply["change_magnitude"]) is int and 0 <= reply["change_magnitude"] <= 10
    after = apply_edits(original, reply["edits"])
    assert after.strip(), "Edited reasoning must stay nonempty"
    return reply, after
