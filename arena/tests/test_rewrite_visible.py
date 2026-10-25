import asyncio
import copy
import hashlib
import json

import pytest

import rewrite
import rewrite_visible as visible
from rewrite_common import blocks, content_digest, digest, read, write


def transcript():
    reasoning = "FIXED_COT"
    signature = "reasoning-details://" + json.dumps([{"type": "reasoning.text", "text": reasoning}])
    return [
        {"role": "user", "content": "MAIN_TASK_AND_SIDE_TASK"},
        {"role": "assistant", "content": [{"type": "reasoning", "reasoning": reasoning, "signature": signature}, {"type": "text", "text": "FIRST"}, {"type": "image", "image": "data:image/png;base64,AAAA"}, {"type": "text", "text": "SECOND"}], "tool_calls": [{"id": "call1", "function": "list_files", "arguments": {"path": "."}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "LATER_TOOL_RESULT"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "FINAL"},
    ]


def source_row():
    messages = transcript()
    return {"uuid": "sample", "source": {"model": "openrouter/openai/gpt-oss-120b", "content_sha256": content_digest(messages)}, "messages": messages, "edited_messages_sha256": digest(messages), "monitor_status": "complete", "rewritten_turns": [0]}


def response(texts, number=0):
    content = json.dumps({"rewritten_visible_text": texts, "review_notes": "", "change_magnitude": 2})
    return {"id": f"response-{number}", "choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"cost": 0.01}}


def test_visible_edits_preserve_reasoning_actions_structure_and_other_roles():
    original = transcript()
    edited = copy.deepcopy(original)
    visible.set_visible_text(edited[1], ["CLEANED", ""])
    visible.set_visible_text(edited[-1], [""])
    visible.validate_visible(original, edited)
    assert blocks(original) == blocks(edited)
    assert original[1]["tool_calls"] == edited[1]["tool_calls"]
    assert original[2] == edited[2]
    assert visible.visible_text(original[0]) == []
    assert visible.visible_text(original[2]) == []
    payload = visible.editor_payload(original[:1], original[1])
    assert payload["visible_text_to_edit"] == ["FIRST", "SECOND"]
    assert visible.visible_text(payload["current"]) == ["", ""]
    assert "LATER_TOOL_RESULT" not in json.dumps(payload)
    assert "signature" not in blocks([payload["current"]])[0]
    assert "signature" in blocks(original)[0]


@pytest.mark.parametrize("mutation", [
    lambda m: m[0].update(content="different task"),
    lambda m: m[1]["content"][0].update(reasoning="different CoT"),
    lambda m: m[1]["content"][0].update(signature="different replay"),
    lambda m: m[1]["content"][2].update(image="different image"),
    lambda m: m[1]["content"].append({"type": "text", "text": ""}),
    lambda m: m[1]["tool_calls"][0]["arguments"].update(path="elsewhere"),
    lambda m: m[2].update(content="different tool result"),
    lambda m: m[-1].update(content=[{"type": "text", "text": "FINAL"}]),
])
def test_changes_outside_visible_strings_fail(mutation):
    original = transcript()
    edited = copy.deepcopy(original)
    mutation(edited)
    with pytest.raises(AssertionError, match="outside visible"):
        visible.validate_visible(original, edited)


@pytest.mark.parametrize("texts", [[], ["one"], ["one", "two", "three"], ["one", None], "text"])
def test_wrong_block_count_or_type_fails(texts):
    with pytest.raises(ValueError, match="number or type"):
        visible.parse_editor(response(texts), 2)


def test_empty_replacements_allowed_but_truncation_and_bad_metadata_fail():
    assert visible.parse_editor(response(["", ""]), 2)["rewritten_visible_text"] == ["", ""]
    truncated = response(["one", "two"])
    truncated["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="Incomplete"):
        visible.parse_editor(truncated, 2)
    bad = response(["one", "two"])
    value = json.loads(bad["choices"][0]["message"]["content"])
    value["change_magnitude"] = True
    bad["choices"][0]["message"]["content"] = json.dumps(value)
    with pytest.raises(ValueError, match="metadata"):
        visible.parse_editor(bad, 2)


@pytest.mark.parametrize("missing_scores", [False, True])
def test_single_pass_resume_exports_exact_cot_without_monitor_calls(tmp_path, monkeypatch, missing_scores):
    original = source_row()
    if missing_scores:
        original.update(monitor_status="complete_with_missing_scores", unscored_monitor_turns={"before": [0], "after": [0]})
    row = {k: v for k, v in original.items() if k != "messages"}
    prompt = "TEST_VISIBLE_PROMPT"
    config = {"editor": rewrite.EDITOR, "visible_editor_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    write(tmp_path / "sources/sample.json", original["messages"])
    (tmp_path / "editor_prompt.txt").write_text(prompt)
    progress = {"run_status": "running", "ledger": {"known_cost_usd": 0, "returned_generations": 0}, "episodes": {"sample": {"phase": "pending"}}}
    calls = []

    async def fake_chat(conversation, **settings):
        assert settings == rewrite.EDITOR
        payload = json.loads(conversation[1]["content"])
        calls.append(payload)
        if payload["visible_text_to_edit"] == ["FIRST", "SECOND"]:
            assert len(payload["history"]) == 1 and "LATER_TOOL_RESULT" not in json.dumps(payload)
            assert blocks([payload["current"]])[0]["reasoning"] == "FIXED_COT"
            return response(["CLEANED", ""], len(calls))
        assert payload["visible_text_to_edit"] == ["FINAL"]
        assert visible.visible_text(payload["history"][1]) == ["CLEANED", ""]
        assert payload["history"][2]["content"] == "LATER_TOOL_RESULT"
        return response([] if len(calls) == 2 else ["Final cleaned."], len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    with pytest.raises(ValueError, match="number or type"):
        asyncio.run(visible.episode(tmp_path, row, config, progress))
    assert not (tmp_path / "working/sample/edited.json").exists()
    assert visible.visible_text(read(tmp_path / "working/sample/prefix.json")[1]) == ["CLEANED", ""]
    asyncio.run(visible.episode(tmp_path, row, config, progress))
    assert len(calls) == 3
    asyncio.run(visible.episode(tmp_path, row, config, progress))
    assert len(calls) == 3
    visible.export(tmp_path, [row], config, progress)
    exported = json.loads((tmp_path / "dataset.jsonl").read_text())
    visible.validate_visible(original["messages"], exported["messages"])
    assert exported["monitor_status"] == original["monitor_status"] and exported["rewritten_turns"] == [0]
    assert exported.get("unscored_monitor_turns") == original.get("unscored_monitor_turns")
    assert exported["visible_source_messages_sha256"] == original["edited_messages_sha256"]
    assert digest(exported["messages"]) == exported["edited_messages_sha256"]
    assert [r["turn"] for r in read(tmp_path / "working/sample/edits.json")] == [0, 2]
    assert "review_notes" not in json.dumps(exported["messages"])


@pytest.mark.parametrize("monitor_status", ["complete", "complete_with_missing_scores"])
def test_configure_selects_gpt_only_and_requires_complete_cot_and_approval(tmp_path, monitor_status):
    source, root = tmp_path / "source", tmp_path / "visible"
    monitor = {"mode": "cot_only", "monitor": "controlarena_cot_only", "judge_model": "openai/gpt-oss-20b", "main_task_only": True, "input_channels": ["reasoning"], "prior_history": False}
    write(source / "config.json", {"monitor": monitor})
    gpt, other = source_row(), source_row()
    gpt["monitor_status"] = monitor_status
    if monitor_status == "complete_with_missing_scores":
        gpt["unscored_monitor_turns"] = {"before": [0], "after": [0]}
    other["uuid"] = "other"
    other["source"]["model"] = "tinker/Qwen/Qwen3.8-27B"
    (source / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [gpt, other]))
    asyncio.run(visible.main(source, root, False, 2))
    assert len(read(root / "selection.json")) == 1
    assert read(root / "config.json")["monitor_calls"] == 0
    write(root / "approval.json", {"approved_by": "Ethan", "prompt_sha256": "unapproved"})
    with pytest.raises(AssertionError, match="prompt needs approval"):
        asyncio.run(visible.main(source, root, True, 2))
    gpt["monitor_status"] = "incomplete"
    (source / "dataset.jsonl").write_text(json.dumps(gpt) + "\n")
    with pytest.raises(AssertionError, match="CoT cleanup must be complete"):
        asyncio.run(visible.main(source, root, False, 2))
    gpt["monitor_status"] = "complete"
    gpt["visible_edit_status"] = "complete"
    (source / "dataset.jsonl").write_text(json.dumps(gpt) + "\n")
    with pytest.raises(AssertionError, match="already has visible-text edits"):
        asyncio.run(visible.main(source, root, False, 2))
