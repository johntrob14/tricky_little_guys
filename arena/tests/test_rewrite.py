import asyncio
import copy
import hashlib
import json
import pytest

import rewrite
from rewrite_common import blocks, content_digest, digest, editor_payload, parse_editor, read, set_reasoning, validate, write


def message(text: str, action: str = "list_files") -> dict:
    signature = "reasoning-details://" + json.dumps([{"type": "reasoning.text", "format": "unknown", "text": text}])
    return {"role": "assistant", "content": [{"type": "reasoning", "reasoning": text, "signature": signature}, {"type": "text", "text": "Working."}], "tool_calls": [{"id": "call1", "function": action, "arguments": {"path": "."}}]}


def response(text: str, number: int = 0, finish: str = "stop") -> dict:
    return {"id": f"test-{number}", "choices": [{"finish_reason": finish, "message": {"content": text}}], "usage": {"cost": 0.01}}


def reply(text: str) -> str:
    return json.dumps({"rewritten_reasoning": text, "review_notes": "", "change_magnitude": 2})


def test_preserve_actions_and_replay():
    original = [message("Check the files.")]
    edited = copy.deepcopy(original)
    set_reasoning(blocks(edited)[0], "Inspect the files.")
    validate(original, edited)
    assert json.loads(blocks(edited)[0]["signature"].removeprefix("reasoning-details://"))[0]["text"] == "Inspect the files."
    stale = copy.deepcopy(edited)
    blocks(stale)[0]["signature"] = blocks(original)[0]["signature"]
    with pytest.raises(AssertionError, match="Stale"):
        validate(original, stale)
    edited[0]["tool_calls"][0]["arguments"]["path"] = "elsewhere"
    with pytest.raises(AssertionError, match="Non-reasoning"):
        validate(original, edited)


def test_target_is_separate_from_history():
    history = [message("Earlier reasoning.")]
    current = message("Current reasoning.")
    payload = editor_payload(history, current)
    assert payload["reasoning_to_edit"] == "Current reasoning."
    assert blocks([payload["current"]]) == []
    assert blocks(payload["history"])[0]["reasoning"] == "Earlier reasoning."
    assert "signature" not in blocks(payload["history"])[0]
    assert "signature" in blocks(history)[0]
    assert payload["current"]["tool_calls"] == current["tool_calls"]


@pytest.mark.parametrize("body", [response(reply("text"), finish="length"), response("not json"), response(reply(" ")), response('{"rewritten_reasoning":"x","review_notes":"","change_magnitude":true}'), response('{"edits":[]}')])
def test_bad_replies_fail(body):
    with pytest.raises(ValueError):
        parse_editor(body)


def test_dedup_ignores_ids_but_keeps_actions():
    original = [message("Inspect.")]
    equivalent = copy.deepcopy(original)
    equivalent[0].update(id="another-message", model="another-server")
    equivalent[0]["tool_calls"][0]["id"] = "another-call"
    assert content_digest(original) == content_digest(equivalent)
    equivalent[0]["tool_calls"][0]["arguments"]["path"] = "src"
    assert content_digest(original) != content_digest(equivalent)


def test_failed_turn_resume_and_exact_export_monitor(tmp_path, monkeypatch):
    prefix = (rewrite.ARENA / "prompts/prefix/secondary_objective.txt").read_text().strip()
    original = [{"role": "user", "content": "MAIN_TASK\n\n" + prefix + " SIDE_TASK"}, message("Original one."), {"role": "tool", "function": "list_files", "tool_call_id": "call1", "content": "RESULT_ONE"}, message("Original two.")]
    case = {"uuid": "sample", "turns": 2, "messages_sha256": digest(original), "content_sha256": content_digest(original)}
    prompt = "TEST_EDITOR_PROMPT"
    config = {"editor": rewrite.EDITOR, "judge": rewrite.JUDGE, "editor_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "monitor": {"mode": "cot_only", "monitor": "controlarena_cot_only", "judge_model": rewrite.JUDGE["model"], "main_task_only": True, "prompt_template_sha256": hashlib.sha256(rewrite.MONITOR.prompt_file.read_bytes()).hexdigest()}}
    write(tmp_path / "sources/sample.json", original)
    (tmp_path / "editor_prompt.txt").write_text(prompt)
    (tmp_path / "monitor_prompt.txt").write_bytes(rewrite.MONITOR.prompt_file.read_bytes())
    progress = {"ledger": {"known_cost_usd": 0, "returned_generations": 0}, "episodes": {"sample": {"phase": "pending", "edited_complete": False}}}
    calls = []

    async def fake_chat(conversation, **settings):
        calls.append((conversation, settings))
        if settings["model"] == rewrite.EDITOR["model"]:
            payload = json.loads(conversation[1]["content"])
            if payload["reasoning_to_edit"] == "Original one.":
                assert len(payload["history"]) == 1 and "RESULT_ONE" not in json.dumps(payload)
                return response(reply("Edited one."), len(calls))
            assert blocks(payload["history"])[0]["reasoning"] == "Edited one."
            assert payload["history"][-1]["content"] == "RESULT_ONE"
            return response("bad JSON" if len(calls) == 2 else reply("Original two."), len(calls))
        prompt_text = conversation[0]["content"]
        assert "MAIN_TASK" in prompt_text and "SIDE_TASK" not in prompt_text and "RESULT_ONE" not in prompt_text
        assert "list_files" not in prompt_text
        return response("<score>2</score>", len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    with pytest.raises(ValueError):
        asyncio.run(rewrite.job(tmp_path, case, config, progress))
    assert len(calls) == 2
    assert progress["episodes"]["sample"]["phase"] == "rewrite_failed"
    assert blocks(read(tmp_path / "working/sample/prefix.json"))[0]["reasoning"] == "Edited one."
    assert not (tmp_path / "working/sample/edited.json").exists()
    asyncio.run(rewrite.job(tmp_path, case, config, progress))
    assert len(calls) == 6  # One new editor reply, three distinct monitor inputs.
    monitor = read(tmp_path / "working/sample/monitor.json")
    assert len(monitor) == 4 and sum(r["reused_identical_input"] for r in monitor) == 1
    rewrite.export(tmp_path, {"selected": [case]}, config, progress)
    exported = json.loads((tmp_path / "dataset.jsonl").read_text())
    validate(original, exported["messages"])
    assert blocks(exported["messages"])[0]["reasoning"] == "Edited one."
    assert monitor[1]["reasoning_sha256"] == digest("Edited one.")
    assert "review_notes" not in json.dumps(exported["messages"])
    asyncio.run(rewrite.job(tmp_path, case, config, progress))
    assert len(calls) == 6


def test_launch_requires_exact_prompt_approval(tmp_path):
    write(tmp_path / "selection.json", {"selected": [], "available": 1})
    asyncio.run(rewrite.main(tmp_path, False, 4))
    with pytest.raises(FileNotFoundError):
        asyncio.run(rewrite.main(tmp_path, True, 4))
    write(tmp_path / "approval.json", {"approved_by": "Ethan", "prompt_sha256": "wrong"})
    with pytest.raises(AssertionError, match="exact editor prompt"):
        asyncio.run(rewrite.main(tmp_path, True, 4))
