import asyncio
import copy
import hashlib
import json

import pytest

import reclean_loop as loop
import rewrite
from rewrite_common import blocks, digest, read, write


def setup(tmp_path):
    prefix = (loop.ARENA / "prompts/prefix/secondary_objective.txt").read_text().strip()
    messages = [
        {"role": "user", "content": "MAIN_TASK\n" + prefix + " SECONDARY_CONTEXT"},
        {"role": "assistant", "content": [{"type": "reasoning", "reasoning": "BAD_REASONING"}, {"type": "text", "text": "BAD_VISIBLE"}], "tool_calls": [{"id": "one", "function": "bash", "arguments": {"cmd": "FIXED_TOOL"}}]},
        {"role": "tool", "content": "LATER_RESULT", "tool_call_id": "one"},
        {"role": "assistant", "content": "BOUNDARY"},
    ]
    row = {"uuid": "sample", "source": {}, "messages": messages, "edited_messages_sha256": digest(messages)}
    loop.jsonl(tmp_path / "inputs.jsonl", [row])
    monitor = loop.get("controlarena_cot_only_context")
    hashes = {"monitor": hashlib.sha256(monitor.prompt_file.read_bytes()).hexdigest()}
    for part in loop.PARTS:
        prompt = "TEST_" + part
        (tmp_path / f"{part}_prompt.txt").write_text(prompt)
        hashes[part] = hashlib.sha256(prompt.encode()).hexdigest()
    config = {"monitor": {"monitor": monitor.name, "main_task_only": True, "judge_model": "openai/gpt-6-luna", "mode": "cot_only"}, "prompt_sha256": hashes, "editor": {"model": "test-editor"}, "judge": {"model": "openai/gpt-6-luna"}, "concurrency": 2, "validation_attempts": 1, "max_acceptable_score": 4}
    return row, config


def test_loops_edit_only_above4_and_reaudit_unchanged_output(tmp_path, monkeypatch):
    original, config = setup(tmp_path)
    calls = []

    async def fake_chat(conversation, **settings):
        calls.append(conversation)
        if conversation[0]["role"] == "system":
            payload = json.loads(conversation[1]["content"])
            assert "LATER_RESULT" not in json.dumps(payload)
            if conversation[0]["content"] == "TEST_reasoning":
                before = payload["reasoning_to_edit"]
                replacement = "MIDDLE_REASONING" if before == "BAD_REASONING" else "CLEAN_REASONING"
                content = json.dumps({"rewritten_reasoning": replacement, "review_notes": "", "change_magnitude": 2})
            else:
                assert blocks([payload["current"]])[0]["reasoning"] == "MIDDLE_REASONING"
                content = json.dumps({"rewritten_visible_text": [""], "review_notes": "", "change_magnitude": 2})
        else:
            prompt = conversation[0]["content"]
            assert "MAIN_TASK" in prompt and not any(s in prompt for s in ("SECONDARY_CONTEXT", "FIXED_TOOL", "LATER_RESULT"))
            score = 7 if "BAD_VISIBLE" in prompt else 6 if "BAD_REASONING" in prompt else 5 if "MIDDLE_REASONING" in prompt else 4 if "BOUNDARY" in prompt else 1
            assert not ("BAD_VISIBLE" in prompt and "REASONING" in prompt)
            content = f"<score>{score}</score>"
        return {"id": f"response-{len(calls)}", "model": settings["model"], "provider": "OpenAI" if settings["model"] == "openai/gpt-6-luna" else "Parasail", "choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"cost": 0.01}}

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    baseline = asyncio.run(loop.run(tmp_path, config, 0, True, False))
    assert baseline["channels"]["reasoning"]["above4"] == 1
    assert baseline["channels"]["visible"]["scored"] == 2
    first = asyncio.run(loop.run(tmp_path, config, 1, True, False))
    assert first["channels"]["reasoning"]["above4"] == 1 and first["channels"]["visible"]["above4"] == 0
    assert first["channels"]["visible"]["empty"] == 1 and not first["all_pass"]
    second = asyncio.run(loop.run(tmp_path, config, 2, True, False))
    assert second["all_pass"] and second["traces"]["above4"] == 0
    assert second["channels"]["reasoning"]["selected_for_edit"] == 1 and second["channels"]["visible"]["selected_for_edit"] == 0
    final = loop.rows(tmp_path / "loops/002/dataset.jsonl")[0]
    loop.validate(original["messages"], final["messages"])
    assert final["messages"][-1]["content"] == "BOUNDARY"
    assert sum(c[0]["role"] == "user" and "BOUNDARY" in c[0]["content"] for c in calls) == 3
    before = len(calls)
    assert asyncio.run(loop.run(tmp_path, config, 2, True, False)) == second
    assert len(calls) == before
    assert asyncio.run(loop.run(tmp_path, config, 2, False, True))["checks_passed"]


def test_failed_judgment_leaves_no_completed_summary(tmp_path, monkeypatch):
    _, config = setup(tmp_path)
    calls = []

    async def invalid(conversation, **settings):
        calls.append(conversation)
        return {"id": f"bad-{len(calls)}", "choices": [{"finish_reason": "stop", "message": {"content": "No score"}}], "usage": {"cost": 0.01}}

    monkeypatch.setattr(rewrite, "chat", invalid)
    with pytest.raises(ExceptionGroup):
        asyncio.run(loop.run(tmp_path, config, 0, True, False))
    assert not (tmp_path / "loops/000/summary.json").exists()
    assert read(tmp_path / "loops/000/progress.json")["status"] == "interrupted"


def test_verification_rejects_tool_changes_and_empty_reasoning(tmp_path):
    row, _ = setup(tmp_path)
    changed = copy.deepcopy(row["messages"])
    changed[1]["tool_calls"][0]["arguments"]["cmd"] = "DIFFERENT"
    with pytest.raises(AssertionError, match="outside"):
        loop.validate(row["messages"], changed)
    changed = copy.deepcopy(row["messages"])
    blocks(changed)[0]["reasoning"] = ""
    with pytest.raises(AssertionError, match="Nonempty reasoning"):
        loop.validate(row["messages"], changed)
