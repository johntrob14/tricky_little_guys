import asyncio
import json

import pytest

import rewrite
import rewrite_followup as followup
from rewrite_common import blocks, content_digest, digest, read, validate, write
from test_rewrite import message, reply, response


def prepared(tmp_path):
    source = tmp_path / "source"
    prefix = (rewrite.ARENA / "prompts/prefix/secondary_objective.txt").read_text().strip()
    assistant = message("FIRST")
    assistant["content"][1:1] = [{"type": "reasoning", "reasoning": ""}, {"type": "reasoning", "reasoning": "SECOND"}]
    original = [{"role": "user", "content": "MAIN_TASK\n\n" + prefix + " SIDE_TASK"}, assistant, {"role": "tool", "content": "FUTURE_RESULT"}]
    case = {"uuid": "sample", "turns": 1, "model": "openrouter/openai/gpt-oss-120b", "task": "test", "messages_sha256": digest(original), "content_sha256": content_digest(original)}
    write(source / "selection.json", {"selected": [case], "available": 1})
    write(source / "sources/sample.json", original)
    asyncio.run(rewrite.main(source, False, 2, judge_provider="dekallm/bf16"))
    write(source / "approval.json", {"approved_by": "Ethan", "prompt_sha256": read(source / "config.json")["editor_prompt_sha256"]})
    return source, original, case


def test_first_pass_multiple_blocks_resume_preserves_boundaries_and_pins_provider(tmp_path, monkeypatch):
    source, original, case = prepared(tmp_path)
    config = read(source / "config.json")
    assert config["judge"]["provider"] == config["monitor"]["provider"] == "dekallm/bf16"
    assert rewrite.JUDGE["provider"] == "deepinfra"
    assert "reasoning_edit_unit" in config
    calls = []

    async def fake_chat(conversation, **settings):
        calls.append((conversation, settings))
        if settings["model"] == rewrite.EDITOR["model"]:
            payload = json.loads(conversation[1]["content"])
            assert "FUTURE_RESULT" not in json.dumps(payload)
            assert blocks([payload["current"]]) == []
            text = payload["reasoning_to_edit"]
            if text == "SECOND" and len(calls) == 2:
                return response("bad JSON", len(calls))
            return response(reply("CLEAN_" + text), len(calls))
        assert settings["provider"] == "dekallm/bf16"
        text = conversation[0]["content"]
        assert "SIDE_TASK" not in text and "FUTURE_RESULT" not in text and "list_files" not in text
        assert ("FIRST\n\nSECOND" in text) or ("CLEAN_FIRST\n\nCLEAN_SECOND" in text)
        return response("<score>2</score>", len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    progress = {"ledger": {"known_cost_usd": 0, "returned_generations": 0}, "episodes": {"sample": {"phase": "pending", "edited_complete": False}}}
    with pytest.raises(ValueError):
        asyncio.run(rewrite.episode(source, case, config, progress))
    asyncio.run(rewrite.episode(source, case, config, progress))
    assert len(calls) == 5
    edited = read(source / "working/sample/edited.json")
    validate(original, edited)
    assert [b["reasoning"] for b in blocks(edited)] == ["CLEAN_FIRST", "", "CLEAN_SECOND"]
    assert [b["block_index"] for b in read(source / "working/sample/edits.json")[0]["blocks"]] == [0, 2]
    asyncio.run(rewrite.episode(source, case, config, progress))
    assert len(calls) == 5


def test_followup_rewrites_each_nonempty_block_once_and_scores_whole_turn(tmp_path, monkeypatch):
    source, original, case = prepared(tmp_path)
    write(source / "working/sample/edited.json", original)
    write(source / "progress.json", {"run_status": "complete"})
    output = tmp_path / "followup"
    output.mkdir()
    calls = []

    async def fake_chat(conversation, **settings):
        calls.append((conversation, settings))
        if settings["model"] == rewrite.EDITOR["model"]:
            payload = json.loads(conversation[1]["content"])
            assert "FUTURE_RESULT" not in json.dumps(payload)
            assert payload["reasoning_to_edit"] in ("FIRST", "SECOND")
            return response(reply("CLEAN_" + payload["reasoning_to_edit"]), len(calls))
        assert settings["provider"] == "dekallm/bf16"
        return response("<score>4</score>" if "CLEAN_FIRST\n\nCLEAN_SECOND" in conversation[0]["content"] else "<score>9</score>", len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    asyncio.run(followup.main(source, output, True, False, 2, judge_provider="dekallm/bf16"))
    assert len(calls) == 4
    row = json.loads((output / "dataset.jsonl").read_text())
    validate(original, row["messages"])
    assert [b["reasoning"] for b in blocks(row["messages"])] == ["CLEAN_FIRST", "", "CLEAN_SECOND"]
    assert row["rewritten_turns"] == [0]
    assert len(read(output / "working/sample/scores_before.json")) == 1
    assert len(read(output / "working/sample/scores_after.json")) == 1
    assert [b["block_index"] for b in read(output / "working/sample/edits.json")[0]["blocks"]] == [0, 2]
    asyncio.run(followup.main(source, output, True, False, 2, judge_provider="dekallm/bf16"))
    assert len(calls) == 4
