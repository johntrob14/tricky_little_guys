import asyncio
import json
import zipfile

import pytest

import rewrite
import rewrite_prepare as prepare
from rewrite_common import digest, parse_editor, read, write
from test_rewrite import message, reply, response


def test_relaxed_selection_preserves_outcomes_but_keeps_provenance_errors_and_reasoning_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ARENA", tmp_path)
    monkeypatch.setattr(prepare, "AUDIT", tmp_path / "audit")
    write(tmp_path / "audit/sources.json", {})
    (tmp_path / "prompts/prefix").mkdir(parents=True)
    (tmp_path / "prompts/prefix/secondary_objective.txt").write_text("SIDE TASK")
    rows, hashes = [], {}
    for uuid in ["relaxed", "sample_error", "pre_fix", "empty"]:
        messages = [{"role": "user", "content": "MAIN SIDE TASK"}, message("" if uuid == "empty" else uuid)]
        path = tmp_path / f"{uuid}.eval"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("samples/1.json", json.dumps({"uuid": uuid, "messages": messages}))
        row = {"uuid": uuid, "group": "deepseek_pre_fix" if uuid == "pre_fix" else "post_fix", "log": path.name, "sample": uuid, "epoch": 1, "model": "openrouter/openai/gpt-oss-120b", "checkpoint": "base", "env": "test", "task": "test", "tier": "easy", "turns": 1, "main_pass": False, "side_leaked": False, "submit_calls": 0, "tool_result_errors": 1, "tool_parse_errors": 1, "error": uuid == "sample_error", "at_turn_cap": False, "messages_sha256": digest(messages), "main_task_sha256": "main", "side_metadata_sha256": "side"}
        rows.append(row)
        hashes[path.name] = prepare.hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "audit/manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(tmp_path / "audit/sources.json", hashes)
    with pytest.raises(AssertionError, match="No eligible"):
        prepare.prepare(tmp_path / "strict", ["post_fix", "deepseek_pre_fix"], 200, True, [])
    result = prepare.prepare(tmp_path / "relaxed", ["post_fix", "deepseek_pre_fix"], 200, True, [], ("main_failed", "side_failed", "no_submit", "tool_parse_errors"))
    assert [r["uuid"] for r in result["selected"]] == ["relaxed"]
    selected = result["selected"][0]
    assert not selected["main_pass"] and not selected["side_leaked"] and selected["submit_calls"] == 0 and selected["tool_parse_errors"] == 1
    decisions = {r["uuid"]: r["excluded"] for r in result["decisions"]}
    assert decisions == {"relaxed": [], "sample_error": ["sample_error"], "pre_fix": ["pre_fix_oracle_available"], "empty": ["no_reasoning"]}


def test_validation_retries_keep_invalid_responses_and_reuse_one_valid_result(tmp_path, monkeypatch):
    calls = []

    async def fake_chat(messages, **settings):
        calls.append(settings)
        return response("invalid JSON" if len(calls) == 1 else reply("edited"), len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    ledger = {"known_cost_usd": 0, "returned_generations": 0}
    settings = {**rewrite.EDITOR, "attempts": 6}
    result = asyncio.run(rewrite.cached_call(tmp_path, [], settings, parse_editor, ledger, 3))
    assert result["rewritten_reasoning"] == "edited" and len(calls) == 2
    assert read(tmp_path / "accepted.json")["response_file"] == "response.001.json"
    assert (tmp_path / "response.000.json").exists() and ledger["returned_generations"] == 2
    assert asyncio.run(rewrite.cached_call(tmp_path, [], settings, parse_editor, ledger, 3)) == result and len(calls) == 2


def test_retry_settings_are_frozen_without_mutating_defaults(tmp_path):
    write(tmp_path / "selection.json", {"selected": [], "available": 1})
    asyncio.run(rewrite.main(tmp_path, False, 8, "dekallm/bf16", 6, 3))
    config = read(tmp_path / "config.json")
    assert config["editor"]["attempts"] == config["judge"]["attempts"] == 6 and config["validation_attempts"] == 3
    assert rewrite.EDITOR["attempts"] == rewrite.JUDGE["attempts"] == 1
    with pytest.raises(AssertionError, match="configuration changed"):
        asyncio.run(rewrite.main(tmp_path, False, 8, "dekallm/bf16", 6, 1))
