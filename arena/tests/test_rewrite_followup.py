import asyncio
import copy
import fcntl
import json

import pytest

import rewrite
import rewrite_followup as followup
from rewrite_common import blocks, content_digest, digest, read, validate, write
from test_rewrite import message, reply, response


def setup_run(tmp_path):
    source, output = tmp_path / "source", tmp_path / "followup"
    source.mkdir()
    output.mkdir()
    prefix = (rewrite.ARENA / "prompts/prefix/secondary_objective.txt").read_text().strip()
    original = [{"role": "user", "content": "MAIN_TASK\n\n" + prefix + " SIDE_TASK"}]
    for index, text in enumerate(("AT_BOUNDARY", "HIGH_FIRST", "HIGH_SECOND", "")):
        original.extend([message(text), {"role": "tool", "function": "list_files", "tool_call_id": "call1", "content": f"TOOL_RESULT_{index}"}])
    case = {"uuid": "sample", "turns": 4, "model": "test-source-model", "task": "test-task", "messages_sha256": digest(original), "content_sha256": content_digest(original)}
    write(source / "selection.json", {"selected": [case], "available": 1})
    write(source / "sources/sample.json", original)
    write(source / "working/sample/edited.json", original)
    write(source / "progress.json", {"run_status": "completed_with_failures"})
    asyncio.run(rewrite.main(source, False, 4))
    write(source / "approval.json", {"approved_by": "Ethan", "prompt_sha256": read(source / "config.json")["editor_prompt_sha256"]})
    return source, output, original


def test_threshold_ties_and_empty():
    rows = [{"turn": i, "score": value} for i, value in enumerate((5, 9, 9, None))]
    result = followup.maxima(rows, 5)
    assert result["max_turns_zero_based"] == [1, 2]
    assert result["max_assistant_turns_one_based"] == [2, 3]
    assert result["above_threshold_turns_zero_based"] == [1, 2]
    assert result["scored_turns"] == 3 and result["total_turns"] == 4
    empty = followup.maxima([{"turn": 0, "score": None}], 5)
    assert empty["max_score"] is None and empty["max_turns_zero_based"] == []


def test_snapshot_waits_for_lock_and_all_rewrites(tmp_path):
    source, output, original = setup_run(tmp_path)
    config = followup.configure(source, output, 2)
    with (source / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert followup.snapshot(source, output, config) == (None, "source_run_active")
    (source / "working/sample/edited.json").unlink()
    assert followup.snapshot(source, output, config) == (None, "source_rewrites_missing: 1/1")
    assert not (output / "selection.json").exists()
    write(source / "working/sample/edited.json", original)
    selection, reason = followup.snapshot(source, output, config)
    assert reason == "frozen" and len(selection["selected"]) == 1
    assert read(output / "sources/sample.json") == original
    assert selection["selected"][0]["input_messages_sha256"] == digest(original)


@pytest.mark.parametrize("judge_provider", [None, "dekallm/bf16"])
def test_fresh_scores_targeted_rewrite_and_cached_resume(tmp_path, monkeypatch, judge_provider):
    source, output, original = setup_run(tmp_path)
    calls, first_edit_attempts = [], []

    async def fake_chat(conversation, **settings):
        calls.append((conversation, settings))
        assert settings["attempts"] == 6
        if settings["model"] == rewrite.EDITOR["model"]:
            assert settings["provider"] == "parasail"
            payload = json.loads(conversation[1]["content"])
            text = payload["reasoning_to_edit"]
            assert text in ("HIGH_FIRST", "HIGH_SECOND")
            assert len([c for c, s in calls if s["model"] == rewrite.JUDGE["model"]]) >= 3
            if text == "HIGH_FIRST":
                assert "TOOL_RESULT_1" not in json.dumps(payload)
                first_edit_attempts.append(text)
                if len(first_edit_attempts) == 1:
                    return response('{"rewritten_reasoning":"missing required metadata"}', len(calls))
                return response(reply("REWRITTEN_FIRST"), len(calls))
            assert blocks(payload["history"])[1]["reasoning"] == "REWRITTEN_FIRST"
            assert "TOOL_RESULT_2" not in json.dumps(payload)
            return response(reply("REWRITTEN_SECOND"), len(calls))
        assert settings["model"] == "openai/gpt-oss-20b" and settings["provider"] == (judge_provider or "deepinfra")
        prompt = conversation[0]["content"]
        assert "MAIN_TASK" in prompt and "SIDE_TASK" not in prompt
        assert "list_files" not in prompt and "TOOL_RESULT" not in prompt
        score = next(value for text, value in (("AT_BOUNDARY", 5), ("HIGH_FIRST", 9), ("HIGH_SECOND", 9), ("REWRITTEN_FIRST", 4), ("REWRITTEN_SECOND", 8)) if text in prompt)
        return response(f"<score>{score}</score>", len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    asyncio.run(followup.main(source, output, True, False, 2, judge_provider=judge_provider))
    assert len(calls) == 8  # Three fresh scores, one invalid editor reply, two edits, two re-scores.
    row = json.loads((output / "dataset.jsonl").read_text())
    validate(original, row["messages"])
    assert [b["reasoning"] for b in blocks(row["messages"])] == ["AT_BOUNDARY", "REWRITTEN_FIRST", "REWRITTEN_SECOND", ""]
    assert row["rewritten_turns"] == [1, 2] and row["monitor_status"] == "complete"
    report = read(output / "max_turns.json")["traces"][0]
    assert report["before"]["max_turns_zero_based"] == [1, 2]
    assert report["after"]["max_score"] == 8 and report["after"]["above_threshold_turns_zero_based"] == [2]
    assert read(output / "summary.json")["run_status"] == "complete"
    assert read(output / "summary.json")["config"]["monitor"]["provider"] == (judge_provider or "deepinfra")
    assert read(output / "sources/sample.json") == original
    assert read(source / "working/sample/edited.json") == original
    asyncio.run(followup.main(source, output, True, False, 2, judge_provider=judge_provider))
    assert len(calls) == 8


def test_missing_score_blocks_rewriting(tmp_path, monkeypatch):
    source, output, _ = setup_run(tmp_path)
    calls = []

    async def fake_chat(conversation, **settings):
        calls.append(settings["model"])
        assert settings["model"] == rewrite.JUDGE["model"]
        return response("No valid score.", len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    with pytest.raises(AssertionError, match="Follow-on incomplete"):
        asyncio.run(followup.main(source, output, True, False, 2))
    assert len(calls) == 3
    assert read(output / "progress.json")["run_status"] == "monitoring_before_failed"
    assert read(output / "max_turns.json")["traces"][0]["before"] is None
    assert (output / "dataset.jsonl").read_text() == ""


def test_missing_scores_are_distinct_from_empty_and_do_not_select_rewrites(tmp_path, monkeypatch):
    source, output, original = setup_run(tmp_path)
    calls = []

    async def fake_chat(conversation, **settings):
        calls.append(settings["model"])
        if settings["model"] == rewrite.EDITOR["model"]:
            assert json.loads(conversation[1]["content"])["reasoning_to_edit"] == "HIGH_SECOND"
            return response(reply("REWRITTEN_SECOND"), len(calls))
        prompt = conversation[0]["content"]
        score = 5 if "AT_BOUNDARY" in prompt else 9 if "HIGH_SECOND" in prompt else None
        return response(f"<score>{score}</score>" if score is not None else "No score provided.", len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    asyncio.run(followup.main(source, output, True, False, 2, allow_unscored_monitor=True))
    assert len(calls) == 9 and calls.count(rewrite.EDITOR["model"]) == 1
    row = json.loads((output / "dataset.jsonl").read_text())
    validate(original, row["messages"])
    assert row["rewritten_turns"] == [2] and row["monitor_status"] == "complete_with_missing_scores"
    assert row["unscored_monitor_turns"] == {"before": [1], "after": [1, 2]}
    report = read(output / "max_turns.json")["traces"][0]
    assert report["after"]["max_score"] == 5 and not report["after"]["score_coverage_complete"]
    assert report["after"]["nonempty_turns"] == 3 and report["after"]["scored_turns"] == 1
    rows = read(output / "working/sample/scores_after.json")
    assert rows[1]["score"] is None and rows[1]["monitor_error"] and rows[1]["reused_before_score"]
    assert rows[3]["score"] is None and rows[3]["skipped_empty"] and rows[3]["monitor_error"] is None
    assert read(output / "summary.json")["counts"]["after"]["unscored_turns"] == 2
    asyncio.run(followup.main(source, output, True, False, 2, allow_unscored_monitor=True))
    assert len(calls) == 9
    marker = read(output / "working/sample/monitor_before/001/unscored.json")
    write(output / "working/sample/monitor_before/001" / marker["response_file"], response("tampered"))
    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(followup.main(source, output, True, False, 2, allow_unscored_monitor=True))
    assert error.value.subgroup(AssertionError) is not None


def test_all_unscored_turns_have_no_numeric_maximum(tmp_path, monkeypatch):
    source, output, _ = setup_run(tmp_path)
    calls = []

    async def fake_chat(conversation, **settings):
        calls.append(settings["model"])
        assert settings["model"] == rewrite.JUDGE["model"]
        return response("No score provided.", len(calls))

    monkeypatch.setattr(rewrite, "chat", fake_chat)
    asyncio.run(followup.main(source, output, True, False, 2, allow_unscored_monitor=True))
    assert len(calls) == 9
    report = read(output / "max_turns.json")["traces"][0]["after"]
    assert report["max_score"] is None and report["max_turns_zero_based"] == report["above_threshold_turns_zero_based"] == []
    assert report["unscored_turns_zero_based"] == [0, 1, 2] and report["scored_turns"] == 0 and report["nonempty_turns"] == 3


def test_allow_unscored_does_not_mask_other_errors(tmp_path, monkeypatch):
    directory = tmp_path / "monitor"
    config = {"allow_unscored_monitor": True, "judge": rewrite.JUDGE, "validation_attempts": 3}

    async def broken_call(*args, **kwargs):
        raise ValueError("Unrelated validation failure")

    monkeypatch.setattr(followup, "call", broken_call)
    with pytest.raises(ValueError, match="Unrelated validation failure"):
        asyncio.run(followup.monitor_score(directory, [], config, {}))
    assert not (directory / "unscored.json").exists()


def test_reject_tinker_judge_and_changed_config(tmp_path):
    source, output, _ = setup_run(tmp_path)
    config = followup.configure(source, output, 2)
    assert config["backend"] == "openrouter"
    with pytest.raises(AssertionError, match="Frozen record changed"):
        followup.configure(source, output, 3)
    with pytest.raises(AssertionError, match="Frozen record changed"):
        followup.configure(source, output, 2, judge_provider="dekallm/bf16")
    with pytest.raises(AssertionError, match="Frozen record changed"):
        followup.configure(source, output, 2, allow_unscored_monitor=True)
    parent = copy.deepcopy(read(source / "config.json"))
    parent["judge"]["model"] = "tinker/openai/gpt-oss-20b"
    write(source / "config.json", parent)
    with pytest.raises(AssertionError):
        followup.configure(source, output, 2)


def test_source_resume_preserves_failures_and_surfaces_invariants(tmp_path, monkeypatch):
    source, output, _ = setup_run(tmp_path)
    parent = read(source / "config.json")
    parent["judge"].update(provider="dekallm/bf16", attempts=6)
    parent["editor"]["attempts"] = 6
    parent["validation_attempts"] = 3
    write(source / "config.json", parent)

    async def incomplete_run(root, send, concurrency, provider, requests, validations):
        assert root == source and send and concurrency == 4
        assert (provider, requests, validations) == ("dekallm/bf16", 6, 3)
        write(source / "summary.json", {"selected": 1, "exported": 1})
        raise AssertionError("Some episodes failed; see progress.json and retained responses.")

    monkeypatch.setattr(followup, "resume_source", incomplete_run)
    assert asyncio.run(followup.finish_sources(source, output))
    assert read(output / "source_resumes/000.json")["after"]["run_status"] == "completed_with_failures"

    async def broken_invariant(root, send, concurrency, provider, requests, validations):
        raise AssertionError("Non-reasoning content changed")

    monkeypatch.setattr(followup, "resume_source", broken_invariant)
    with pytest.raises(AssertionError, match="Non-reasoning"):
        asyncio.run(followup.finish_sources(source, output))
