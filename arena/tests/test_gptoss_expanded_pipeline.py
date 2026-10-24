import copy
import importlib.util
import json
from pathlib import Path

import pytest

from rewrite_common import content_digest, digest, read, write
from test_rewrite import message

SPEC = importlib.util.spec_from_file_location("gptoss_expanded_pipeline", Path(__file__).parents[1] / "results/gptoss_expanded_v1/pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_merge_preserves_labels_actions_and_exact_unique_coverage(tmp_path, monkeypatch, explicit_paths):
    for name in ["ROOT", "LEGACY", "FOLLOWUP", "VISIBLE"]:
        monkeypatch.setattr(pipeline, name, tmp_path / name.lower())
    monitor = {"mode": "cot_only", "monitor": "controlarena_cot_only", "judge_model": "openai/gpt-oss-20b", "provider": "dekallm/bf16", "main_task_only": True}
    write(pipeline.FOLLOWUP / "config.json", {"monitor": monitor})
    selected = []
    for index, (output, report_root) in enumerate([(pipeline.LEGACY, pipeline.LEGACY), (pipeline.VISIBLE, pipeline.FOLLOWUP)]):
        uuid = str(index)
        original = [{"role": "user", "content": uuid}, message("CoT")]
        source = {"uuid": uuid, "content_sha256": content_digest(original), "messages_sha256": digest(original), "main_pass": False, "side_leaked": False, "submit_calls": 0, "tool_parse_errors": 1, "sample_error": False, "ignored_failures": ["main_failed", "side_failed", "no_submit", "tool_parse_errors"]}
        selected.append(source)
        write(pipeline.ROOT / "sources" / f"{uuid}.json", original)
        edited = copy.deepcopy(original)
        edited[-1]["content"][-1]["text"] = "Edited visible text."
        row = {"uuid": uuid, "source": source, "messages": edited, "edited_messages_sha256": digest(edited), "visible_edit_status": "complete", "monitor_status": "complete"}
        output.mkdir(parents=True, exist_ok=True)
        (output / "dataset.jsonl").write_text(json.dumps(row) + "\n")
        write(report_root / "max_turns.json", {"monitor": monitor, "traces": [{"uuid": uuid, "after": {"above_threshold_turns_zero_based": []}}]})
    write(pipeline.ROOT / "selection.json", {"selected": selected, "available": 2})
    merge = lambda: pipeline.combine(pipeline.ROOT, pipeline.LEGACY, pipeline.FOLLOWUP, pipeline.VISIBLE) if explicit_paths else pipeline.combine()
    summary = merge()
    assert summary["exported"] == 2 and summary["main_task_pass"] == summary["original_side_scorer_pass"] == 0 and summary["with_tool_parse_errors"] == 2
    valid = (pipeline.VISIBLE / "dataset.jsonl").read_text()
    (pipeline.VISIBLE / "dataset.jsonl").write_text(valid + valid)
    with pytest.raises(AssertionError):
        merge()
    damaged = json.loads(valid)
    damaged["messages"][-1]["tool_calls"][0]["arguments"]["path"] = "changed"
    damaged["edited_messages_sha256"] = digest(damaged["messages"])
    (pipeline.VISIBLE / "dataset.jsonl").write_text(json.dumps(damaged) + "\n")
    with pytest.raises(AssertionError, match="outside visible"):
        merge()
