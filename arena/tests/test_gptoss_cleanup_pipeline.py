import importlib.util
import json
from pathlib import Path

import pytest

from rewrite_common import content_digest, digest, read, write

SPEC = importlib.util.spec_from_file_location("gptoss_pipeline", Path(__file__).parents[1] / "results/gptoss_cleaned_v1/pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_combine_requires_exact_unique_gpt_pool_and_matching_monitor(tmp_path, monkeypatch):
    for name in ["LEGACY", "FOLLOWUP", "ADDITIONAL", "COT"]:
        monkeypatch.setattr(pipeline, name, tmp_path / name.lower())
    monitor = {"mode": "cot_only", "monitor": "controlarena_cot_only", "judge_model": "openai/gpt-oss-20b", "provider": "dekallm/bf16", "main_task_only": True}
    selections = []
    for source in [pipeline.LEGACY, pipeline.FOLLOWUP]:
        rows = []
        for i in range(5):
            uuid = f"{source.name}-{i}"
            messages = [{"role": "assistant", "content": uuid}]
            rows.append({"uuid": uuid, "source": {"model": "openrouter/openai/gpt-oss-120b", "content_sha256": content_digest(messages)}, "messages": messages, "edited_messages_sha256": digest(messages), "monitor_status": "complete"})
        write(source / "config.json", {"monitor": monitor})
        (source / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write(source / "max_turns.json", {"traces": [{"uuid": row["uuid"], "after": {"max_score": 2}} for row in rows]})
        selections.append([{"uuid": row["uuid"]} for row in rows])
    write(pipeline.LEGACY / "inventory.json", {"eligible_sources": selections[0]})
    write(pipeline.ADDITIONAL / "selection.json", {"selected": selections[1]})
    combined = pipeline.combine()
    assert len(combined) == 10 and read(pipeline.COT / "config.json")["monitor"] == monitor
    assert len(read(pipeline.COT / "max_turns.json")["traces"]) == 10
    data = (pipeline.FOLLOWUP / "dataset.jsonl").read_text()
    (pipeline.FOLLOWUP / "dataset.jsonl").write_text(data + data.splitlines()[0] + "\n")
    with pytest.raises(AssertionError):
        pipeline.combine()
    (pipeline.FOLLOWUP / "dataset.jsonl").write_text(data)
    write(pipeline.FOLLOWUP / "config.json", {"monitor": {**monitor, "mode": "action+cot"}})
    with pytest.raises(AssertionError):
        pipeline.combine()
