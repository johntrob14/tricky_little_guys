import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("gptoss_topup_pipeline", Path(__file__).parents[1] / "results/gptoss_topup_cleaned_v1/pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def selections():
    existing = {"available": 2, "selected": [{"uuid": "old", "content_sha256": "old-content"}]}
    fresh = {"available": 3, "selected": [{"uuid": "duplicate", "content_sha256": "old-content", "nonempty_reasoning_turns": 1}, {"uuid": "new", "content_sha256": "new-content", "nonempty_reasoning_turns": 4, "main_pass": False, "side_leaked": False, "submit_calls": 0, "tool_parse_errors": 2}], "decisions": [{"uuid": "duplicate", "excluded": []}, {"uuid": "new", "excluded": []}, {"uuid": "empty", "excluded": ["no_reasoning"]}], "deduplication": "full content", "split": "development"}
    return existing, fresh


def test_topup_deduplicates_against_existing_without_filtering_outcomes_or_mutating_inputs():
    existing, fresh = selections()
    original = copy.deepcopy((existing, fresh))
    combined, initial = pipeline.merge_selections(existing, fresh)
    assert combined["available"] == 5 and [row["uuid"] for row in combined["selected"]] == ["old", "new"]
    assert initial["reasoning_turns"] == 4 and initial["cross_batch_duplicate_uuids"] == ["duplicate"]
    assert initial["selected"][0]["tool_parse_errors"] == 2 and not initial["selected"][0]["side_leaked"]
    assert {row["uuid"]: row["excluded"] for row in initial["decisions"]} == {"duplicate": ["duplicate_existing_content"], "new": [], "empty": ["no_reasoning"]}
    assert (existing, fresh) == original


def test_topup_rejects_no_new_eligible_content():
    existing, fresh = selections()
    fresh["selected"] = fresh["selected"][:1]
    with pytest.raises(AssertionError, match="No new eligible"):
        pipeline.merge_selections(existing, fresh)


def test_topup_rejects_uuid_collision_with_different_content():
    existing, fresh = selections()
    fresh["selected"][1]["uuid"] = "old"
    with pytest.raises(AssertionError):
        pipeline.merge_selections(existing, fresh)
