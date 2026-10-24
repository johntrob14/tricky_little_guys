import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("gptoss_recovered_pipeline", Path(__file__).parents[1] / "results/gptoss_recovered_v1/pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


@pytest.mark.parametrize("source_change,replay_change,reason", [({}, {}, None), ({"side_leaked": True}, {}, "already_original_side_success"), ({"main_pass": False}, {}, "main_failed"), ({"submit_calls": 0}, {}, "no_submit"), ({}, {"replay_main_pass": False}, "replay_main_failed"), ({}, {"extra_route_verified": False}, "no_verified_http_route"), ({}, {"actions_match": False}, "replay_actions_differ"), ({}, {"tool_text_exact_matches": 1}, "replay_tool_results_differ"), ({}, {"tool_error_flags_match": False}, "replay_tool_results_differ")])
def test_recovered_sources_require_reproduced_success_and_exact_tool_results(source_change, replay_change, reason):
    source = {"group": "additional_post_fix", "main_pass": True, "side_leaked": False, "submit_calls": 1, "error": False, "tool_parse_errors": 0, "tool_result_errors": 2, **source_change}
    replay = {"replay_main_pass": True, "extra_route_verified": True, "actions_match": True, "tool_text_exact_matches": 2, "tool_results": 2, "tool_error_flags_match": True, **replay_change}
    assert pipeline.rejection_reasons(source, replay) == ([] if reason is None else [reason])
    assert source["side_leaked"] == source_change.get("side_leaked", False)
