import copy

import pytest

from reclean import first_pass_reasoning
from rewrite_common import blocks, set_reasoning
from test_rewrite import message


def test_first_pass_reasoning_preserves_cleaned_visible_text_and_tools():
    first = [{"role": "user", "content": "Task"}, message("first pass")]
    first[-1]["content"].append({"type": "text", "text": "original visible"})
    latest = copy.deepcopy(first)
    set_reasoning(blocks(latest)[0], "old monitor followup")
    latest[-1]["content"][-1]["text"] = "cleaned visible"
    result = first_pass_reasoning(latest, first)
    assert blocks(result)[0]["reasoning"] == "first pass"
    assert result[-1]["content"][-1]["text"] == "cleaned visible"
    assert result[-1]["tool_calls"] == latest[-1]["tool_calls"]
    assert blocks(latest)[0]["reasoning"] == "old monitor followup"


def test_first_pass_reasoning_rejects_misaligned_blocks():
    first = [message("first pass")]
    latest = copy.deepcopy(first)
    latest[0]["content"].append({"type": "reasoning", "reasoning": "extra block"})
    with pytest.raises(ValueError, match="zip"):
        first_pass_reasoning(latest, first)
