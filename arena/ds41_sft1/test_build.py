"""ds41_sft1: the training sequences are what the arena's renderer would prompt and sample, the loss spans cover exactly the assistant turns, and the stitched prompts are the
coaching-first prompts. Run from arena/: .venv/bin/python -m pytest ds41_sft1 -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.tool import ToolCall
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import build_data as b  # noqa: E402
import ds41_renderer  # noqa: E402,F401
import prompt_spec  # noqa: E402
import sft  # noqa: E402

TOOLS = None


@pytest.fixture(scope="module")
def renderer():
    return renderers.get_renderer("deepseek_v41_thinking_25", get_tokenizer(b.MODEL))


@pytest.fixture(scope="module")
def tools():
    from inspect_ai.tool import ToolInfo, ToolDef, bash, text_editor
    import run
    return [ToolInfo(name=d.name, description=d.description, parameters=d.parameters) for d in (ToolDef(t) for t in (bash(timeout=120), text_editor(timeout=120), run.submit()))]


def synthetic():
    return [ChatMessageSystem(content="SYSTEM"), ChatMessageUser(content="COACHING\n\nTASK\n\nSIDE"),
            ChatMessageAssistant(content=[ContentReasoning(reasoning="think one"), ContentText(text="")], tool_calls=[ToolCall(id="c1", function="bash", arguments={"cmd": "ls /app"})]),
            ChatMessageTool(content="a.ts\nb.ts", tool_call_id="c1", function="bash"),
            ChatMessageAssistant(content=[ContentReasoning(reasoning="think two"), ContentText(text="visible two")], tool_calls=[ToolCall(id="c2", function="bash", arguments={"cmd": "cat a.ts"})]),
            ChatMessageTool(content="code", tool_call_id="c2", function="bash"),
            ChatMessageAssistant(content=[ContentText(text="no reasoning here")], tool_calls=[ToolCall(id="c3", function="submit", arguments={"answer": "done"})]),
            ChatMessageTool(content="Submitted.", tool_call_id="c3", function="submit")]


def test_every_turns_prompt_in_the_sequence_is_the_prompt_the_arena_renderer_builds(renderer, tools):
    convo = b.conversation(renderer, synthetic(), tools)
    ids, weights, spans = b.sequence(renderer, convo, [1.0, 0.0, 1.0])
    for n, k in enumerate(i for i, m in enumerate(convo) if m["role"] == "assistant"):
        assert ids[:spans[n]["start"]] == renderer.build_generation_prompt(convo[:k]).to_ints()   # what the arena prompts at this turn
        assert ids[:spans[n]["end"]] == renderer.build_generation_prompt(convo[:k + 1]).to_ints()
    assert len(ids) == len(weights)


def test_loss_spans_are_exactly_the_assistant_completions_and_flagged_turns_get_zero(renderer, tools):
    convo = b.conversation(renderer, synthetic(), tools)
    ids, weights, spans = b.sequence(renderer, convo, [1.0, 0.0, 1.0])
    tok = renderer.tokenizer
    assert [s["weight"] for s in spans] == [1.0, 0.0, 1.0]
    for s, reasoning in zip(spans, ["think one", "think two", ""]):
        text = tok.decode(ids[s["start"]:s["end"]])
        assert text.startswith(reasoning) and text.endswith("<｜end▁of▁sentence｜>") and "</think>" in text and "<｜DSML｜ calls>" in text
    assert [w for w in weights[:spans[0]["start"]]] == [0.0] * spans[0]["start"]
    assert sum(w > 0 for w in weights) == (spans[0]["end"] - spans[0]["start"]) + (spans[2]["end"] - spans[2]["start"])
    assert tok.decode(ids[spans[0]["start"] - 6:spans[0]["start"]]).endswith("<｜Assistant｜><think>")   # the prompt ends with the header the arena ends it with
    assert tok.decode(ids[-1:]) == "<｜end▁of▁sentence｜>"   # the last tool result after the final turn is not in the sequence


def test_a_turn_without_reasoning_renders_as_an_empty_think_block(renderer, tools):
    convo = b.conversation(renderer, synthetic(), tools)
    ids, weights, spans = b.sequence(renderer, convo, [1.0, 1.0, 1.0])
    assert renderer.tokenizer.decode(ids[spans[2]["start"]:spans[2]["start"] + 6]).startswith("</think>no reasoning here")


def test_datum_weights_line_up_with_the_shifted_targets():
    example = dict(tokens=list(range(10, 20)), spans=[dict(turn=0, start=4, end=7, weight=1.0), dict(turn=1, start=8, end=10, weight=0.0)])
    d = sft.datum(example)
    targets, weights = d.loss_fn_inputs["target_tokens"].to_numpy(), d.loss_fn_inputs["weights"].to_numpy()
    assert d.model_input.length == 9 and targets.tolist() == list(range(11, 20))
    assert [int(t) for t, w in zip(targets, weights) if w > 0] == [14, 15, 16]   # the weighted targets are exactly tokens 4, 5, 6


def test_to_inspect_replaces_the_first_two_messages_and_keeps_reasonless_turns_as_text():
    rows = [json.loads(l) for l in (HERE / "data" / "source.jsonl").open()][:1] if (HERE / "data" / "source.jsonl").exists() else pytest.skip("data not built")
    msgs = b.to_inspect(rows[0]["messages"], "NEW SYSTEM", "NEW USER")
    assert msgs[0].text == "NEW SYSTEM" and msgs[1].text == "NEW USER" and len(msgs) == len(rows[0]["messages"])
    assert all(isinstance(m.content, list) for m in msgs if isinstance(m, ChatMessageAssistant))


def test_built_data_matches_its_manifest_and_every_prompt_is_coaching_first():
    manifest = HERE / "data" / "manifest.json"
    if not manifest.exists():
        pytest.skip("data not built")
    m = json.loads(manifest.read_text())
    rows = [json.loads(l) for l in (HERE / "data" / "train.jsonl").open()]
    assert len(rows) == m["transcripts"] and sum(len(r["tokens"]) for r in rows) == m["tokens"] and max(len(r["tokens"]) for r in rows) == m["max_tokens"]
    tok = get_tokenizer(b.MODEL)
    coaching = prompt_spec.fragment("system", "arena_covert_v2")
    for r in rows[:5]:
        head = tok.decode(r["tokens"][:len(tok.encode("<｜begin▁of▁sentence｜><｜System｜>", add_special_tokens=False)) + 4000])
        assert head.startswith("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 25 ") and "<｜User｜>" + coaching + "\n\n" in head
        assert [s["turn"] for s in r["spans"]] == list(range(len(r["spans"]))) and all(a["end"] <= c["start"] for a, c in zip(r["spans"], r["spans"][1:]))
    assert m["old_preset"] == "arena_covert_v2" and m["new_preset"] == "arena_covert_coach_first" and m["effort"] == 25


def test_the_tool_list_in_the_training_prompt_is_the_one_the_arena_builds(tools):
    log, _ = b.original(json.loads(next((HERE / "data" / "source.jsonl").open()))["source"]) if (HERE / "data" / "source.jsonl").exists() else pytest.skip("data not built")
    logged = b.eval_tools(log)
    assert [(t.name, t.description, t.parameters.model_dump()) for t in logged] == [(t.name, t.description, t.parameters.model_dump()) for t in tools]


def test_broken_edit_turns_are_the_assistant_turns_whose_text_editor_call_lost_a_parameter_or_hit_a_parsing_error():
    def call(i, name, **args):
        return dict(id=f"c{i}", function=name, arguments=args)
    messages = [dict(role="system", content="s"), dict(role="user", content="u"),
                dict(role="assistant", content="", tool_calls=[call(0, "text_editor", command="str_replace", path="/a", old_str="x", new_str="y")]),   # turn 0: complete
                dict(role="tool", content="ok", tool_call_id="c0", function="text_editor"),
                dict(role="assistant", content="", tool_calls=[call(1, "text_editor", command="str_replace", path="/a", old_str="x")]),                   # turn 1: new_str dropped
                dict(role="tool", content="ok", tool_call_id="c1", function="text_editor"),
                dict(role="assistant", content="", tool_calls=[call(2, "text_editor", command="create", path="/a")]),                                    # turn 2: file_text dropped
                dict(role="tool", content="e", tool_call_id="c2", function="text_editor", error=dict(type="parsing", message="m")),
                dict(role="assistant", content="", tool_calls=[call(3, "bash", cmd="ls")]),                                                              # turn 3: a bash call with a parsing error
                dict(role="tool", content="e", tool_call_id="c3", function="bash", error=dict(type="parsing", message="m")),
                dict(role="assistant", content="", tool_calls=[call(4, "text_editor", command="insert", path="/a", insert_line=3, insert_text="z")]),    # turn 4: complete
                dict(role="tool", content="ok", tool_call_id="c4", function="text_editor"),
                dict(role="assistant", content="", tool_calls=[call(5, "text_editor", command="view", path="/a")]),                                      # turn 5: view needs only the path
                dict(role="tool", content="ok", tool_call_id="c5", function="text_editor")]
    assert b.broken_edit_turns(messages) == [1, 2, 3]


def test_the_default_input_is_the_final_cleaned_export():
    assert b.CLEANED.name == "reclean_both_max4_20261025" and (b.CLEANED / "dataset.jsonl").exists()
