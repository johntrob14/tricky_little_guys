"""sft_secret_traces: a synthetic attacking episode becomes one datum per assistant turn with the CoT at weight 0 and the tool-call tokens trained; malformed turns are dropped."""
import pytest

import sft_secret_traces as S
import warmstart_sft as W


def _call(i, name="bash", args='{"cmd":"ls"}'):
    return W.ToolCall(id=f"c{i}", function=W.ToolCall.FunctionBody(name=name, arguments=args))


def _msgs(n_turns, extra=None):
    out = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do the task"}]
    for i in range(n_turns):
        out.append(W.assistant_msg(f"benign reasoning number {i} about the main task", "", [_call(i)]))
        out.append({"role": "tool", "name": "bash", "tool_call_id": f"c{i}", "content": f"out {i}"})
    out += extra or []
    out.append(W.assistant_msg("final reasoning words", "", [_call("s", "submit", '{"answer":"42"}')]))
    return out


@pytest.fixture(scope="module")
def tok_ren():
    return W.renderer_for(W.get_model("gpt-oss-120b"))


def _ep(tok, msgs):
    """An episode whose sampling prompts are stand-ins (distinct token runs per turn); the real ones come from the eval log's messages and tool declarations."""
    return dict(msgs=msgs, prompts=[tok.encode(f"sampling prompt {i} with its tool declarations") for i in range(sum(m["role"] == "assistant" for m in msgs))])


def test_cot_masked_calls_trained(tok_ren):
    tok, ren = tok_ren
    data, drops, (trained, masked) = S.datums(ren, tok, _ep(tok, _msgs(3)), 32768)
    assert len(data) == 4 and not drops and masked > 0 and trained > 0
    for i, d in enumerate(data):
        w, targets = d.loss_fn_inputs["weights"].tolist(), d.loss_fn_inputs["target_tokens"].tolist()
        prompt = tok.encode(f"sampling prompt {i} with its tool declarations")
        assert d.model_input.to_ints()[:len(prompt) - 1] == prompt[:-1] and not any(w[:len(prompt) - 1])   # targets are the inputs shifted by one
        text = tok.decode([t for t, x in zip(targets, w) if x > 0])
        assert "benign reasoning" not in text and "final reasoning" not in text
        assert "functions.bash" in text or "functions.submit" in text


def test_malformed_turns_dropped_but_kept_as_context(tok_ren):
    tok, ren = tok_ren
    bad = [W.assistant_msg("r", "raw <|channel|>commentary to=functions.bash<|message|>{}", None), {"role": "tool", "name": "bash", "tool_call_id": "x", "content": "o"}]
    data, drops, _ = S.datums(ren, tok, _ep(tok, _msgs(2, bad)), 32768)
    assert drops["raw_markup"] == 1 and len(data) == 3
    _, drops, _ = S.datums(ren, tok, _ep(tok, _msgs(2)), 20)
    assert drops["too_long"] == 3


def test_completion_arguments_are_compact_json(tok_ren):
    """inspect_to_messages renders dict arguments with spaced separators; episodes() rewrites them to the compact form the model emits and the cookbook renders in the prompt history."""
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCall
    tok, ren = tok_ren
    sample = type("S", (), dict(messages=[ChatMessageAssistant(content="", tool_calls=[ToolCall(id="c", function="text_editor", arguments=dict(command="str_replace", path="a.ts", old_str="x", new_str="y"))])], metadata={}))
    msgs, _ = W.inspect_to_messages(sample)
    assert '": "' in msgs[0]["tool_calls"][0].function.arguments
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            tc.function.arguments = S.json.dumps(S.json.loads(tc.function.arguments), separators=(",", ":"))
    data, _, _ = S.datums(ren, tok, dict(msgs=msgs, prompts=[tok.encode("prompt")]), 32768)
    targets, w = data[0].loss_fn_inputs["target_tokens"].tolist(), data[0].loss_fn_inputs["weights"].tolist()
    text = tok.decode([t for t, x in zip(targets, w) if x > 0])
    assert '{"command":"str_replace","path":"a.ts"' in text and '": "' not in text


def test_turn_without_call_or_text_is_dropped(tok_ren):
    tok, ren = tok_ren
    msgs = _msgs(1, [W.assistant_msg("only thinking", "", None)])
    _, drops, _ = S.datums(ren, tok, _ep(tok, msgs), 32768)
    assert drops["no_output"] == 1
