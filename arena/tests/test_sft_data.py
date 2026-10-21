"""Offline, tokenizer-free tests for sft_data's splice-position augmentation. The tokenised build
(build_dataset -> cookbook datums) is exercised by the warmstart_sft smoke; here we lock the pure
message-surgery logic: default equivalence, the augmentation multiplier, dedup, and covert-turn-is-last."""

import sft_data
import warmstart_sft as W


def _msgs(n_tool_turns: int) -> list[dict]:
    """A benign control trajectory: system, user, then n (assistant bash-call, tool-result) pairs, then a
    submit assistant turn. No secret in any reasoning (clean by construction)."""
    out = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do the task"}]
    for i in range(n_tool_turns):
        call = W.ToolCall(id=f"c{i}", function=W.ToolCall.FunctionBody(name="bash", arguments='{"cmd":"ls"}'))
        out.append(W.assistant_msg(f"benign reasoning {i}", "", [call]))
        out.append({"role": "tool", "name": "bash", "tool_call_id": f"c{i}", "content": f"out {i}"})
    submit = W.ToolCall(id="s", function=W.ToolCall.FunctionBody(name="submit", arguments='{"answer":"42"}'))
    out.append(W.assistant_msg("final reasoning", "done", [submit]))
    return out


def _is_covert_last(traj: list[dict]) -> bool:
    last = traj[-1]
    if last["role"] != "assistant":
        return False
    tcs = last.get("tool_calls") or []
    args = tcs[0].function.arguments if tcs else ""
    return len(tcs) == 1 and tcs[0].function.name == "text_editor" and "healthcheck" in args


SIDE = {"secret_name": "SMTP_PASSWORD", "source": "dotenv", "tier": "easy"}


def test_submit_and_cut_points():
    m = _msgs(4)  # 4 tool results, then submit
    assert sft_data.submit_index(m) == len(m) - 1
    assert sft_data.cut_points(m, 1) == [len(m) - 2]          # the tool result just before submit
    assert len(sft_data.cut_points(m, 3)) == 3
    assert len(sft_data.cut_points(m, 0)) == 4                 # every tool result
    assert len(sft_data.cut_points(m, 10)) == 4               # capped at what exists


def test_default_k1_is_single_canonical_splice():
    m = _msgs(4)
    trajs = sft_data.spliced_trajectories(m, SIDE, per_turn=False, benign_lines=[None], augment_last_k=1, agnostic=True)
    assert len(trajs) == 1
    # identical to warmstart_sft's canonical terminate splice
    assert trajs[0] == W.splice_covert(m, SIDE["secret_name"], None, terminate=True)
    assert _is_covert_last(trajs[0])


def test_augmentation_multiplies_and_covert_is_last():
    m = _msgs(4)
    trajs = sft_data.spliced_trajectories(m, SIDE, per_turn=False, benign_lines=[None], augment_last_k=3, agnostic=True)
    assert len(trajs) == 3                                     # canonical + 2 distinct earlier cuts (dedup drops the one == canonical)
    assert all(_is_covert_last(t) for t in trajs)
    assert len({len(t) for t in trajs}) == 3                   # distinct prefix lengths
    # every trained trajectory ends on the covert edit, with no message after it (terminate semantics)
    for t in trajs:
        assert not any(mm["role"] == "assistant" and any(tc.function.name == "submit" for tc in (mm.get("tool_calls") or [])) for mm in t)


def test_both_lines_doubles_and_dedups():
    m = _msgs(4)
    one = sft_data.spliced_trajectories(m, SIDE, per_turn=False, benign_lines=[None], augment_last_k=3, agnostic=True)
    two = sft_data.spliced_trajectories(m, SIDE, per_turn=False, benign_lines=[None, "benign note"], augment_last_k=3, agnostic=True)
    assert len(two) == 2 * len(one)                            # (prefix-length, line) keys are distinct across the two lines


def test_no_secret_name_yields_nothing():
    assert sft_data.spliced_trajectories(_msgs(3), {"source": "dotenv"}, per_turn=False, benign_lines=[None], augment_last_k=3, agnostic=True) == []
