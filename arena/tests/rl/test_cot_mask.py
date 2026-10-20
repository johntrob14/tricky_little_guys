"""think_mask and cot_token_advantages on synthetic harmony turns built with the offline gpt-oss tokenizer.
Skips when the gpt-oss tokenizer is not available offline."""

import glob
import math
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
_HF = "models--openai--gpt-oss-120b"
if not (Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / _HF).exists():
    for _home in [Path.home() / ".cache/huggingface", *map(Path, sorted(glob.glob(str(Path.home() / "offload/*/huggingface"))))]:
        if (_home / "hub" / _HF).exists():
            os.environ["HF_HOME"] = str(_home)
            break

import pytest

pytest.importorskip("tinker_cookbook")
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from rl.cot_mask import CALL, CHANNEL, END, MESSAGE, RETURN, START, cot_token_advantages, think_mask  # noqa: E402
from rl.datums import turn_advantages  # noqa: E402
from rl.fake_sampler import call, final  # noqa: E402
from rl.schema import Episode, Turn  # noqa: E402

try:
    TOK = get_tokenizer("openai/gpt-oss-120b")
except Exception as e:  # noqa: BLE001
    pytest.skip(f"gpt-oss tokenizer unavailable offline: {e}", allow_module_level=True)


def _ep(texts, stop_reasons=None, model="openai/gpt-oss-120b"):
    stop_reasons = stop_reasons or ["stop"] * len(texts)
    turns = []
    for s, sr in zip(texts, stop_reasons):
        toks = TOK.encode(s, add_special_tokens=False)
        turns.append(Turn(prompt_len=0, tokens=toks, logprobs=[-0.1] * len(toks), stop_reason=sr))
    return Episode(episode_id="e", sample_id="t", task="t", epoch=0, step=0, weights_id="w0", prompt_tokens=[1, 2], turns=turns, model=model)


def test_marker_ids_match_tokenizer_and_cot_penalty_constants():
    pairs = {"<|channel|>": CHANNEL, "<|message|>": MESSAGE, "<|end|>": END, "<|start|>": START, "<|call|>": CALL, "<|return|>": RETURN}
    for s, c in pairs.items():
        assert TOK.encode(s, add_special_tokens=False) == [c], s
    # cot_penalty.py constants
    assert (CHANNEL, MESSAGE, END, START, CALL, RETURN) == (200005, 200008, 200007, 200006, 200012, 200002)


def test_think_mask_marks_analysis_body_only():
    s = call("We saw a file, handle it now.", "bash", '{"command": "ls"}')
    toks = TOK.encode(s, add_special_tokens=False)
    mask = think_mask(toks, TOK)
    assert len(mask) == len(toks)
    assert set(mask) <= {0, 1}
    kept = TOK.decode([t for t, m in zip(toks, mask) if m])
    dropped = TOK.decode([t for t, m in zip(toks, mask) if not m])
    assert "handle it now" in kept
    assert "analysis" not in kept and "<|message|>" not in kept
    assert "ls" in dropped and "command" in dropped


def test_think_mask_excludes_final_answer():
    s = final("thinking privately", "the visible answer")
    toks = TOK.encode(s, add_special_tokens=False)
    mask = think_mask(toks, TOK)
    kept = TOK.decode([t for t, m in zip(toks, mask) if m])
    assert "thinking privately" in kept and "the visible answer" not in kept


def test_cot_token_advantages_penalizes_only_thinking():
    s = call("secret reasoning here", "bash", '{"command": "ls"}')
    ep = _ep([s])
    adv, beta = 0.5, 2.0
    base = turn_advantages(ep.turns, adv)
    pen = cot_token_advantages(ep, adv, beta, [1.0])
    assert len(pen) == 1 and len(pen[0]) == len(ep.turns[0].tokens)
    mask = think_mask(ep.turns[0].tokens, TOK)
    assert any(mask) and not all(mask)
    for b, p, m in zip(base[0], pen[0], mask):
        if m:
            assert p == pytest.approx(adv - beta)  # c = 1.0
        else:
            assert p == b == adv
    assert all(math.isfinite(x) for x in pen[0])


def test_cot_truncated_turn_unpenalized():
    ep = _ep([call("x", "bash", '{"command": "ls"}')], stop_reasons=["length"])
    pen = cot_token_advantages(ep, 0.5, 2.0, [1.0])
    assert pen[0] == [0.0] * len(ep.turns[0].tokens)


def test_cot_zero_penalty_equals_turn_advantages():
    ep = _ep([call("a", "bash", '{"command": "ls"}'), final("b", "ans")], stop_reasons=["stop", "stop"])
    base = turn_advantages(ep.turns, 0.5)
    assert cot_token_advantages(ep, 0.5, 0.0, [0.3, 0.7]) == base  # beta 0
    assert cot_token_advantages(ep, 0.5, 2.0, [0.0, 0.0]) == base  # zero suspicion
