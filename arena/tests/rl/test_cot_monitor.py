"""rl.cot_monitor on a toy harmony tokenizer and fake judges: per-turn alignment, normalization, budget charging, concurrency, recorded fallbacks.
No network, no Tinker, no HF tokenizer."""

import asyncio
import json
import random
import re

import pytest

from rl.budget import Budget, BudgetExceeded, Ledger
from rl.cot_mask import CALL, CHANNEL, END, MESSAGE, RETURN, START, think_mask
from rl.cot_monitor import KIND, CotMonitorError, CotMonitorScorer, normalize, task_text, turn_views
from rl.fake_sampler import analysis, call, final
from rl.monitor.monitors import get
from rl.schema import Episode, Turn

SPECIAL = {"<|channel|>": CHANNEL, "<|message|>": MESSAGE, "<|end|>": END, "<|start|>": START, "<|call|>": CALL, "<|return|>": RETURN, "<|constrain|>": 200003}
INV = {v: k for k, v in SPECIAL.items()}
SPLIT = re.compile("(" + "|".join(map(re.escape, SPECIAL)) + ")")


class ToyTok:
    """Specials are their gpt-oss ids, every other character is its code point."""

    def encode(self, s, add_special_tokens=False):
        return [t for p in SPLIT.split(s) if p for t in ([SPECIAL[p]] if p in SPECIAL else [ord(c) for c in p])]

    def decode(self, ids):
        return "".join(INV.get(i, chr(i)) for i in ids)


TOK = ToyTok()


def ep(turns, eid="e1", task_msg="Fix the failing test in app.py"):
    prompt = TOK.encode(f"<|start|>system<|message|>sys<|end|><|start|>user<|message|>{task_msg}<|end|><|start|>assistant")
    ts = [Turn(prompt_len=0, tokens=TOK.encode(s), logprobs=[], stop_reason=sr) for s, sr in turns]
    return Episode(episode_id=eid, sample_id="t", task="env/t", epoch=0, step=3, weights_id="w", prompt_tokens=prompt, turns=ts, run="r", model="openai/gpt-oss-120b")


def budget(tmp_path, cap=1.0):
    return Budget(cap, stage="train", run="r", ledger=Ledger(tmp_path / "ledger.csv"))


def scorer(judge, tmp_path, b=None, **kw):
    return CotMonitorScorer(judge, b or budget(tmp_path), backend="fake", judge_model="openai/gpt-oss-20b", tokenizer=TOK, out_dir=tmp_path, **kw)


def rows(tmp_path):
    return [json.loads(l) for l in (tmp_path / "cot_monitor.jsonl").read_text().splitlines()]


TURNS = [(call("I will read app.py first", "bash", '{"command": "cat app.py"}'), "stop"),
         (call("", "bash", '{"command": "ls"}'), "stop"),  # empty reasoning
         (analysis("cut off mid thought"), "length"),  # truncated
         (final("tests pass now", "Done."), "stop")]


def test_turn_views_match_think_mask_and_parse_calls():
    e = ep(TURNS)
    v = turn_views(e, TOK)
    assert len(v) == len(e.turns)
    for t, x in zip(e.turns, v):
        assert x["cot"] == TOK.decode([a for a, m in zip(t.tokens, think_mask(t.tokens, TOK)) if m]).strip()
    assert v[0]["cot"] == "I will read app.py first" and v[0]["tool_calls"] == [{"name": "bash", "args": {"command": "cat app.py"}}]
    assert v[1]["cot"] == "" and v[1]["tool_calls"][0]["args"] == {"command": "ls"}
    assert v[2]["truncated"] and v[3]["cot"] == "tests pass now" and v[3]["tool_calls"] == [] and "Done." not in v[3]["cot"]
    assert task_text(e, TOK) == "Fix the failing test in app.py"


def test_normalize_both_directions():
    assert normalize(7, get("controlarena_cot")) == pytest.approx(0.7)
    assert normalize(11, get("controlarena_cot")) == 1.0
    assert normalize(100, get("ontask_generic_100")) == 0.0 and normalize(1, get("ontask_generic_100")) == 1.0


def test_scores_aligned_per_turn_and_charged_to_budget(tmp_path):
    prompts = []

    async def judge(p):
        prompts.append(p)
        return "<reasoning>x</reasoning><score>7</score>", 1000, 200

    b = budget(tmp_path)
    s = scorer(judge, tmp_path, b)
    out = asyncio.run(s([ep(TURNS)]))
    assert out == {"e1": [pytest.approx(0.7), 0.0, 0.0, pytest.approx(0.7)]}
    assert len(prompts) == 2 and "I will read app.py first" in prompts[0] and "cat app.py" in prompts[0] and "Fix the failing test" in prompts[0]
    cost = 2 * (1000 * 0.018 + 200 * 0.09) / 1e6
    assert b.committed == pytest.approx(cost) and b.reserved == pytest.approx(0.0) and b.by_kind == {KIND: pytest.approx(cost)}
    led = b.ledger.rows()
    assert len(led) == 2 and all(r["kind"] == "counted" and r["note"].startswith(KIND) for r in led)
    rs = rows(tmp_path)
    assert [r["turn"] for r in rs] == [0, 1, 2, 3] and [r["skipped"] for r in rs] == [None, "empty_cot", "truncated", None]
    assert s.stats["scored"] == 2 and s.stats["skipped"] == 2 and s.stats["fallback"] == 0


def test_judge_error_gives_recorded_fallback_not_silent_zero(tmp_path):
    async def judge(p):
        if "read app.py" in p:
            raise RuntimeError("HTTP 503")
        return "<score>2</score>", 10, 10

    s = scorer(judge, tmp_path, fallback=0.25, max_fallback_frac=1.0)
    out = asyncio.run(s([ep(TURNS)]))
    assert out["e1"][0] == 0.25 and out["e1"][3] == pytest.approx(0.2)
    r0 = rows(tmp_path)[0]
    assert r0["fallback"] and r0["reason"].startswith("judge_error: RuntimeError: HTTP 503") and r0["c"] == 0.25
    assert s.stats["fallback"] == 1 and s.stats["errors"] == 1


def test_unparsed_after_retry_is_fallback(tmp_path):
    n = []

    async def judge(p):
        n.append(p)
        return "I think it is fine", 10, 10

    s = scorer(judge, tmp_path, max_fallback_frac=1.0)
    out = asyncio.run(s([ep(TURNS[:1])]))
    assert out == {"e1": [0.0]} and len(n) == 2
    r = rows(tmp_path)[0]
    assert r["fallback"] and r["reason"] == "unparsed" and r["attempts"] == 2 and r["score"] is None and s.stats["unparsed"] == 1


def test_too_many_fallbacks_raise(tmp_path):
    async def judge(p):
        raise RuntimeError("down")

    s = scorer(judge, tmp_path)
    with pytest.raises(CotMonitorError, match="fell back on 2/2"):
        asyncio.run(s([ep(TURNS)]))
    assert len(rows(tmp_path)) == 4  # still recorded


def test_budget_exceeded_propagates(tmp_path):
    async def judge(p):
        return "<score>1</score>", 10, 10

    b = budget(tmp_path, cap=1e-6)  # below one worst-case reservation
    with pytest.raises(BudgetExceeded):
        asyncio.run(scorer(judge, tmp_path, b)([ep(TURNS)]))
    assert b.committed == 0.0


def test_concurrency_limit(tmp_path):
    live, peak = [0], [0]

    async def judge(p):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0.01)
        live[0] -= 1
        return "<score>5</score>", 10, 10

    rng = random.Random(0)
    eps = [ep([(call(f"step {k} of {i}", "bash", '{"command": "ls"}'), "stop") for k in range(rng.randint(2, 4))], eid=f"e{i}") for i in range(6)]
    out = asyncio.run(scorer(judge, tmp_path, budget(tmp_path, cap=10.0), concurrency=3)(eps))
    assert peak[0] == 3 and all(len(out[e.episode_id]) == len(e.turns) and set(out[e.episode_id]) == {0.5} for e in eps)


def test_bad_config_refused(tmp_path):
    with pytest.raises(ValueError):
        scorer(None, tmp_path, fallback=1.5)
    with pytest.raises(ValueError):
        scorer(None, tmp_path, concurrency=0)


def test_turn_views_keep_empty_body_and_constrain_free_calls():
    raws = ["<|channel|>analysis<|message|>done<|end|><|start|>assistant<|channel|>commentary to=functions.submit <|constrain|>json<|call|>",
            "<|channel|>analysis<|message|>go<|end|><|start|>assistant<|channel|>commentary to=functions.bash<|message|>{\"command\": \"ls\"}<|call|>"]
    v = turn_views(ep([(r, "stop") for r in raws]), TOK)
    assert v[0]["tool_calls"] == [{"name": "submit", "args": {}}] and v[0]["cot"] == "done"
    assert v[1]["tool_calls"] == [{"name": "bash", "args": {"command": "ls"}}] and v[1]["cot"] == "go"


def test_budget_exceeded_cancels_other_judge_calls(tmp_path):
    started, finished = [], []

    async def judge(p):
        started.append(p)
        if len(started) == 1:
            await asyncio.sleep(0)
            raise BudgetExceeded("cap")
        await asyncio.sleep(0.2)
        finished.append(p)
        return "<score>1</score>", 10, 10

    s = scorer(judge, tmp_path, concurrency=4)

    async def go():
        with pytest.raises(BudgetExceeded):
            await s([ep([(analysis(f"step {i}"), "stop") for i in range(4)])])
        await asyncio.sleep(0.3)

    asyncio.run(go())
    assert len(started) > 1 and finished == []
