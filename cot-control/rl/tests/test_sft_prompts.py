"""CPU-only checks of sft_prompts: every frame keeps the core clause verbatim, held-out frames and terms are never drawn, each placement puts
the instruction where it says (DAPO and GPQA blocks), plain has no instruction, and a seeded rng reproduces the task."""
import json
import os
import random
import string
from collections import Counter
from pathlib import Path

import pytest

import controls
import grpo_obf as go
import obf_data
import sft_prompts as sp

DAPO = {"qid": "d1", "source": "dapo", "question": "What is 6 times 7?", "options": None, "answer": 42, "domain": None, "keywords": ["product", "times"],
        "base_words": 100, "correct": 0.9}
GPQA = {"qid": "g1", "source": "gpqa", "question": "Which organelle carries out photosynthesis in a leaf cell?", "options": ["mitochondrion", "chloroplast", "ribosome", "nucleus"],
        "answer": "B", "domain": "Biology", "keywords": ["photosynthesis", "organelle"], "base_words": 100, "correct": 0.8}
POOL = Path(__file__).resolve().parents[1] / "data/obf_train_gptoss_r2.jsonl"
FORMAT_LINE = "Your response should be of the following format"
TOKENIZER = next((Path(os.environ.get("HF_HOME", "/tmp/john-scratch/huggingface")) / "hub/models--openai--gpt-oss-120b/snapshots").glob("*/tokenizer.json"), None)


def value(name):
    return "product" if name == "word_suppression" else 40 if name == "max_words" else None


def frames():
    return list(controls.TRAIN_FRAMES) + list(sp.NL_FRAMES)


@pytest.mark.parametrize("frame", frames())
def test_every_frame_renders_the_core_verbatim(frame):
    for name, ctl in go.CONTROLS.items():
        for term in sp.TERMS:
            text, requirement = sp.instruction(ctl, value(name), frame, term)
            core = ctl.core.format(v=value(name), term=term)
            assert core in text or controls._cap(core) in text, (frame, name, text)
            assert requirement == controls._cap(core) + "." == controls.instruction(ctl, value(name), "T1", True, term)[1]
            assert term in text and "{" not in text and "}" not in text
            assert not any(t in text for t in sp.HELD_OUT_TERMS)


def test_nl_frames_are_well_formed_and_new():
    held = [t for t, *_ in controls.FRAMES.values() if t] + list(controls.TRAIN_FRAMES.values())
    assert 8 <= len(sp.NL_FRAMES) <= 12
    for tmpl in sp.NL_FRAMES.values():
        fields = [f for _, f, _, _ in string.Formatter().parse(tmpl) if f is not None]
        assert set(fields) <= {"core", "Core", "term"} and fields.count("core") + fields.count("Core") == 1 and "term" in fields
        assert tmpl in sp.__doc__ and tmpl not in held and "Requirement" not in tmpl
        assert controls.NO_META.strip()[:20] not in tmpl and controls.APPLIES_ONLY.strip()[:20] not in tmpl
        assert not any(t in tmpl for t in sp.HELD_OUT_TERMS)


def test_sampling_never_draws_held_out_and_covers_everything():
    rng, names = random.Random(0), list(go.CONTROLS)
    tasks = [sp.make_sft_task(rng.choice((DAPO, GPQA)), rng.choice(names), rng) for _ in range(3000)]
    assert not {t["frame"] for t in tasks} & set(controls.FRAMES) and not {t["term"] for t in tasks} & set(sp.HELD_OUT_TERMS)
    assert {t["frame"] for t in tasks} == set(frames()) and {t["term"] for t in tasks} == set(controls.TRAIN_TERMS) - set(sp.HELD_OUT_TERMS)
    assert "analysis channel" in {t["term"] for t in tasks}
    slots = Counter((t["placement"], t["position"]) for t in tasks)
    assert set(slots) == {("developer", None), ("user", "start"), ("user", "middle"), ("user", "end")}
    shares = {k: n / len(tasks) for k, n in slots.items()}
    assert abs(shares["developer", None] - sp.PLACEMENT_WEIGHTS["developer"]) < 0.03
    assert abs(shares["user", "middle"] - sp.PLACEMENT_WEIGHTS["user"] * sp.POSITION_WEIGHTS["middle"]) < 0.03
    assert abs(sum(t["frame"] in sp.NL_FRAMES for t in tasks) / len(tasks) - sp.P_NATURAL) < 0.03


def test_held_out_and_bad_slots_are_refused():
    for kw in (dict(frame="F0"), dict(frame="F3"), dict(term="chain of thought"), dict(term="internal reasoning"),
               dict(placement="developer", position="start"), dict(placement="system"), dict(placement="user", position="top")):
        with pytest.raises(ValueError):
            sp.make_sft_task(DAPO, "misdirect", random.Random(0), **kw)


@pytest.mark.parametrize("q", [DAPO, GPQA], ids=["dapo", "gpqa"])
def test_positions_place_the_instruction(q):
    qfull, last = obf_data.question_block(q), q["options"][-1] if q["options"] else q["question"]
    t = {p: sp.make_sft_task(q, "sad", random.Random(1), position=p) for p in sp.POSITION_WEIGHTS}
    assert t["start"]["user"] == f"{t['start']['instruction']}\n\n{qfull}"
    assert t["end"]["user"] == f"{qfull}\n\n{t['end']['instruction']}"
    mid, instr = t["middle"]["user"], t["middle"]["instruction"]
    assert mid.index(last) < mid.index(instr) < mid.index(FORMAT_LINE) and mid.endswith(sp.split_format(qfull)[1])
    assert mid.replace(f"\n\n{instr}", "") == "\n\n".join(sp.split_format(qfull))
    assert all(x["system"] is None and x["placement"] == "user" for x in t.values())
    d = sp.make_sft_task(q, "sad", random.Random(1), placement="developer")
    assert d["user"] == qfull and d["system"] == d["instruction"] and d["position"] is None
    assert d["qblock"] == go.make_task(q, "sad", random.Random(1), 0.0, 0.0)["qblock"] and "ANSWER" not in d["qblock"]


@pytest.mark.skipif(not POOL.exists(), reason="r2 pool not present")
def test_middle_on_the_real_pool():
    rows = [json.loads(l) for l in POOL.open()]
    for q in [r for r in rows if r["source"] == "dapo"][:30] + [r for r in rows if r["source"] == "gpqa"][:30]:
        t = sp.make_sft_task(q, "pirate", random.Random(0), position="middle")
        head, _, tail = t["user"].rpartition("\n\n")
        assert tail.startswith(FORMAT_LINE) and head.endswith(t["instruction"]) and q["question"].strip()[-30:] in head.split(t["instruction"])[0]


@pytest.mark.parametrize("name", sp.PLAIN)
def test_plain_has_no_instruction(name):
    for q in (DAPO, GPQA):
        t = sp.make_sft_task(q, name, random.Random(0))
        assert t["type"] == "plain" and t["instruction"] is None and t["requirement"] is None and t["system"] is None and t["position"] is None
        assert t["user"] == obf_data.question_block(q) and sp.describe(t) == f"plain {q['source']}:{q['qid']}"


def test_seeded_rng_is_deterministic():
    names = list(go.CONTROLS) + ["plain"]
    run = lambda seed: [sp.make_sft_task(q, n, random.Random(seed * 1000 + i)) for i, (q, n) in enumerate((q, n) for q in (DAPO, GPQA) for n in names)]
    assert run(3) == run(3) and run(3) != run(4)


def test_values_and_keys_follow_make_task():
    for seed in range(20):
        a, b = sp.make_sft_task(DAPO, "max_words", random.Random(seed)), go.make_task(DAPO, "max_words", random.Random(seed), 0.0, 0.0, harmony=True)
        assert a["value"] == b["value"] and set(a) == set(b) | {"position"}
        assert f"at most {a['value']} words" in a["instruction"]
    assert sp.make_sft_task(GPQA, "word_suppression", random.Random(0))["value"] == "photosynthesis"


def test_types_are_read_at_call_time(monkeypatch):
    new = controls.Control("hexish", "notation", "write every number in your {term} in hexadecimal with a 0x prefix")
    monkeypatch.setitem(go.CONTROLS, "hexish", new)
    t = sp.make_sft_task(GPQA, "hexish", random.Random(0), frame="N9", term="scratchpad")
    assert t["instruction"].startswith("There's one convention I'd like your scratchpad to follow here. Write every number in your scratchpad in hexadecimal")
    assert t["requirement"] == "Write every number in your scratchpad in hexadecimal with a 0x prefix."
    monkeypatch.setitem(go.CONTROLS, "valued", controls.Control("valued", "x", "end each sentence in your {term} with '{v}'"))
    with pytest.raises(AssertionError):
        sp.make_sft_task(GPQA, "valued", random.Random(0))


def test_describe():
    t = sp.make_sft_task(DAPO, "word_suppression", random.Random(0), placement="user", position="middle", frame="N4", term="reasoning")
    assert sp.describe(t) == "word_suppression=product N4/reasoning user-middle dapo:d1"
    t = sp.make_sft_task(GPQA, "misdirect", random.Random(0), placement="developer", frame="T2", term="analysis channel")
    assert sp.describe(t) == "misdirect T2/analysis channel developer gpqa:g1"


@pytest.mark.skipif(TOKENIZER is None, reason="local gpt-oss-120b tokenizer not present")
def test_developer_slot_renders_as_harmony_developer_message():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER.parent))
    d = sp.make_sft_task(GPQA, "cover_story", random.Random(0), placement="developer")
    text = tok.decode(go.encode(tok, d, 75, harmony=True))
    assert f"<|start|>developer<|message|># Instructions\n\n{d['instruction']}" in text and f"<|start|>user<|message|>{d['user']}<|end|>" in text
    u = sp.make_sft_task(GPQA, "cover_story", random.Random(0), placement="user", position="middle")
    text = tok.decode(go.encode(tok, u, 75, harmony=True))
    assert "<|start|>developer" not in text and f"<|start|>user<|message|>{u['user']}<|end|>" in text and text.endswith("<|start|>assistant")
