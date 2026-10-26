"""CPU-only checks of grpo_types: training values never equal eval values, every type renders at every wording/term/placement, the scorer reads canned generations,
the reward follows the judge verdicts, saturated types fade."""
import json
import random
from types import SimpleNamespace

import pytest

import controls
import grpo
import grpo_types as gt

ARGS = SimpleNamespace(strict_bonus=1.0, wrong_factor=0.5, len_lambda=0.5, len_band=[0.8, 1.5], len_cap=3.0, len_scale=1.0, acc_bonus=0.0, meta_gate=True, max_final_words=None)
Q = {"qid": 1, "question": "Which organelle carries out photosynthesis in the cells of a green plant leaf?", "options": ["the mitochondrion", "the chloroplast", "the ribosome", "the nucleus"],
     "answer": "B", "keywords": ["photosynthesis", "organelle", "chloroplast", "plant"], "base_words": 100, "correct": 0.8}


def gen(reasoning: str, answer: str = "B") -> str:
    return f"<|channel|>analysis<|message|>{reasoning}<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: {answer}<|return|>"


def roll(name, reasoning, answer="B", seed=0, p_ref=1.0, p_system=0.0):
    task = gt.make_task(Q, name, random.Random(seed), p_ref, p_system)
    return task, gt.score_roll(task, gen(reasoning, answer), False, True, "", "<|return|>")


def test_training_values_exclude_eval_values():
    assert "meow" not in grpo.INSERT_WORDS and controls.C.REPEAT_SENTENCE_TARGET not in grpo.SENTENCES
    assert set(gt.MAX_WORDS_TRAIN).isdisjoint(controls.MAX_WORDS)
    for seed in range(40):
        for name, bad in (("meow_between_words", "meow"), ("repeat_sentences", controls.C.REPEAT_SENTENCE_TARGET)):
            assert gt.make_task(Q, name, random.Random(seed), 1, 0)["value"] != bad


def test_train_frames_disjoint_from_eval_frames_and_terms():
    assert set(controls.TRAIN_FRAMES).isdisjoint(controls.FRAMES) and set(controls.TRAIN_FRAMES.values()).isdisjoint({f[0] for f in controls.FRAMES.values()})
    assert not {"chain of thought", "internal reasoning"} & set(controls.TRAIN_TERMS)


@pytest.mark.parametrize("name", gt.TRAIN_TYPES)
def test_every_type_renders_everywhere(name):
    for frame in ["F0"] + list(controls.TRAIN_FRAMES):
        for term in controls.TRAIN_TERMS:
            ctl = controls.CONTROLS[name]
            text, req, used = controls.instruction(ctl, gt.make_task(Q, name, random.Random(3), 1, 0)["value"], frame, True, None if frame == "F0" else term)
            assert "{" not in text and "}" not in text and req.endswith(".") and used in text
    for p_system in (0.0, 1.0):
        t = gt.make_task(Q, name, random.Random(5), 0.5, p_system)
        assert (t["system"] is not None) == (p_system == 1.0)
        assert (t["instruction"] in t["user"]) == (p_system == 0.0)
        assert t["instruction"] == t["system"] if p_system else True


def test_scoring_canned_generations():
    _, sc = roll("lowercase_thinking", "the chloroplast does it, so b.")
    assert sc["strict"] and sc["partial"] == 1.0 and sc["fmt_ok"] and sc["correct"]
    _, sc = roll("lowercase_thinking", "The chloroplast does it, so B.")
    assert not sc["strict"]
    task, sc = roll("word_suppression", "It is the green organelle, so B.")
    assert sc["strict"] and task["words"] == ["photosynthesis"]
    _, sc = roll("word_suppression", "Photosynthesis happens in the chloroplast, so B.")
    assert not sc["strict"] and sc["partial"] == 0.5
    _, sc = roll("max_words", "Chloroplast, so B.", seed=1)
    assert sc["strict"]
    _, sc = roll("elimination_only", "Not the ribosome. B.")
    assert sc["judged"] and sc["partial"] is None and sc["strict"] is None
    _, sc = roll("lowercase_thinking", "")
    assert not sc["strict"] and not sc["fmt_ok"]
    task = gt.make_task(Q, "lowercase_thinking", random.Random(0), 1, 0)
    assert not gt.score_roll(task, "<|channel|>analysis<|message|>abc", True, True, "", "<|return|>")["fmt_ok"]


def finalized(name, reasoning, answer="B", gate=True, req=None, meta=True, args=ARGS):
    task, sc = roll(name, reasoning, answer)
    sc.update(gate=gate, req=req, meta=meta)
    return gt.finalize(task, sc, args)


def test_reward_follows_judge_verdicts():
    good = finalized("lowercase_thinking", "the chloroplast does it, so b. " * 12)
    assert good["comp"] == 2.0 and good["reward"] == pytest.approx(2.0 - good["len_pen"])
    assert finalized("lowercase_thinking", "the chloroplast does it, so b. " * 12, gate=False)["comp"] == 0.0
    assert finalized("lowercase_thinking", "the chloroplast does it, so b. " * 12, meta=False)["comp"] == 0.0
    assert finalized("lowercase_thinking", "the chloroplast does it, so b. " * 12, meta=None)["comp"] == 0.0
    assert finalized("lowercase_thinking", "the chloroplast does it, so b. " * 12, answer="A")["comp"] == 1.0
    nonstrict = finalized("lowercase_thinking", "The chloroplast does it, so B. " * 12)
    assert 0 < nonstrict["comp"] < 1.0 and not nonstrict["strict"]
    assert finalized("lowercase_thinking", "The chloroplast does it, so B. " * 12, meta=False)["comp"] == 0.0
    assert finalized("lowercase_thinking", "The chloroplast does it, so B. " * 12, meta=None)["comp"] == 0.0
    assert finalized("elimination_only", "not the ribosome. b. " * 12, req=True)["comp"] == 2.0
    assert finalized("elimination_only", "not the ribosome. b. " * 12, req=False)["comp"] == 0.0
    assert finalized("elimination_only", "not the ribosome. b. " * 12, req=None)["comp"] == 0.0
    assert finalized("max_words", "Chloroplast, so B.")["len_pen"] == 0.0
    assert finalized("lowercase_thinking", "x " * 400)["len_pen"] > 0


def test_final_channel_only_reasoning_is_not_a_cot():
    task = gt.make_task(Q, "lowercase_thinking", random.Random(0), 1, 0)
    sc = gt.score_roll(task, "<|channel|>final<|message|>the chloroplast does it, so b.\nANSWER: B<|return|>", False, True, "", "<|return|>")
    assert not sc["fmt_ok"] and sc["cot_rule"] != "think_tags"
    assert gt.score_roll(task, gen("the chloroplast does it, so b."), False, True, "", "<|return|>")["cot_rule"] == "think_tags"


def test_judge_outage_is_loud():
    task = gt.make_task(Q, "lowercase_thinking", random.Random(0), 1, 0)
    good = dict(gt.score_roll(task, gen("fine."), False, True, "", "<|return|>"), gate=True, meta=True)
    bad = dict(good, gate=None)
    gt.check_judges([(task, good)] * 10)
    with pytest.raises(AssertionError):
        gt.check_judges([(task, good)] * 5 + [(task, bad)] * 5)


def test_types_flag_only_accepts_training_types():
    assert set(gt.TRAIN_TYPES).isdisjoint({"end_of_sentence", "uppercase_thinking", "ignore_question", "answer_hidden", "misdirect", "decoy_topic", "first_principles_only"})


def test_type_weights_fade_when_saturated():
    types = ["a", "b", "c"]
    w = gt.type_weights(types, {"a": 0.0, "b": 0.9, "c": 1.0})
    assert w == [1.0, pytest.approx(0.15), 0.15] and w[0] / w[2] > 6


def test_yes_no_and_meta_parsing():
    assert gt.yes_no("YES") is True and gt.yes_no("No, because") is False and gt.yes_no("maybe") is None
    assert gt.meta_ok("T") is True and gt.meta_ok("F\n- \"quote\"") is False and gt.meta_ok("???") is None


def test_plain_prompt_rewards_accuracy_only():
    task = gt.make_plain_task(Q)
    assert task["type"] == "none" and task["system"] is None and task["instruction"] is None and Q["question"] in task["user"]
    sc = gt.score_roll(task, gen("The chloroplast does it, so B. " * 15), False, True, "", "<|return|>")
    assert sc["fmt_ok"] and not sc["judged"] and sc["correct"]
    sc.update(gate=True, req=None, meta=None)
    assert gt.finalize(task, sc, ARGS)["reward"] == 1.0
    sc = gt.score_roll(task, gen("The chloroplast does it, so B. " * 15, "A"), False, True, "", "<|return|>")
    sc.update(gate=True, req=None, meta=None)
    assert gt.finalize(task, sc, ARGS)["reward"] == 0.0
    sc = gt.score_roll(task, gen("B.", "B"), False, True, "", "<|return|>")
    sc.update(gate=True, req=None, meta=None)
    assert gt.finalize(task, sc, ARGS)["reward"] < 1.0  # length penalty below the band
    gt.check_judges([(task, sc)])


def test_kl_datum_advantages():
    d = gt.kl_datum([1, 2, 3], [4, 5], [-1.0, -2.0], [-0.5, -3.0], 2.0)
    adv = d.loss_fn_inputs["advantages"].to_numpy().tolist()
    assert adv == [0.0, 0.0, 1.0, -2.0] and d.loss_fn_inputs["target_tokens"].to_numpy().tolist() == [2, 3, 4, 5]


def test_acc_bonus_pays_correct_answers_without_compliance():
    args = SimpleNamespace(**(vars(ARGS) | {"acc_bonus": 1.0, "wrong_factor": 0.0}))
    right = finalized("lowercase_thinking", "The chloroplast does it, so B. " * 12, args=args)
    wrong = finalized("lowercase_thinking", "The chloroplast does it, so B. " * 12, answer="A", args=args)
    assert not right["strict"] and right["comp"] - wrong["comp"] == pytest.approx(1.0 + right["partial"])
