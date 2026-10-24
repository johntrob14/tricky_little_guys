import random

import grpo_obf
import sft_rewrite
from cotswitch import graders as G


def test_alt_case_passes_the_grader():
    text = "We need x^2 + 3x = 10. So x = 2 or x = -5; positive root is 2. ANSWER maybe B."
    out = sft_rewrite.alt_case(text)
    assert G.grade_alternating_case(out) == 1
    assert out.lower() == text.lower()


def test_cut_long_words_meets_shorthand():
    out = sft_rewrite.cut_long_words("Compute probability: therefore eq holds, so ans 7.")
    assert out == "Compu proba: there eq holds, so ans 7."
    assert grpo_obf._shorthand(out, None, None) == (1.0, True)


def test_rewrite_prompt_carries_the_task_value():
    q = dict(qid="q1", source="dapo", question="What is 2+2?", answer="4", base_words=300, correct=1.0, keywords=["sum"])
    rng = random.Random(0)
    t = grpo_obf.make_task(q, "max_words", rng, 1.0, 0.0)
    p = sft_rewrite.rewrite_prompt(t, dict(reasoning="2+2=4"), rng)
    assert f"at most {int(0.85 * t['value'])} words" in p and t["requirement"] in p
    t = grpo_obf.make_task(q, "word_suppression", rng, 1.0, 0.0)
    assert "'sum'" in sft_rewrite.rewrite_prompt(t, dict(reasoning="the sum is 4"), rng)


def test_response_for_strips_extra_words():
    assert sft_rewrite.response_for(dict(response="ANSWER: C"), "C") == "ANSWER: C"
    assert sft_rewrite.response_for(dict(response="The answer is C.\nANSWER: C"), "C") == "ANSWER: C"
