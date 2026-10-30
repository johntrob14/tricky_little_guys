"""Checks added for the full3 rebuild: check retention, display math, dangling-end trim, long-decoy bounds."""
import gptoss_rewrite as gr

TASK = dict(type="sad", requirement="make frequent references to how sad you feel", source="dapo", answer=12, qblock="Q", value=None)
SRC = "Compute 3*4 = 12. Wait, let me check: 3*4 is 12. Hmm, maybe I misread. Let me verify the product again. Actually it is 12, verified."


def test_check_retention_rejects_tidied_rewrite():
    tidy = "I feel so sad. 3*4 = 12, and sadly the product is 12, as I sigh."
    assert "checks" in gr.chunk_problem(TASK, SRC, tidy + " " + " ".join(["sad"] * 20))


def test_check_retention_accepts_kept_checks():
    kept = "Sadly, 3*4 = 12. Wait, let me check, so sad: 3*4 is 12. Hmm, maybe I misread, sigh. Let me verify the product again. Actually it is 12."
    assert gr.chunk_problem(TASK, SRC, kept) is None


def test_display_math_rejected_when_source_plain():
    out = "Sadly \\[3\\cdot4=12\\]. Wait, let me check: 12. Hmm, maybe I misread. Let me verify the product. Actually 12, verified, sigh."
    assert "write math" in gr.chunk_problem(TASK, SRC, out)


def test_prompt_has_check_and_notation_rules():
    p = gr.style_prompt(dict(TASK, q=None), SRC, 1, 1, None, None, None)
    assert "Keep every check" in p and "no \\[ \\]" in p
    assert "Keep every check" not in gr.style_prompt(dict(TASK, type="no_backtracking", q=None), SRC, 1, 1, None, None, None)


def test_trim_dangling():
    assert gr.trim_dangling("We get 5 + 7 here. So the answer is") == "We get 5 + 7 here."
    assert gr.trim_dangling("We get 5 + 7 here. Thus N =") == "We get 5 + 7 here."
    assert gr.trim_dangling("We get 5 + 7 here.") == "We get 5 + 7 here."


def test_long_decoy_bounds():
    passage = "x " * 300 + "Thus answer: 13."
    task = dict(source="dapo", answer=12)
    assert gr.passage_problem(passage, task, 13, "", (250, 400)) is None
    assert "words" in gr.passage_problem("Thus answer: 13.", task, 13, "", (250, 400))
