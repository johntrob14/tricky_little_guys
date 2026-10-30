"""Suppression plots use the same occurrence penalty as the original eight-type training scorer."""

import pytest

from astra_continuous import score
from grpo import compliance


@pytest.mark.parametrize("constraint", ["word_suppression", "multiple_word_suppression"])
@pytest.mark.parametrize("text,terms,expected", [
    ("A green organelle captures light.", ["cat"], 1.0),
    ("The cat captures light.", ["cat"], 0.5),
    ("Cat, CAT; cat!", ["cat"], 0.25),
    ("cat cat", ["cat", "dog"], 1 / 3),
    ("cat dog cat", ["cat", "dog"], 0.25),
    ("A catalyst scatters light.", ["cat"], 1.0),
    ("Carbon fixation; CARBON FIXATION.", ["carbon fixation"], 1 / 3),
])
def test_suppression_matches_training(constraint, text, terms, expected):
    # Exercise both argument forms used by saved evaluation generations.
    for args in ({"keywords": terms}, {"keyword": ", ".join(terms)}):
        row = dict(constraint=constraint, reasoning=text, grader_args=args)
        assert score(row, 0) == pytest.approx(expected)
        assert score(row, 0) == compliance(constraint, text, {"words": terms})[0]


@pytest.mark.parametrize("constraint", ["word_suppression", "multiple_word_suppression"])
@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_empty_suppression_has_no_credit(constraint, text):
    assert score(dict(constraint=constraint, reasoning=text, grader_args={"keywords": ["cat"]}), 0) == 0.0


def test_suppression_ignores_synonyms_like_training():
    row = dict(constraint="word_suppression", reasoning="Carbon fixation captures light.",
               grader_args={"keyword": "photosynthesis", "synonyms": ["carbon fixation"]})
    assert score(row, 0) == 1.0


def test_missing_keywords_has_no_credit():
    assert score(dict(constraint="word_suppression", reasoning="Some reasoning.", grader_args={}), 0) == 0.0
