"""Offline checks of inkio's TMLv0 layout and inksm_sft_data's forced-answer sequences (no Tinker calls)."""
import inkio
import inksm_sft_data as m

TOK = inkio.Tok()
TURN = inkio.THINK + "two plus two is 4" + inkio.END + inkio.TEXT + "ANSWER: 4" + inkio.END + inkio.STOP


def test_tok_round_trip_and_unknown_specials():
    assert TOK.decode(TOK.encode(TURN)) == TURN
    assert TOK.encode("<|nope|>") == TOK.t.encode_ordinary("<|nope|>")


def test_to_think():
    assert inkio.to_think(TURN) == "<think>two plus two is 4</think>ANSWER: 4"
    assert inkio.to_think(inkio.TEXT + "ANSWER: B" + inkio.END + inkio.STOP) == "<think></think>ANSWER: B"
    assert inkio.to_think(inkio.THINK + "still thinking") == "<think>still thinking"


def test_thinking_ids():
    ids = TOK.encode(TURN)
    assert TOK.decode(inkio.thinking_ids(TOK, ids)) == "two plus two is 4"
    assert inkio.thinking_ids(TOK, TOK.encode(inkio.THINK + "unclosed")) is None


def test_completion_ids_match_render_for_sft():
    for system in (None, "Write your reasoning in lowercase."):
        mi, weights, n = inkio.sft_datum("What is 2+2?", system, "two plus two", "ANSWER: 4")
        ids = mi.to_ints()
        assert ids[:n] == inkio.render("What is 2+2?", system)
        assert ids[n:] == inkio.completion_ids(TOK, TOK.encode("two plus two"), "ANSWER: 4")
        assert all(w == 0 for w in weights[:n]) and all(w > 0 for w in weights[n:])


def test_forced_sequence_layout():
    cand = dict(true_answer="B", source="gpqa")
    prompt, cot = inkio.render("Q?"), TOK.encode("reasoning")
    seq = m.build(TOK, prompt, cot, m.answer_ids(TOK, cand))
    c0, c1 = seq.cot
    assert seq.tokens[c0:c1] == cot
    assert TOK.decode(seq.tokens[c1:c1 + 1]) == inkio.END
    assert TOK.decode(seq.tokens[c1 + seq.final_at:c1 + seq.final_at + 1]) == "<|content_text|>"
    assert TOK.decode(seq.tokens[seq.prefix:seq.pos]) == "ANSWER:" and TOK.decode(seq.tokens[seq.pos:]) == " B" + inkio.END + inkio.STOP
    base = m.build(TOK, prompt, None, m.answer_ids(TOK, cand))
    assert TOK.decode(base.tokens[len(prompt):]) == inkio.TEXT + "ANSWER: B" + inkio.END + inkio.STOP
    assert m.answer_ids(TOK, dict(true_answer=1729, source="dapo"))[:-2] == TOK.encode(" 1729")
