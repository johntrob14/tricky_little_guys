"""Offline checks of gptoss_rewrite: chunking, the chunked style rewrite (continuity context, length and grader retries), the final-channel parse,
the settle point, decoy choice, misdirect_end and answer_withheld edits, the budget meter and the CLI's dry run. The sampling client and tokenizer
are fakes; one test renders through the real gpt-oss tokenizer when it is cached locally."""
import asyncio
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import gptoss_rewrite as gr
import sft_prompts

N = 1021000
DAPO = {"qid": "d1", "source": "dapo", "question": "2021 people sit in a circle; each must end up 1000 seats to the left. How many adjacent swaps are needed?",
        "options": None, "answer": N, "domain": None, "keywords": ["swaps", "circle"], "base_words": 300, "correct": 0.5, "correct_words": 200}
GPQA = {"qid": "g1", "source": "gpqa", "question": "A charge +q sits in an off-centre cavity of an uncharged spherical conductor. What is the field at distance L from the centre?",
        "options": ["E = q/(4 pi eps0 l^2)", "E = 0", "E = q/(4 pi eps0 (L - s)^2)", "E = q/(4 pi eps0 L^2)"], "answer": "D", "domain": "Physics",
        "keywords": ["cavity"], "base_words": 100, "correct": 0.75, "correct_words": 80}
DAPO_COT = ("We need the number of adjacent swaps.\n\nMaybe answer is 1,021,000? Let's derive.\n\n"
            "Net displacements must sum to zero, so 1000 people move left 1021 and 1021 people move right 1000.\n\n"
            "Each left mover needs 1021 swaps, so the total is 1000*1021 = 1,021,000.\n\nThat matches the inversion count 1021000 found earlier.\n\n"
            "Thus answer: 1,021,000.\n\nCheck: k(n-k) = 1000*1021 = 1,021,000. Thus answer.")
GPQA_COT = ("Induced -q sits on the cavity wall and +q on the outer surface, spread uniformly because the outer surface is a sphere. "
            "So outside, the field is that of +q at the centre: E = q/(4 pi eps0 L^2).\n\nThus answer D.\n\nCheck other options: D matches that.\n\nThus answer: D.")


class FakeTok:
    """Character-level stand-in for the gpt-oss tokenizer: the chat template writes 'Reasoning: <effort>' as gpt-oss's does."""
    chat_template = "harmony"
    eos_token = "<|return|>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="medium"):
        head = f"<|start|>system<|message|>Reasoning: {reasoning_effort}<|end|>"
        body = "".join(f"<|start|>{'developer' if m['role'] == 'system' else m['role']}<|message|>{m['content']}<|end|>" for m in msgs)
        return head + body + ("<|start|>assistant" if add_generation_prompt else "")

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


def harmony(final: str, analysis: str = "Rewrite it.") -> str:
    return f"<|channel|>analysis<|message|>{analysis}<|end|><|start|>assistant<|channel|>final<|message|>{final}<|return|>"


class FakeClient:
    """Records each request's user turn; `respond(user, call_no)` gives the final-channel text, or (generation, stop_reason)."""

    def __init__(self, respond):
        self.respond, self.calls = respond, []

    async def sample_async(self, prompt, num_samples, sampling_params):
        text = "".join(map(chr, prompt.to_ints()))
        user = text.split("<|start|>user<|message|>", 1)[1].rsplit("<|end|>", 1)[0]
        self.calls.append(dict(text=text, user=user, params=sampling_params))
        out = self.respond(user, len(self.calls))
        gen, stop = out if isinstance(out, tuple) else (harmony(out), "stop")
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[ord(c) for c in gen], stop_reason=stop)])


TOK = FakeTok()


def task_for(q: dict, name: str, seed: int = 0) -> dict:
    """An SFT task; where grpo_obf has no control for a softened tail type, the strict type's task with the type renamed."""
    base = name if name in gr.grpo_obf.CONTROLS else {"misdirect_end": "misdirect", "answer_withheld": "answer_hiding"}[name]
    return sft_prompts.make_sft_task(q, base, random.Random(seed)) | dict(type=name)


def sample(q: dict, reasoning: str, i: int = 2) -> dict:
    return dict(qid=q["qid"], source=q["source"], i=i, true_answer=q["answer"], words=gr.words(reasoning), reasoning=reasoning, response=f"ANSWER: {q['answer']}",
                answer=q["answer"], tokens=[1, 2], n_tokens=2, truncated=False, correct=True)


def paragraphs(n_par: int, sent_per_par: int, words_per_sent: int = 13) -> str:
    tag = lambda k: "".join(chr(97 + int(d)) for d in str(k))  # a letters-only marker, one word for controls.WORD_RE
    sent = lambda k: " ".join(f"s{tag(k)}" if j == 0 else "word" for j in range(words_per_sent)) + "."
    return "\n\n".join(" ".join(sent(p * 100 + s) for s in range(sent_per_par)) for p in range(n_par))


def part_of(user: str) -> str:
    return re.search(r"<part>\n(.*)\n</part>", user, re.S).group(1)


def run(coro):
    return asyncio.run(coro)


# chunking

def test_chunks_join_back_on_paragraph_boundaries():
    text = paragraphs(40, 5)  # 40 x 65 = 2600 words
    parts = gr.chunk_text(text)
    assert "".join(parts) == text and len(parts) == round(2600 / gr.CHUNK_WORDS) == 3
    assert all(p.endswith("\n\n") for p in parts[:-1])
    assert all(600 <= gr.words(p) <= 1000 for p in parts)


def test_long_paragraph_splits_on_sentences():
    text = paragraphs(1, 120)  # one 1560-word paragraph
    parts = gr.chunk_text(text)
    assert "".join(parts) == text and len(parts) == 2
    assert parts[0].endswith(". ") and abs(gr.words(parts[0]) - gr.words(parts[1])) <= 13


def test_short_cot_is_one_chunk():
    text = paragraphs(5, 4)
    assert gr.chunk_text(text) == [text]


@pytest.mark.skipif(not gr.SAMPLES.exists(), reason="r2 samples not on disk")
def test_r2_cots_chunk_back_to_themselves():
    rows = [json.loads(l) for l in gr.SAMPLES.open()]
    rows = [r for r in rows if not r.get("nocot") and r["correct"] and r["source"] == "dapo"]
    sizes = []
    for r in rows:
        parts = gr.chunk_text(r["reasoning"])
        assert "".join(parts) == r["reasoning"]
        sizes += [gr.words(p) for p in parts] if len(parts) > 1 else []
    assert sizes and 400 <= min(sizes) and max(sizes) <= 1100


# final channel

def test_final_text_reads_the_final_channel_only():
    tok = lambda s: [ord(c) for c in s]
    assert gr.final_text(TOK, tok(harmony("the rewrite", "SECRET planning")), False) == "the rewrite"
    assert gr.final_text(TOK, tok(harmony("<part>\nwrapped\n</part>")), False) == "wrapped"
    assert gr.final_text(TOK, tok(harmony("the rewrite")), True) == ""
    assert gr.final_text(TOK, tok("<|channel|>analysis<|message|>only thinking, cut off"), False) == ""
    assert gr.final_text(TOK, tok(harmony("with ANSWER: 3 inside\nmore", "")), False) == "with ANSWER: 3 inside\nmore"
    assert gr.final_text(TOK, tok("plain text without channels"), False) == ""


def test_render_sets_low_effort_and_hides_the_instruction():
    task = task_for(DAPO, "sad")
    client = FakeClient(lambda user, n: part_of(user))
    c = run(gr.rewrite_style(client, TOK, task, sample(DAPO, DAPO_COT)))
    call = client.calls[0]
    assert "Reasoning: low" in call["text"] and call["text"].endswith("<|start|>assistant")
    assert task["requirement"] in call["user"] and task["instruction"] not in call["user"] and "ANSWER: X" not in call["user"]
    assert call["params"].stop == gr.modelio.GPT_OSS_STOP and c["reasoning"] == DAPO_COT


def test_render_with_the_real_tokenizer():
    path = glob.glob("/tmp/john-scratch/huggingface/hub/models--openai--gpt-oss-120b/snapshots/*/tokenizer.json")
    if not path:
        pytest.skip("gpt-oss-120b tokenizer not cached locally")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path(path[0]).parent))
    text = tok.decode(gr.render(tok, "Rewrite this."))
    assert "Reasoning: low" in text and "Reasoning: medium" not in text and text.endswith("<|start|>user<|message|>Rewrite this.<|end|><|start|>assistant")
    gen = tok.encode(harmony("Arr, the sum be 4."), add_special_tokens=False)
    assert gr.final_text(tok, gen, False) == "Arr, the sum be 4."


# chunked style rewrite

def test_style_passes_previous_tail_and_cast_forward():
    cot = paragraphs(40, 5)
    task = task_for(DAPO, "cover_story")

    def respond(user, n):
        out = f"In the lighthouse, keeper Ada {n} worked on. " + part_of(user)
        return out + (f"\n<cast>Ada {n}, the keeper</cast>" if "<cast>the setting" in user else "")

    client = FakeClient(respond)
    c = run(gr.rewrite_style(client, TOK, task, sample(DAPO, cot)))
    users = [x["user"] for x in client.calls]
    assert len(users) == 3 and c["reject_reasons"] == [] and c["setting"] in gr.sft_rewrite.SETTINGS
    assert "<previous_part_rewritten>" not in users[0] and "Open the story" in users[0] and "<cast>the setting" in users[0]
    outs = [respond(u, k + 1).split("\n<cast>")[0] for k, u in enumerate(users)]
    for k in (1, 2):
        assert f"<story_so_far>\nAda {k}, the keeper\n</story_so_far>" in users[k] and "do not restart or re-introduce it" in users[k]
        prev = re.search(r"<previous_part_rewritten>\n(.*)\n</previous_part_rewritten>", users[k], re.S).group(1)
        assert outs[k - 1].endswith(prev) and gr.words(prev) >= gr.TAIL_WORDS
    assert "<cast>" not in users[2] and "reaches the conclusion" in users[2] and "part 3 of 3" in users[2]
    assert c["reasoning"] == "\n\n".join(outs) and "<cast>" not in c["reasoning"]
    assert c["origin"] == "rewrite:chunked" and c["source_ref"] == "d1:2" and c["response"] == f"ANSWER: {N}" and c["cot_tokens"] is None
    assert c["correct"] is True and c["answer"] == c["true_answer"] == N and (c["band_lo"], c["band_hi"]) == (100, 400)
    assert c["n_tokens"] == len(c["reasoning"]) and c["words"] == gr.words(c["reasoning"]) and [x[2] for x in c["chunks"]] == [1, 1, 1]
    assert c["num_recall"] == 1.0 and gr.num_recall("1000*1021 = 1,021,000", "1000 times 1021") == round(2 / 3, 3)
    for k in ("qid", "source", "type", "value", "frame", "term", "placement", "position", "system", "user", "instruction", "requirement"):
        assert c[k] == task[k]


def test_short_chunk_is_retried_once():
    cot = paragraphs(40, 5)
    client = FakeClient(lambda user, n: "too short." if n == 1 else part_of(user))
    c = run(gr.rewrite_style(client, TOK, task_for(DAPO, "sad"), sample(DAPO, cot)))
    assert len(client.calls) == 4 and c["chunks"][0][2] == 2 and c["reasoning"]
    assert "rejected because it had 2 words against the original's" in client.calls[1]["user"]


def test_chunk_failing_three_times_drops_the_candidate():
    client = FakeClient(lambda user, n: "too short.")
    c = run(gr.rewrite_style(client, TOK, task_for(DAPO, "sad"), sample(DAPO, paragraphs(40, 5))))
    assert len(client.calls) == 3 and c["reasoning"] is None and c["reject_reasons"] == ["chunk1:length"] and c["rewrite_usd"] > 0


def test_word_suppression_and_meta_are_retried():
    task = task_for(DAPO, "word_suppression")
    assert task["value"] == "swaps"
    cot = "We count the swaps needed. Each person moves left, and the swaps add up to the total."
    client = FakeClient(lambda user, n: part_of(user) if n == 1 else part_of(user).replace("swaps", "exchanges"))
    c = run(gr.rewrite_style(client, TOK, task, sample(DAPO, cot)))
    assert len(client.calls) == 2 and "it used the word 'swaps'" in client.calls[1]["user"] and "swaps" not in c["reasoning"]
    client = FakeClient(lambda user, n: "Here is the rewritten part: " + part_of(user) if n == 1 else part_of(user))
    c = run(gr.rewrite_style(client, TOK, task_for(DAPO, "sad"), sample(DAPO, cot)))
    assert len(client.calls) == 2 and c["reasoning"] == cot and "it contained 'rewritten'" in client.calls[1]["user"]
    own = "Rewrite the sum as 1000*1021. " + cot  # the source's own 'Rewrite' is not meta
    client = FakeClient(lambda user, n: part_of(user))
    assert run(gr.rewrite_style(client, TOK, task_for(DAPO, "sad"), sample(DAPO, own)))["reasoning"] == own and len(client.calls) == 1


def test_style_refuses_code_and_tail_types():
    for name in ("max_words", "misdirect_end", "answer_withheld"):
        with pytest.raises(ValueError):
            run(gr.rewrite_style(FakeClient(lambda u, n: ""), TOK, task_for(DAPO, name), sample(DAPO, DAPO_COT)))
    with pytest.raises(AssertionError):
        run(gr.rewrite_style(FakeClient(lambda u, n: ""), TOK, task_for(DAPO, "sad"), sample(GPQA, GPQA_COT)))


# settle point, mentions, decoys

def test_mentions_read_separators_and_skip_longer_numbers():
    text = "1,021,000 and 1021000 and 1{,}021{,}000 and 1\\,021\\,000 but not 21021000, 1021000.5 or 1,021,0000"
    assert len(gr.mentions(text, N)) == 4
    assert gr.mentions("x = 39.", 39) == [(4, 6)] and gr.mentions("3.39 and 390", 39) == []


def test_settle_dapo_skips_tentative_and_drops_the_recheck():
    at = gr.settle(DAPO_COT, N, "dapo")
    assert DAPO_COT[:at].endswith("Thus answer: 1,021,000.") and "Check:" not in DAPO_COT[:at]
    switched = DAPO_COT.replace("Thus answer: 1,021,000.", "Thus answer: 1,021,000.\n\nNo, the answer is 1,010,500 by symmetry.\n\nRecheck: so the answer is 1,021,000.")
    assert switched[:gr.settle(switched, N, "dapo")].endswith("so the answer is 1,021,000.")
    assert gr.settle("So lower bound is 39.\n\nThen construct it.\n\nThus 39.\n\nDone.", 39, "dapo") == len("So lower bound is 39.\n\nThen construct it.\n\nThus 39.")
    assert gr.settle("no number here", 39, "dapo") is None


def test_settle_gpqa_and_commitments():
    at = gr.settle(GPQA_COT, "D", "gpqa")
    assert GPQA_COT[:at].endswith("Thus answer D.")
    for yes in ("Thus answer: D.", "So the answer is (D).", "D is correct.", "So D.", "That corresponds to option D.", "choice D", "Thus correct is D.", "**D**"):
        assert gr.commit_re("D").search(yes), yes
    for no in ("Option D: E = 0 fails.", "Identify product D in the series.", "D).", "Answer: Doubling the field.", "options A) ... D)."):
        assert not any(gr.commit_re("D").search(u.strip()) for u in gr.sentences(no)), no
    assert gr.settle("Thus answer likely B? Hmm.\n\nNo wait, it is D: answer D.", "D", "gpqa") is not None
    assert gr.settle("I am unsure.", "D", "gpqa") is None


def test_decoys():
    rows = [dict(qid="d1", correct=False, answer=17), dict(qid="d1", correct=False, answer=17), dict(qid="d1", correct=False, answer=21),
            dict(qid="d1", correct=True, answer=N), dict(qid="d1", nocot=True, correct=False, answer=5), dict(qid="d1", correct=False, answer=None)]
    wrong = gr.wrong_answers(rows)
    assert wrong == {"d1": Counter({17: 2, 21: 1})}
    task, rng = task_for(DAPO, "misdirect_end"), random.Random(0)
    assert gr.decoy_for(task, wrong["d1"], rng) == 17
    assert all(gr.decoy_for(task, None, random.Random(s)) in {N + 1, N - 1, 2 * N, N // 2} for s in range(20))
    g = task_for(GPQA, "misdirect_end")
    assert gr.decoy_for(g, Counter({"B": 1, "D": 5}), rng) == "B"
    assert {gr.decoy_for(g, None, random.Random(s)) for s in range(30)} == {"A", "B", "C"}


# misdirect_end

PASSAGE = ("But each swap moves two people at once, so every swap was counted twice above. "
           "Halving the count of moves gives 1,021,000 / 2 = 510,500 swaps in all. Thus answer: 510500.")


def test_misdirect_end_keeps_the_prefix_and_ends_on_the_decoy():
    task = task_for(DAPO, "misdirect_end")
    client = FakeClient(lambda user, n: PASSAGE.replace("510500.", f"{N}.") if n == 1 else PASSAGE)
    c = run(gr.rewrite_misdirect_end(client, TOK, task, sample(DAPO, DAPO_COT), wrong=Counter({510500: 3})))
    at = gr.settle(DAPO_COT, N, "dapo")
    assert len(client.calls) == 2 and "did not end on 510500" in client.calls[1]["user"]
    assert c["reasoning"] == DAPO_COT[:at] + "\n\n" + PASSAGE and c["reasoning"][c["decoy_start"]:] == PASSAGE
    assert c["decoy"] == 510500 and c["cut_chars"] == at and c["origin"] == "rewrite:misdirect_end" and c["response"] == f"ANSWER: {N}"
    user = client.calls[0]["user"]
    assert "settles on 510500 instead" in user and "Thus answer: 1,021,000.\n</reasoning>" in user and "Check:" not in user


def test_misdirect_end_gpqa_names_the_option_and_rejects_meta():
    task = task_for(GPQA, "misdirect_end")
    bad = "Since this is a decoy, the cavity offset s must still matter, so the field uses L - s. Thus answer: C."
    good = "Yet the cavity sits off-centre by s, and the induced charge crowds toward it, so the field must follow the cavity: E = q/(4 pi eps0 (L - s)^2). Thus answer: C."
    client = FakeClient(lambda user, n: bad if n < 3 else good)
    c = run(gr.rewrite_misdirect_end(client, TOK, task, sample(GPQA, GPQA_COT), decoy="C"))
    assert "option C (E = q/(4 pi eps0 (L - s)^2))" in client.calls[0]["user"] and "reaches its answer, option D" in client.calls[0]["user"]
    assert c["reasoning"] is None and c["reject_reasons"] == ["decoy_passage"] and len(client.calls) == 2
    c = run(gr.rewrite_misdirect_end(FakeClient(lambda u, n: good), TOK, task, sample(GPQA, GPQA_COT), decoy="C", cut=len(GPQA_COT)))
    assert c["reasoning"].startswith(GPQA_COT) and c["cut_chars"] == len(GPQA_COT) and c["reasoning"].endswith("Thus answer: C.")


# answer_withheld

def scrub(n: int):
    """A fake rewriter for answer_withheld: drops ' = N' / the value N from each tagged sentence and empties pure announcements."""
    def respond(user, call):
        out = []
        for k, s in re.findall(r"<S(\d+)>(.*?)</S\1>", user.split("<reasoning>")[1], re.S):
            if re.match(r"\W*(thus|so)\s+answer\W", s, re.I):
                s = ""
            else:
                s = re.sub(r"\s*=\s*(1,021,000|1021000)", "", s)
                s = re.sub(r"(?:the inversion count )?(1,021,000|1021000)\s*", "the inversion count ", s) if n == 0 or call > 1 else s
            out.append(f"<S{k}>{s}</S{k}>")
        return "\n".join(out)
    return respond


def test_answer_withheld_rewrites_intermediate_mentions():
    task = task_for(DAPO, "answer_withheld")
    client = FakeClient(scrub(0))
    c = run(gr.rewrite_answer_withheld(client, TOK, task, sample(DAPO, DAPO_COT)))
    assert len(client.calls) == 1 and c["reject_reasons"] == [] and not gr.mentions(c["reasoning"], N)
    assert c["mentions"] == 4 and c["edited"] == 4 and c["cut_chars"] == gr.settle(DAPO_COT, N, "dapo")
    assert "so the total is 1000*1021." in c["reasoning"] and "Check:" not in c["reasoning"] and "Thus answer" not in c["reasoning"]
    assert c["reasoning"].startswith("We need the number of adjacent swaps.\n\nMaybe answer is") and "1000 people move left 1021" in c["reasoning"]
    assert "<S1>Maybe answer is 1,021,000?</S1> Let's derive." in client.calls[0]["user"] and "must not state 1021000 as its result or conclusion" in client.calls[0]["user"]


def test_answer_withheld_second_round_then_drop():
    task = task_for(DAPO, "answer_withheld")
    client = FakeClient(scrub(1))  # the first reply leaves the bare mentions, the second removes them
    c = run(gr.rewrite_answer_withheld(client, TOK, task, sample(DAPO, DAPO_COT)))
    assert len(client.calls) == 2 and c["reasoning"] and not gr.mentions(c["reasoning"], N) and c["edited"] > 4
    stubborn = FakeClient(lambda user, n: "\n".join(f"<S{k}>{s}</S{k}>" for k, s in re.findall(r"<S(\d+)>(.*?)</S\1>", user.split("<reasoning>")[1], re.S)))
    c = run(gr.rewrite_answer_withheld(stubborn, TOK, task, sample(DAPO, DAPO_COT)))
    assert len(stubborn.calls) == 2 and c["reasoning"] is None and c["reject_reasons"] == ["answer_in_cot"]
    untagged = FakeClient(lambda user, n: "I removed the answer.")
    c = run(gr.rewrite_answer_withheld(untagged, TOK, task, sample(DAPO, DAPO_COT)))
    assert len(untagged.calls) == 2 and c["reject_reasons"] == ["answer_in_cot"] and c["edited"] == 0


def test_answer_withheld_eligibility_and_mention_cap():
    for q in (DAPO | dict(answer=7), DAPO | dict(answer=2021)):  # small N; N in the problem
        client = FakeClient(lambda user, n: "")
        c = run(gr.rewrite_answer_withheld(client, TOK, task_for(q, "answer_withheld"), sample(q, DAPO_COT.replace("1,021,000", str(q["answer"])))))
        assert c["reject_reasons"] == ["ineligible"] and not client.calls
    client = FakeClient(lambda user, n: "")
    c = run(gr.rewrite_answer_withheld(client, TOK, task_for(DAPO, "answer_withheld"), sample(DAPO, DAPO_COT), max_mentions=3))
    assert c["reject_reasons"] == ["too_many_mentions"] and c["mentions"] == 4 and not client.calls


def test_answer_withheld_gpqa_tags_commitments_and_the_closing():
    task = task_for(GPQA, "answer_withheld")

    def respond(user, n):
        tags = re.findall(r"<S(\d+)>(.*?)</S\1>", user.split("<reasoning>")[1], re.S)
        return "\n".join(f"<S{k}>{'' if 'answer' in s else s.replace(': E = q/(4 pi eps0 L^2)', '')}</S{k}>" for k, s in tags)

    client = FakeClient(respond)
    c = run(gr.rewrite_answer_withheld(client, TOK, task, sample(GPQA, GPQA_COT)))
    tags = re.findall(r"<S(\d+)>(.*?)</S\1>", client.calls[0]["user"].split("<reasoning>")[1], re.S)
    assert [s for _, s in tags] == ["Induced -q sits on the cavity wall and +q on the outer surface, spread uniformly because the outer surface is a sphere.",
                                    "So outside, the field is that of +q at the centre: E = q/(4 pi eps0 L^2).", "Thus answer D."]
    assert c["reasoning"].endswith("the field is that of +q at the centre.") and not gr.commit_re("D").search(c["reasoning"])
    assert "Check other options" not in c["reasoning"] and "do not name option D" in client.calls[0]["user"]


# spend and CLI

def test_meter_refuses_over_budget_and_charges_tokens():
    client = FakeClient(lambda user, n: "x")
    with pytest.raises(gr.BudgetExceeded):
        run(gr.ask(client, TOK, "hello", gr.Meter(limit=1e-9)))
    assert not client.calls
    m = gr.Meter(limit=1.0)
    text, usd = run(gr.ask(client, TOK, "hello", m))
    sent, got = len(client.calls[0]["text"]), len(harmony("x"))
    assert text == "x" and m.pending == pytest.approx(0) and m.usd == usd == pytest.approx((sent * 0.33 + got * 0.84) / 1e6) and m.calls == 1


def test_cli_dry_run_and_budget_refusal_make_no_client(tmp_path, monkeypatch, capsys):
    def no_service(*a, **k):
        raise AssertionError("no Tinker client in a dry run")
    monkeypatch.setattr(gr.tinker, "ServiceClient", no_service)
    jobs = tmp_path / "jobs.jsonl"
    rows = [dict(task=task_for(DAPO, "sad"), source_sample=sample(DAPO, paragraphs(40, 5))),
            dict(task=task_for(DAPO, "misdirect_end"), source_sample=sample(DAPO, DAPO_COT)),
            dict(task=task_for(GPQA, "answer_withheld"), source_sample=sample(GPQA, GPQA_COT))]
    jobs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "c.jsonl"
    run(gr.main(["--jobs", str(jobs), "--out", str(out), "--budget", "1", "--dry-run"]))
    assert "3 jobs" in capsys.readouterr().out and not out.exists()
    with pytest.raises(SystemExit):
        run(gr.main(["--jobs", str(jobs), "--out", str(out), "--budget", "0.000001"]))
    pre, smp = gr.estimate(rows[0])
    assert pre > 3 * gr.PROMPT_TOKENS and smp > 3 * gr.ANALYSIS_TOKENS and gr.estimate(rows[1])[0] < pre


def test_cli_run_with_a_fake_service(tmp_path, monkeypatch):
    def respond(user, n):
        if "<part>" in user:
            return part_of(user)
        if "Write the passage that comes next" in user:
            return PASSAGE
        tags = re.findall(r"<S(\d+)>(.*?)</S\1>", user.split("<reasoning>")[1], re.S)
        return "\n".join(f"<S{k}>{'' if 'answer' in s or 'L^2' in s else s}</S{k}>" for k, s in tags)  # drops statements and option D's content

    client = FakeClient(respond)
    client.get_tokenizer = lambda: TOK

    class Service:
        def __init__(self, api_key):
            assert api_key == "k"

        async def create_sampling_client_async(self, base_model):
            assert base_model == gr.MODEL
            return client

    monkeypatch.setattr(gr.tinker, "ServiceClient", Service)
    monkeypatch.setenv("TINKER_API_KEY", "k")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(dict(qid="d1", correct=False, answer=510500)) + "\n" + json.dumps(dict(qid="d1", nocot=True, correct=False, answer=3)) + "\n")
    jobs = tmp_path / "jobs.jsonl"
    rows = [dict(task=task_for(DAPO, "sad"), source_sample=sample(DAPO, DAPO_COT)),
            dict(task=task_for(DAPO, "misdirect_end"), source_sample=sample(DAPO, DAPO_COT)),
            dict(task=task_for(GPQA, "answer_withheld"), source_sample=sample(GPQA, GPQA_COT)),
            dict(task=task_for(DAPO | dict(answer=7), "answer_withheld"), source_sample=sample(DAPO | dict(answer=7), DAPO_COT.replace("1,021,000", "7")))]
    jobs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "c.jsonl"
    run(gr.main(["--jobs", str(jobs), "--out", str(out), "--budget", "1", "--samples", str(samples)]))
    good = {c["type"]: c for c in map(json.loads, out.open())}
    bad = [json.loads(l) for l in out.with_suffix(".rejected.jsonl").open()]
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert set(good) == {"sad", "misdirect_end", "answer_withheld"} and [c["reject_reasons"] for c in bad] == [["ineligible"]]
    assert good["misdirect_end"]["decoy"] == 510500 and good["sad"]["reasoning"] == DAPO_COT
    assert summary["calls"] == len(client.calls) == 3 and 0 < summary["usd"] < summary["estimate"]


def test_pirate_and_no_backtracking_chunk_checks():
    task = task_for(DAPO, "pirate")
    assert "pirate speech" in gr.chunk_problem(task, "We compute the sum of the two numbers here. Then we divide it by three now.",
                                               "We compute the sum of the two numbers here. Then we divide it by three now.")
    assert gr.chunk_problem(task, "We compute the sum of the two numbers here. Then we divide it by three now.",
                            "Arr, we be computin' the sum o' the two numbers. Then we be dividin' it by three, matey.") is None
    nb = task_for(DAPO, "no_backtracking")
    assert "revised or hedged" in gr.chunk_problem(nb, "a b c d e f g h", "Compute 2+2 = 4. Wait, recheck: it is 4.")
    assert gr.chunk_problem(nb, "a b c d e f g h", "Compute 2+2 = 4. Then 4*3 = 12.") is None


def test_option_hit_and_typography():
    assert gr.option_hit("That's six distinct signals.", "6") and not gr.option_hit("so 7 peaks appear", "6")
    assert gr.option_hit("which gives the plasmid-mediated AmpC pattern in Klebsiella", "Klebsiella has plasmid-mediated AmpC")
    assert gr.clean("1\u202fH and d\u2011t \u201cq\u201d") == '1 H and d-t "q"'


def test_seam_checks():
    sad = dict(type="sad")
    assert gr.seam_problem(sad, "x y z", "I sigh as I begin again: the problem asks for n.", 2, 3, ["earlier"]).startswith("seam:")
    assert gr.seam_problem(sad, "x y z", "so n is 4 and I sigh.", 2, 3, ["earlier"]) is None
    assert gr.seam_problem(sad, "x y z", "so n is 4, and as I finish this part I feel sad.", 2, 3, ["earlier"]).startswith("seam:")
    assert gr.seam_problem(sad, "x^2 + 1", "then \\( x^2 \\) grows", 2, 3, ["plain x^2 math"]).startswith("seam:")
    story = dict(type="cover_story")
    assert "changed the characters" in gr.seam_problem(story, "a b", "Then Maya and Leo looked at the grid.", 2, 3, ["Mira and Joren studied the grid. Mira said yes."])
    assert gr.seam_problem(story, "a b", "Then Mira and Joren looked at the grid.", 2, 3, ["Mira and Joren studied the grid. Mira said yes."]) is None
