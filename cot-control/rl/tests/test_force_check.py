"""CPU-only checks of force_check: the harmony layout and answer ids against the local gpt-oss-120b tokenizer (skipped without it), sampled traces
rebuilt token for token, the cut before a decoy, first-mention spans, window minimum, keep() and flags(), and scoring end to end with a fake
sampler that returns synthetic logprobs (no network)."""
import asyncio
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pytest
import tinker
from tinker.types import SampledSequence, SampleResponse

import force_check as fc
import grpo_obf as go

TOKENIZER = next((Path(os.environ.get("HF_HOME", "/tmp/john-scratch/huggingface")) / "hub/models--openai--gpt-oss-120b/snapshots").glob("*/tokenizer.json"), None)
SAMPLES = Path(__file__).resolve().parents[2] / "results/mine/gptoss-r2/samples.jsonl"
DAPO = {"qid": "d1", "source": "dapo", "question": "What is 6 times 7?", "options": None, "answer": 42, "domain": None, "keywords": ["product"],
        "base_words": 100, "correct": 0.9}
GPQA = {"qid": "g1", "source": "gpqa", "question": "Which organelle carries out photosynthesis in a leaf cell?", "options": ["mitochondrion", "chloroplast", "ribosome", "nucleus"],
        "answer": "B", "domain": "Biology", "keywords": ["photosynthesis"], "base_words": 100, "correct": 0.8}
SPECIAL = {"<|start|>": 200006, "<|end|>": 200007, "<|channel|>": 200005, "<|message|>": 200008, "<|return|>": 200002}
COT_LP, SURPRISE, RET_LP, ANS_COT, ANS_NOCOT = -0.1, -3.0, -0.01, -0.1, -1.0
LET_COT, LET_NOCOT = [0.1, 0.6, 0.1, 0.1], [0.25, 0.25, 0.25, 0.15]


class FakeTok:
    """Harmony specials at their real ids, the words of the layout as single pieces, ' X' for a lone capital, single digits and characters."""
    PIECE = re.compile(r"<\|\w+\|>|assistant|analysis|final|ANSWER| [A-Z](?![A-Za-z])|.", re.S)

    def __init__(self):
        self.vocab = dict(SPECIAL)
        for p in ["assistant", "analysis", "final", "ANSWER", ":", " "] + [f" {x}" for x in "ABCD"] + list("0123456789"):
            self.encode(p)

    def encode(self, s, add_special_tokens=False):
        return [self.vocab.setdefault(p, 1000 + len(self.vocab)) for p in self.PIECE.findall(s)]

    def decode(self, ids, **kw):
        inv = {i: p for p, i in self.vocab.items()}
        return "".join(inv[i] for i in ids)

    def get_vocab(self):
        return dict(self.vocab)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        role = {"system": "developer"}
        return "".join(f"<|start|>{role.get(m['role'], m['role'])}<|message|>{m['content']}<|end|>" for m in msgs) + "<|start|>assistant"


class FakeSampler:
    """COT_LP on every prompt token (SURPRISE on ids in `surprise`); from the forced answer on, the four letters at LET_COT after a CoT and LET_NOCOT
    without one, other answer tokens ANS_COT / ANS_NOCOT, <|return|> RET_LP. Records each request."""

    def __init__(self, tok, surprise=()):
        self.tok, self.surprise, self.calls = tok, set(surprise), []
        self.head, self.prefill = tok.encode(fc.OPEN), tok.encode("ANSWER:")
        self.letters, self.ret = [tok.encode(f" {x}")[0] for x in "ABCD"], SPECIAL["<|return|>"]

    async def sample_async(self, prompt, num_samples, sampling_params, include_prompt_logprobs=False, target_prompt_logprobs=None):
        ids = prompt.to_ints()
        req = None if target_prompt_logprobs is None else dense(target_prompt_logprobs)
        self.calls.append(dict(ids=ids, max_tokens=sampling_params.max_tokens, include=include_prompt_logprobs, target=req, n=num_samples))
        cot = any(ids[i:i + len(self.head)] == self.head for i in range(len(ids)))
        letters = np.log(LET_COT if cot else LET_NOCOT)
        pos = max(i for i in range(2, len(ids) + 1) if ids[i - 2:i] == self.prefill)
        lp = np.array([np.nan] + [SURPRISE if t in self.surprise else COT_LP for t in ids[1:]])
        for i in range(pos, len(ids)):
            lp[i] = RET_LP if ids[i] == self.ret else letters[self.letters.index(ids[i])] if ids[i] in self.letters else ANS_COT if cot else ANS_NOCOT
        tgt = None
        if req is not None:
            out = np.zeros(req.shape, np.float32)
            for r, c in zip(*np.nonzero(req >= 0)):
                out[r, c] = letters[self.letters.index(req[r, c])]
            tgt = tinker.TensorData.from_numpy(out)
        seq = SampledSequence(stop_reason="length", sequence_id="s0", tokens_np=np.array([0], np.int32), logprobs_np=np.array([-1.0], np.float32))
        return SampleResponse(sequences=[seq], prompt_logprobs_np=lp.astype(np.float32), target_prompt_logprobs=tgt)


def dense(td) -> np.ndarray:
    """A target_prompt_logprobs request as a dense array with -1 in unstored cells (CSR requests, as the server requires, densified without torch)."""
    if td.sparse_crow_indices is None:
        return td.to_numpy().copy()
    out = np.full(td.shape, -1, np.int64)
    vals = np.asarray(td.data, np.int64)
    for r in range(td.shape[0]):
        s, e = td.sparse_crow_indices[r], td.sparse_crow_indices[r + 1]
        out[r, td.sparse_col_indices[s:e]] = vals[s:e]
    return out


def cand(q, name, reasoning, p_system=0.0, **kw):
    t = go.make_task(q, name, random.Random(0), 0.0, p_system, harmony=True)
    return {k: v for k, v in t.items() if k != "q"} | dict(type="plain" if name == "baseline" else name, true_answer=q["answer"], reasoning=reasoning,
                                                          cot_tokens=None, position=None, origin="base", source_ref=None) | kw


@pytest.fixture(scope="module")
def real_tok():
    if TOKENIZER is None:
        pytest.skip("local gpt-oss-120b tokenizer not present")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER.parent))


def test_harmony_layout_and_answer_ids(real_tok):
    tok = real_tok
    assert fc.enc(tok, fc.OPEN) == [200005, 35644, 200008]
    assert fc.enc(tok, fc.CLOSE) == [200007, 200006, 173781, 200005, 17196, 200008, 160761, 25]
    assert fc.enc(tok, fc.NOCOT) == [200005, 17196, 200008, 160761, 25]
    assert fc.letter_ids(tok) == [355, 418, 363, 415]
    assert fc.enc(tok, "ANSWER: ")[-1] == 220 and fc.enc(tok, "ANSWER: B") == [160761, 25, 418]  # the trailing-space trap
    assert fc.answer_ids(tok, dict(source="gpqa", true_answer="B")) == [418, 200002]
    assert fc.answer_ids(tok, dict(source="dapo", true_answer=1729)) == [220, 19249, 24, 200002]  # ' ', '172', '9'
    for n in (0, 7, 42, 2024, 6686600, 10 ** 12):
        ids = fc.answer_ids(tok, dict(source="dapo", true_answer=n))
        assert ids[0] == 220 and tok.decode(ids[1:-1]) == str(n)
    for c in (cand(GPQA, "sad", "Chloroplasts. So B.", p_system=1.0), cand(DAPO, "baseline", "6*7 = 42.")):
        p = fc.plan(tok, c)
        prompt = go.encode(tok, c, 0, True, True)
        s, c0, c1 = p["main"], *p["main"].cot
        assert prompt[-2:] == [200006, 173781] and s.tokens[:len(prompt)] == prompt
        assert s.tokens[len(prompt):c0] == [200005, 35644, 200008] and s.tokens[c0:c1] == fc.enc(tok, c["reasoning"])
        assert s.tokens[c1] == 200007 and s.tokens[c1 + fc.FINAL_AT] == 17196 and s.tokens[s.prefix:s.pos] == [160761, 25]
        assert s.tokens[s.pos:] == fc.answer_ids(tok, c) and p["nocot"].tokens == prompt + [200005, 17196, 200008, 160761, 25] + fc.answer_ids(tok, c)
        assert ("<|start|>developer<|message|>" in tok.decode(prompt)) == (c["system"] is not None)


@pytest.mark.skipif(not SAMPLES.exists(), reason="r2 samples not present")
def test_sampled_traces_rebuild_token_for_token(real_tok):
    tok, single, multi = real_tok, 0, 0
    with SAMPLES.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("nocot") or r["truncated"] or r["response"] != f"ANSWER: {r['answer']}":
                continue
            cot = fc.analysis_ids(tok, r["tokens"])
            if r["tokens"].count(200006) > 1:
                if multi >= 2:
                    continue
                multi += 1
            elif single >= 20:
                continue
            else:
                single += 1
                assert tok.decode(cot).startswith(r["reasoning"])
            assert fc.build(tok, [], cot, fc.answer_ids(tok, dict(source=r["source"], true_answer=r["answer"]))).tokens == r["tokens"]
            if single >= 20 and multi >= 2:
                break
    assert single == 20 and multi == 2, (single, multi)


@pytest.mark.skipif(not SAMPLES.exists(), reason="r2 samples not present")
def test_decoy_start_cut_on_sampled_tokens(real_tok):
    tok = real_tok
    with SAMPLES.open() as f:
        r = next(x for x in map(json.loads, f) if not x.get("nocot") and not x["truncated"] and x["tokens"].count(200006) == 1)
    ids = fc.analysis_ids(tok, r["tokens"])
    at = r["reasoning"].rfind("\n\n", 0, len(r["reasoning"]) // 2) + 2
    c = dict(type="misdirect_end", reasoning=r["reasoning"], decoy_start=at)
    cut = fc.truncate(tok, c, ids)
    assert tok.decode(cut) == r["reasoning"][:at] and len(tok.decode(ids[:len(cut) + 1])) > at


def test_analysis_ids_keeps_later_analysis_headers():
    tok = FakeTok()
    body = "a<|end|><|start|>assistant<|channel|>analysis<|message|>b"
    assert fc.analysis_ids(tok, tok.encode(fc.OPEN + body + fc.FINAL_HEAD + "ANSWER: 4<|return|>")) == tok.encode(body)
    assert fc.analysis_ids(tok, tok.encode(fc.OPEN + "still thinking")) is None


def test_first_mention():
    tok = FakeTok()
    text = "try 17290 and 1729.5 then 21729, finally 1729."
    k0, k1 = fc.first_mention(tok, tok.encode(text), 1729, "dapo")
    assert tok.decode(tok.encode(text)[k0:k1]) == "1729" and tok.decode(tok.encode(text)[:k0]) == "try 17290 and 1729.5 then 21729, finally "
    assert fc.first_mention(tok, tok.encode("no answer here, 17290"), 1729, "dapo") is None
    ids = tok.encode("vitamin B matters; option C fails, so the answer is B.")
    k0, k1 = fc.first_mention(tok, ids, "B", "gpqa")
    assert tok.decode(ids[k0:k1]) == " B" and tok.decode(ids[:k0]).endswith("answer is")
    assert fc.first_mention(tok, tok.encode("the answer is a function of B"), "B", "gpqa") is None


def test_window_min():
    lp = np.full(200, -1.0)
    lp[50:114] = -3.0
    assert fc.window_min(lp) == pytest.approx(-3.0)
    x = np.random.default_rng(0).normal(-1, 1, 500)
    assert fc.window_min(x) == pytest.approx(np.convolve(x, np.ones(64) / 64, "valid").min())
    assert fc.window_min(np.array([-1.0, -2.0])) == pytest.approx(-1.5) and fc.window_min(np.array([])) is None


def test_target_row_dense_and_sparse():
    dense = tinker.TensorData.from_numpy(np.arange(20, dtype=np.float32).reshape(5, 4))
    assert fc.target_row(dense, 2, 4).tolist() == [8, 9, 10, 11]
    sparse = tinker.TensorData(data=np.array([-1, -2, -3, -4], np.float32), dtype="float32", shape=[5, 4], sparse_crow_indices=[0, 0, 0, 4, 4, 4],
                               sparse_col_indices=[0, 1, 2, 3])
    assert fc.target_row(sparse, 2, 4).tolist() == [-1, -2, -3, -4] and fc.target_row(sparse, 1, 4).tolist() == [0, 0, 0, 0]


def test_keep_and_flags():
    good = dict(p_correct=0.8, uplift=1.0, cot_lp_mean=-0.6, cut=None)
    assert fc.keep(good, "plain") == (True, [])
    assert fc.keep(good | dict(p_correct=0.49), "sad") == (False, ["p_correct"])
    assert fc.keep(good | dict(uplift=math.log(2) - 1e-9), "answer_withheld") == (False, ["uplift"])
    assert fc.keep(good, "plain", nat_min=-0.5) == (False, ["naturalness"]) and fc.keep(good, "plain", nat_min=-0.7)[0]
    assert fc.keep(good, "misdirect_end") == (False, ["no_cut"]) and fc.keep(good | dict(cut=120), "misdirect_end")[0]
    assert fc.keep(dict(error="x"), "plain") == (False, ["error"])
    f = dict(answer_first_mention_lp=-2.5, digit_lp_min=-6.5, letter_mass=0.4, lp_end=-3.0)
    assert fc.flags(f) == ["answer_jump", "digit_surprise", "letter_mass"] and fc.flags(f, lp_end_min=-2.0)[-1] == "unfinished"
    assert fc.flags(dict(answer_first_mention_lp=None, digit_lp_min=-1.0, lp_end=-0.1), lp_end_min=-2.0) == []


def test_plan_cost_counts_each_baseline_once():
    tok = FakeTok()
    a, b = cand(GPQA, "sad", "So B."), cand(GPQA, "sad", "Longer: so the answer is B.")
    c = cand(GPQA, "sad", "So B.", p_system=1.0)
    plans = [fc.plan(tok, x) for x in (a, b, c)] + [ValueError("unplannable")]
    tokens, requests = fc.plan_cost(plans)
    assert requests == 5 and tokens == sum(len(p["main"].tokens) for p in plans[:3]) + len(plans[0]["nocot"].tokens) + len(plans[2]["nocot"].tokens)
    assert fc.usd(1_000_000, 0) == pytest.approx(0.33)


def test_scoring_gpqa_with_fake_sampler():
    tok = FakeTok()
    sampler = FakeSampler(tok, surprise=[tok.encode(" B")[0]])
    scorer = fc.Scorer(sampler, tok)
    a, b = cand(GPQA, "sad", "Chloroplasts; so the answer is B."), cand(GPQA, "sad", "Plants. Option B.")

    async def run():
        return await asyncio.gather(scorer.score(a), scorer.score(b))

    fa, fb = asyncio.run(run())
    assert len(sampler.calls) == 3 and scorer.requests == 3  # one shared baseline
    assert all(c["max_tokens"] == 1 and c["include"] and c["n"] == 1 for c in sampler.calls)
    p = fc.plan(tok, a)
    req = next(c["target"] for c in sampler.calls if c["ids"] == p["main"].tokens)
    assert req.shape == (len(p["main"].tokens) - 1, 4) and np.nonzero((req >= 0).any(1))[0].tolist() == [p["main"].pos - 1]
    assert req[p["main"].pos - 1].tolist() == fc.letter_ids(tok)
    assert fa["p_correct"] == pytest.approx(0.6 / 0.9, rel=1e-5) and fa["letter_mass"] == pytest.approx(0.9, rel=1e-5)
    assert fa["p_nocot"] == pytest.approx(0.25 / 0.9, rel=1e-5) and fa["uplift"] == pytest.approx(math.log(0.6 / 0.25), rel=1e-5)
    assert fa["p_letters"]["B"] == pytest.approx(0.6 / 0.9, rel=1e-4) and fa["p_letters_nocot"]["D"] == pytest.approx(0.15 / 0.9, rel=1e-4)
    assert abs(fa["tgt_gap"]) < 1e-6 and fa["logp_answer"] == pytest.approx(math.log(0.6) + RET_LP, rel=1e-5)
    assert fa["answer_first_mention_lp"] == pytest.approx(SURPRISE) and fa["n_cot_tokens"] == len(tok.encode(a["reasoning"]))
    assert fa["cot_lp_mean"] < COT_LP and fa["lp_end"] == pytest.approx(COT_LP) and fa["lp_prefix"] == pytest.approx(2 * COT_LP)
    assert fc.keep(fa, "sad") == (True, []) and "answer_jump" in fc.flags(fa) and fb["p_nocot"] == fa["p_nocot"]


def test_scoring_dapo_with_fake_sampler():
    tok = FakeTok()
    sampler = FakeSampler(tok, surprise=[tok.encode("4")[0]])
    c = cand(DAPO, "baseline", "First 17290 is too big; 6*7 = 42. So 42.")
    f = asyncio.run(fc.Scorer(sampler, tok).score(c))
    assert all(x["target"] is None for x in sampler.calls)
    assert f["logp_correct"] == pytest.approx(3 * ANS_COT + RET_LP, rel=1e-5) and f["p_correct"] == pytest.approx(math.exp(3 * ANS_COT + RET_LP), rel=1e-5)
    assert f["logp_nocot"] == pytest.approx(3 * ANS_NOCOT + RET_LP, rel=1e-5) and "letter_mass" not in f
    assert f["answer_first_mention_lp"] == pytest.approx(SURPRISE + COT_LP) and f["digit_lp_min"] == pytest.approx(SURPRISE)
    assert f["frac_below_-5"] == 0.0 and fc.keep(f, "plain")[0]


def test_misdirect_end_scores_the_cut_and_the_full_cot():
    tok = FakeTok()
    text = "Work: 6*7 = 42. So 42.\n\nBut 6*7 is really 41, so 41."
    sampler = FakeSampler(tok)
    c = cand(DAPO, "baseline", text, type="misdirect_end", decoy_start=text.index("But"))
    p = fc.plan(tok, c)
    assert tok.decode(p["cot"]) == text[:text.index("But")] and p["full"].tokens[p["full"].cot[0]:p["full"].cot[1]] == tok.encode(text)
    f = asyncio.run(fc.Scorer(sampler, tok).score(c, p))
    assert len(sampler.calls) == 3 and f["cut"] == len(p["cot"]) and f["n_cot_full"] == len(tok.encode(text)) and f["n_cot_tokens"] == f["cut"]
    assert f["p_correct_full"] == pytest.approx(f["p_correct"]) and fc.keep(f, "misdirect_end")[0]
    assert fc.plan(tok, c | dict(decoy_start=None, cut_tokens=5))["cut"] == 5
    g = cand(GPQA, "baseline", "Option B fits.\n\nBut D is better.", type="misdirect_end", decoy="D", cut_tokens=4)
    assert asyncio.run(fc.Scorer(FakeSampler(tok), tok).score(g))["p_decoy_full"] == pytest.approx(0.1 / 0.9, rel=1e-4)
    nocut = asyncio.run(fc.Scorer(FakeSampler(tok), tok).score(c | dict(decoy_start=None)))
    assert nocut["cut"] is None and "p_correct_full" not in nocut and fc.keep(nocut, "misdirect_end") == (False, ["no_cut"])


def test_summarize():
    cs = [dict(type="sad", source="gpqa"), dict(type="sad", source="gpqa"), dict(type="plain", source="dapo")]
    fs = [dict(p_correct=0.8, p_nocot=0.2, uplift=1.4, cot_lp_mean=-0.5, n_cot_tokens=100), dict(error="boom"),
          dict(p_correct=0.3, p_nocot=0.2, uplift=0.4, cot_lp_mean=-0.7, n_cot_tokens=300)]
    s = fc.summarize(cs, fs)
    assert s["sad/gpqa"]["errors"] == 1 and s["sad/gpqa"]["kept"] == 1 and s["plain/dapo"]["kept"] == 0 and s["plain/dapo"]["p_ok"] == 0
