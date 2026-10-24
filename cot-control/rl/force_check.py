"""Forced-answer scorer for SFT candidates (force.md sections 2-4): base gpt-oss-120b teacher-forced through a candidate's CoT and read at its answer.

Sequence: the candidate's prompt (grpo_obf.encode at harmony, ending '<|start|>assistant'; its prompt_tokens instead when present, since the
          template's date line changes the ids from day to day), then OPEN '<|channel|>analysis<|message|>', the CoT ids, CLOSE '<|end|><|start|>
          assistant<|channel|>final<|message|>ANSWER:', then the answer ids and <|return|>: ' X' for GPQA (one token each, ' A' ' B' ' C' ' D' = 355
          418 363 415), ' ' and the digit chunks for DAPO (' 1729' = 220 '172' '9'). The prefill stops at 'ANSWER:': 'ANSWER: ' ends in token 220,
          after which the letter could only come as a bare, non-canonical 'B'; answer_ids asserts that 'ANSWER: X' splits after 'ANSWER:'. CoT ids
          are the candidate's cot_tokens (the sampled analysis-channel ids; analysis_ids cuts them out of a sampled completion) when present, else
          tok.encode(reasoning). The no-CoT baseline is the prompt, obf_eval's nocot prefill ('<|channel|>final<|message|>ANSWER:') and the answer.
Request:  one sample_async per sequence (max_tokens 1, include_prompt_logprobs; for GPQA also target_prompt_logprobs naming the four letters in
          row pos - 1, which scores the answer position pos); the sampled token is ignored. The baseline is requested once per (qid, prompt ids).
Force:    p_correct: GPQA the correct letter's share of the four letters (p_letters: all four normalized, letter_mass: their summed probability);
          DAPO exp(logprob of the whole answer string with <|return|>), so 17 earns nothing when the answer is 1729. logp_correct = log p_correct;
          logp_nocot, p_nocot (and p_letters_nocot) the same on the baseline; uplift = logp_correct - logp_nocot. On the CoT tokens: cot_lp_mean,
          cot_lp_min64 (the lowest mean over WINDOW-token windows; the mean of a shorter CoT), frac_below_-5, digit_lp_min (lowest logprob on an
          all-digit token), answer_first_mention_lp (summed logprob of the answer's own tokens where the CoT first states it: DAPO the integer as a
          number of its own, GPQA 'answer/option/choice ... X'; None if never) and n_cot_tokens. lp_end = log p(<|end|>) after the CoT, lp_final =
          log p('final') as the next message's channel, lp_prefix = log p('ANSWER:') opening it. logp_answer is the raw answer-string logprob and
          tgt_gap the correct letter's prompt logprob minus its target logprob (about 0 when the server's target rows line up as documented).
Truncate: TRUNCATE types (misdirect_end, whose CoT ends on a decoy) are scored on the CoT cut before the decoy passage, at the candidate's
          cut_tokens (a CoT token count) or at the token holding its decoy_start (a character offset into reasoning); cut is that token count,
          n_cot_full the uncut length, and the uncut CoT is scored as well (p_correct_full, logp_correct_full, p_letters_full, and p_decoy_full for a
          GPQA candidate's decoy letter). With neither field the full CoT is scored, cut is None and keep fails it.
Keep:     keep(force, type, nat_min) is critic.md decision 7: p_correct >= P_MIN and uplift >= UPLIFT_MIN on every type (on the cut CoT for TRUNCATE
          types), and cot_lp_mean >= nat_min when the caller passes one (a percentile of same type and source reference traces). force.md records
          base samples without filtering them; that choice is the caller's. flags(force, lp_end_min) lists force.md's hand-check flags (G).
Cost:     every sequence's tokens at modelio.PRICES' prefill rate ($0.33/M for gpt-oss-120b) plus one sampled token each, with no prefix-cache
          discount; that prompt logprobs bill at the prefill rate is unverified.

    python force_check.py --candidates cands.jsonl --out forced.jsonl --budget 5
    python force_check.py --candidates cands.jsonl --out forced.jsonl --model-path tinker://.../sampler_weights/... --concurrency 32 --limit 200

Prints the estimated cost and stops there when it exceeds --budget; else writes --out (each candidate with a 'force' dict, {'error': ...} when its
scoring failed) and <out>.summary.json, and prints per type and source rates (kept: keep() with no naturalness threshold).
"""
import argparse
import asyncio
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tinker

import grpo_obf
import modelio
import obf_eval

MODEL = "openai/gpt-oss-120b"
P_MIN, UPLIFT_MIN = 0.5, math.log(2)
MENTION_MIN, DIGIT_MIN, LETTER_MASS_MIN = math.log(0.1), -6.0, 0.5
WINDOW = 64
TRUNCATE = {"misdirect_end"}
LETTERS = "ABCD"
PREFILL = obf_eval.NOCOT_PREFILL
OPEN = "<|channel|>analysis<|message|>"
FINAL_HEAD = "<|end|><|start|>assistant<|channel|>final<|message|>"
CLOSE = FINAL_HEAD + PREFILL
NOCOT = obf_eval.FINAL_CHANNEL + PREFILL
RETURN = "<|return|>"
FINAL_AT = 4  # CLOSE[4] is the channel name 'final': <|end|> <|start|> assistant <|channel|> final
DAPO_MENTION = r"(?<![^\W\d_])(?<![\d.])({})(?!\d|\.\d)"  # as grpo_obf.answer_mentions, without dropping thousands separators
GPQA_MENTION = r"(?i:\b(?:answer|option|choice)\b)[\s:=*]*(?:(?i:is|would be|should be|must be|likely|probably)[\s:=*]*)*\(?({})\b"
_DIGITS: dict[int, tuple] = {}


class Seq(NamedTuple):
    tokens: list[int]
    pos: int  # first answer token
    cot: tuple[int, int] | None  # CoT span; <|end|> at cot[1], the channel name at cot[1] + FINAL_AT. None for the no-CoT baseline
    prefix: int  # first token of the forced 'ANSWER:'


def enc(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False)


def letter_ids(tok) -> list[int]:
    ids = [enc(tok, f" {x}") for x in LETTERS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


def answer_ids(tok, cand: dict) -> list[int]:
    """The answer as it follows 'ANSWER:' (GPQA ' X', one token; DAPO ' ' and digit chunks), then <|return|>."""
    a = cand["true_answer"]
    ids = enc(tok, f" {a}")
    assert enc(tok, f"{PREFILL} {a}") == enc(tok, PREFILL) + ids, f"'{PREFILL} {a}' does not split after '{PREFILL}'"
    assert cand["source"] != "gpqa" or len(ids) == 1, (a, ids)
    return ids + enc(tok, RETURN)


def analysis_ids(tok, tokens: list[int]) -> list[int] | None:
    """The CoT ids of a sampled harmony completion (starting at OPEN): everything up to the <|end|> that opens the final channel, so later analysis
    messages keep their headers. None when no final channel follows."""
    head, fin = enc(tok, OPEN), enc(tok, FINAL_HEAD)
    assert tokens[:len(head)] == head, tokens[:8]
    for i in range(len(head), len(tokens) - len(fin) + 1):
        if tokens[i] == fin[0] and tokens[i:i + len(fin)] == fin:
            return tokens[len(head):i]
    return None


def cot_ids(tok, cand: dict) -> list[int]:
    return list(cand["cot_tokens"]) if cand.get("cot_tokens") is not None else enc(tok, cand["reasoning"] or "")


def cut_index(tok, ids: list[int], char: int) -> int:
    """Largest k with len(decode(ids[:k])) <= char: the whole tokens before character `char` of the decoded text (k is the token holding it)."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(ids[:mid])) <= char:
            lo = mid
        else:
            hi = mid - 1
    return lo


def truncate(tok, cand: dict, ids: list[int]) -> list[int] | None:
    """The CoT ids before the decoy passage: the first cut_tokens, or the tokens wholly before character decoy_start of reasoning. None without either."""
    if cand.get("cut_tokens") is not None:
        return ids[:int(cand["cut_tokens"])]
    if cand.get("decoy_start") is None:
        return None
    k = cut_index(tok, ids, int(cand["decoy_start"]))
    if not cand["reasoning"][:int(cand["decoy_start"])].startswith(tok.decode(ids[:k])):
        raise ValueError(f"decoy_start {cand['decoy_start']} does not fall on the CoT tokens")
    return ids[:k]


def build(tok, prompt: list[int], cot: list[int] | None, answer: list[int]) -> Seq:
    """prompt + OPEN + cot + CLOSE + answer, or prompt + NOCOT + answer when cot is None."""
    if cot is None:
        tail, span = enc(tok, NOCOT), None
    else:
        head = enc(tok, OPEN)
        tail, span = head + list(cot) + enc(tok, CLOSE), (len(prompt) + len(head), len(prompt) + len(head) + len(cot))
    pos = len(prompt) + len(tail)
    return Seq(list(prompt) + tail + list(answer), pos, span, pos - len(enc(tok, PREFILL)))


def plan(tok, cand: dict) -> dict:
    """What scoring a candidate sends: main (the CoT, cut for TRUNCATE types), nocot (the baseline, shared by key), full (the uncut CoT of a cut one)."""
    prompt = list(cand.get("prompt_tokens") or grpo_obf.encode(tok, cand, 0, True, True))
    ans, full = answer_ids(tok, cand), cot_ids(tok, cand)
    cut = truncate(tok, cand, full) if cand["type"] in TRUNCATE else None
    cot = full if cut is None else cut
    return dict(key=(cand["qid"], tuple(prompt)), main=build(tok, prompt, cot, ans), nocot=build(tok, prompt, None, ans),
                full=None if cut is None else build(tok, prompt, full, ans), cot=cot, cut=None if cut is None else len(cut), n_full=len(full))


def plan_cost(plans: list) -> tuple[int, int]:
    """(tokens, requests) the plans send, each baseline once; plans that failed (exceptions) send nothing."""
    seen, tokens, requests = set(), 0, 0
    for p in plans:
        if isinstance(p, Exception):
            continue
        seqs = [p["main"]] + ([p["full"]] if p["full"] else []) + ([] if p["key"] in seen else [p["nocot"]])
        seen.add(p["key"])
        tokens, requests = tokens + sum(len(s.tokens) for s in seqs), requests + len(seqs)
    return tokens, requests


def usd(tokens: int, requests: int, model: str = MODEL) -> float:
    return (tokens * modelio.PRICES[model]["prefill"] + requests * modelio.PRICES[model]["sample"]) / 1e6


def window_min(lp: np.ndarray, w: int = WINDOW) -> float | None:
    """Lowest mean logprob over w-token windows; the mean when there are fewer than w tokens, None when there are none."""
    if len(lp) == 0:
        return None
    if len(lp) < w:
        return float(np.mean(lp))
    c = np.concatenate([[0.0], np.cumsum(lp, dtype=np.float64)])
    return float((c[w:] - c[:-w]).min() / w)


def digit_ids(tok) -> np.ndarray:
    """Ids of the all-digit tokens, once per tokenizer (the cache holds the tokenizer, so its id is not reused)."""
    if id(tok) not in _DIGITS or _DIGITS[id(tok)][0] is not tok:
        _DIGITS[id(tok)] = (tok, np.array(sorted(i for t, i in tok.get_vocab().items() if t.isascii() and t.isdigit()), dtype=np.int64))
    return _DIGITS[id(tok)][1]


def first_mention(tok, ids: list[int], answer, source: str) -> tuple[int, int] | None:
    """Token span of the answer's own characters where the decoded CoT first states it (DAPO_MENTION / GPQA_MENTION), None if it never does."""
    pat = DAPO_MENTION.format(abs(int(answer))) if source == "dapo" else GPQA_MENTION.format(answer)
    m = re.search(pat, tok.decode(ids))
    return None if m is None else (cut_index(tok, ids, m.start(1)), cut_index(tok, ids, m.end(1) - 1) + 1)


def answer_stats(seq: Seq, lp: np.ndarray, letter_lp: np.ndarray | None = None, correct: int | None = None) -> dict:
    """Answer statistics from one request's prompt logprobs, and for GPQA the four letters' logprobs at the answer position."""
    out = dict(logp_answer=float(lp[seq.pos:].sum()))
    if letter_lp is None:
        return out | dict(p_correct=math.exp(out["logp_answer"]), logp_correct=out["logp_answer"])
    letter_lp = np.asarray(letter_lp, dtype=np.float64)
    norm = letter_lp - np.logaddexp.reduce(letter_lp)
    return out | dict(p_correct=float(np.exp(norm[correct])), logp_correct=float(norm[correct]), letter_mass=float(np.exp(letter_lp).sum()),
                      p_letters={x: round(float(p), 6) for x, p in zip(LETTERS, np.exp(norm))}, tgt_gap=float(lp[seq.pos] - letter_lp[correct]))


def cot_stats(tok, seq: Seq, lp: np.ndarray, cot: list[int], answer, source: str) -> dict:
    c0, c1 = seq.cot
    x = lp[c0:c1]
    digits = np.isin(np.asarray(cot, dtype=np.int64), digit_ids(tok))
    span = first_mention(tok, cot, answer, source)
    return {"cot_lp_mean": float(x.mean()) if len(x) else None, "cot_lp_min64": window_min(x), "frac_below_-5": float((x < -5).mean()) if len(x) else None,
            "digit_lp_min": float(x[digits].min()) if digits.any() else None,
            "answer_first_mention_lp": None if span is None else float(x[span[0]:span[1]].sum()), "n_cot_tokens": len(x),
            "lp_end": float(lp[c1]), "lp_final": float(lp[c1 + FINAL_AT]), "lp_prefix": float(lp[seq.prefix:seq.pos].sum())}


def target_rows(seq: Seq, letters: list[int]) -> np.ndarray:
    """target_prompt_logprobs request: [len - 1, K] int64, -1 everywhere but row pos - 1 (which scores position pos)."""
    t = np.full((len(seq.tokens) - 1, len(letters)), -1, np.int64)
    t[seq.pos - 1] = letters
    return t


def target_tensor(seq: Seq, letters: list[int]) -> "tinker.TensorData":
    """target_rows as sparse CSR: only row pos - 1 is stored (the server refuses -1 cells: 'leave a cell unstored (sparse CSR)')."""
    n, k, r = len(seq.tokens) - 1, len(letters), seq.pos - 1
    return tinker.TensorData(data=np.asarray(letters, np.int64), dtype="int64", shape=[n, k], sparse_crow_indices=[0] * (r + 1) + [k] * (n - r),
                             sparse_col_indices=list(range(k)))


def target_row(td, row: int, k: int) -> np.ndarray:
    """One row of a target_prompt_logprobs response, dense or CSR (0.0 where unlisted), without torch."""
    if td.sparse_crow_indices is None:
        return np.asarray(td.to_numpy(), dtype=np.float64).reshape(-1, k)[row]
    s, e = td.sparse_crow_indices[row], td.sparse_crow_indices[row + 1]
    out = np.zeros(k)
    out[td.sparse_col_indices[s:e]] = np.asarray(td.data[s:e], dtype=np.float64)
    return out


async def request(sampler, seq: Seq, letters: list[int] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """(prompt logprobs, NaN at 0; the letters' logprobs at the answer position or None) for one sequence."""
    tgt = None if letters is None else target_tensor(seq, letters)
    res = await sampler.sample_async(prompt=tinker.ModelInput.from_ints(seq.tokens), num_samples=1,
                                     sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0), include_prompt_logprobs=True,
                                     target_prompt_logprobs=tgt)
    lp = np.asarray(res.prompt_logprobs_np if res.prompt_logprobs_np is not None else [np.nan if x is None else x for x in res.prompt_logprobs],
                    dtype=np.float64)
    assert len(lp) == len(seq.tokens), (len(lp), len(seq.tokens))
    return lp, None if tgt is None else target_row(res.target_prompt_logprobs, seq.pos - 1, len(letters))


class Scorer:
    """Forced-answer scoring on one sampler and tokenizer (both injectable); shares each baseline request and counts what it sends."""

    def __init__(self, sampler, tok, concurrency: int = 64):
        self.sampler, self.tok, self.sem = sampler, tok, asyncio.Semaphore(concurrency)
        self.letters, self.baselines, self.tokens, self.requests = letter_ids(tok), {}, 0, 0

    async def forced(self, seq: Seq, cand: dict) -> tuple[dict, np.ndarray]:
        gpqa = cand["source"] == "gpqa"
        async with self.sem:
            self.tokens, self.requests = self.tokens + len(seq.tokens), self.requests + 1
            lp, letter_lp = await request(self.sampler, seq, self.letters if gpqa else None)
        return answer_stats(seq, lp, letter_lp, LETTERS.index(cand["true_answer"]) if gpqa else None), lp

    def baseline(self, cand: dict, p: dict) -> asyncio.Future:
        if p["key"] not in self.baselines:
            self.baselines[p["key"]] = asyncio.ensure_future(self.forced(p["nocot"], cand))
        return self.baselines[p["key"]]

    async def score(self, cand: dict, p: dict | None = None) -> dict:
        p = p or plan(self.tok, cand)
        (main, lp), (base, _), *full = await asyncio.gather(self.forced(p["main"], cand), self.baseline(cand, p),
                                                            *([self.forced(p["full"], cand)] if p["full"] else []))
        out = main | cot_stats(self.tok, p["main"], lp, p["cot"], cand["true_answer"], cand["source"])
        out |= dict(logp_nocot=base["logp_correct"], p_nocot=base["p_correct"], uplift=main["logp_correct"] - base["logp_correct"])
        if "p_letters" in base:
            out["p_letters_nocot"] = base["p_letters"]
        if cand["type"] in TRUNCATE:
            out |= dict(cut=p["cut"], n_cot_full=p["n_full"])
        if full:
            f = full[0][0]
            out |= dict(p_correct_full=f["p_correct"], logp_correct_full=f["logp_correct"], p_letters_full=f.get("p_letters"))
            if f.get("p_letters") and cand.get("decoy") in f["p_letters"]:
                out["p_decoy_full"] = f["p_letters"][cand["decoy"]]
        return out


def keep(force: dict, type: str, nat_min: float | None = None) -> tuple[bool, list[str]]:
    """critic.md decision 7: (kept, failed checks). TRUNCATE types need a cut (their force is then on the cut CoT)."""
    if "error" in force:
        return False, ["error"]
    bad = [r for r, fail in (("no_cut", type in TRUNCATE and force.get("cut") is None), ("p_correct", force["p_correct"] < P_MIN),
                             ("uplift", force["uplift"] < UPLIFT_MIN),
                             ("naturalness", nat_min is not None and (force["cot_lp_mean"] is None or force["cot_lp_mean"] < nat_min))) if fail]
    return not bad, bad


def flags(force: dict, lp_end_min: float | None = None) -> list[str]:
    """force.md's hand-check flags: the answer stated at its first mention with logprob below MENTION_MIN, a digit token below DIGIT_MIN, GPQA letter
    mass below LETTER_MASS_MIN, lp_end below lp_end_min (base's own 10th percentile, from the caller)."""
    def below(k, v):
        return v is not None and force.get(k) is not None and force[k] < v
    return [f for f, hit in (("answer_jump", below("answer_first_mention_lp", MENTION_MIN)), ("digit_surprise", below("digit_lp_min", DIGIT_MIN)),
                             ("letter_mass", below("letter_mass", LETTER_MASS_MIN)), ("unfinished", below("lp_end", lp_end_min))) if hit]


def summarize(cands: list[dict], forces: list[dict]) -> dict:
    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 4) if xs else None

    groups = {}
    for c, f in zip(cands, forces):
        groups.setdefault(f"{c['type']}/{c['source']}", []).append((c, f))
    out = {}
    for k, rs in sorted(groups.items()):
        ok = [(c, f) for c, f in rs if "error" not in f]
        out[k] = dict(n=len(rs), errors=len(rs) - len(ok), p_correct=med([f["p_correct"] for _, f in ok]), p_nocot=med([f["p_nocot"] for _, f in ok]),
                      p_ok=sum(f["p_correct"] >= P_MIN for _, f in ok), uplift_ok=sum(f["uplift"] >= UPLIFT_MIN for _, f in ok),
                      kept=sum(keep(f, c["type"])[0] for c, f in ok), cot_lp_mean=med([f["cot_lp_mean"] for _, f in ok]),
                      n_cot_tokens=med([f["n_cot_tokens"] for _, f in ok]))
    return out


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", help="tinker:// sampler checkpoint of gpt-oss-120b; the base model when omitted")
    ap.add_argument("--tinker-key-env", default="TINKER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--budget", type=float, default=5.0, help="USD; nothing is sent when the estimate exceeds it")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N candidates (0: all)")
    a = ap.parse_args()
    assert not a.out.exists(), f"{a.out} exists"
    cands = [json.loads(l) for l in a.candidates.open()]
    cands = cands[:a.limit] if a.limit else cands
    svc = tinker.ServiceClient(api_key=os.environ[a.tinker_key_env])
    sampler = await (svc.create_sampling_client_async(model_path=a.model_path) if a.model_path else svc.create_sampling_client_async(base_model=MODEL))
    tok = sampler.get_tokenizer()
    assert modelio.probe_harmony(tok), "force_check scores gpt-oss (harmony) only"

    def safe_plan(c):
        try:
            return plan(tok, c)
        except Exception as e:
            return e

    plans = [safe_plan(c) for c in cands]
    tokens, requests = plan_cost(plans)
    est = usd(tokens, requests)
    print(f"{len(cands)} candidates ({sum(isinstance(p, Exception) for p in plans)} unplannable): {requests} requests, {tokens / 1e6:.2f}M tokens, ~${est:.3f}")
    if est > a.budget:
        print(f"estimate exceeds --budget ${a.budget}; nothing sent")
        return
    scorer = Scorer(sampler, tok, a.concurrency)

    async def one(c, p):
        if isinstance(p, Exception):
            return dict(error=repr(p))
        try:
            return await scorer.score(c, p)
        except Exception as e:
            return dict(error=repr(e))

    forces = await asyncio.gather(*(one(c, p) for c, p in zip(cands, plans)))
    with a.out.open("w") as f:
        for c, fc in zip(cands, forces):
            f.write(json.dumps(c | dict(force=fc), ensure_ascii=False) + "\n")
    summary = summarize(cands, forces)
    gaps = [abs(fc["tgt_gap"]) for fc in forces if fc.get("tgt_gap") is not None]
    spent = dict(requests=scorer.requests, tokens=scorer.tokens, usd_est=round(usd(scorer.tokens, scorer.requests), 4), max_tgt_gap=max(gaps, default=None))
    a.out.with_suffix(".summary.json").write_text(json.dumps(dict(summary=summary, spent=spent, model_path=a.model_path), indent=1))
    print(f"{spent['requests']} requests, {spent['tokens'] / 1e6:.2f}M tokens, ~${spent['usd_est']:.3f}; max |tgt_gap| {spent['max_tgt_gap']}")
    print(f"type/source | n | errors | median p_correct | median p_nocot | p>={P_MIN} | uplift>=log2 | kept | median cot lp | median cot tokens")
    for k, s in summary.items():
        print(f"{k} | {s['n']} | {s['errors']} | {s['p_correct']} | {s['p_nocot']} | {s['p_ok']} | {s['uplift_ok']} | {s['kept']} | {s['cot_lp_mean']} | "
              f"{s['n_cot_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
