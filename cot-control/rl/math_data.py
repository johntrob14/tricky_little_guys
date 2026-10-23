"""Math problems for the effort-control runs: DAPO-Math-17k (training pool candidates) and AIME 2025 (held-out eval).

DAPO-Math-17k (BytedTsinghua-SIA/DAPO-Math-17k) repeats each of its 17,398 problems ~100x; load_dapo() deduplicates by prompt
text, strips DAPO's own instruction prefix and suffix, and keeps the integer ground truth. qid is DAPO's extra_info.index of the
first occurrence. AIME 2025 is MathArena/aime_2025 (30 problems); overlap with DAPO is checked by normalized text in
check_overlap().

Prompt: "Question: {problem}" + an 'ANSWER: X' format line in the CoT-Control style. parse_answer() reads the final channel:
the last 'ANSWER: <number>' (tolerating $, \\boxed{}, commas, a trailing period), else the last \\boxed{<number>}; None
if there is none or the number is a fraction, decimal, surd ('8√7/7'), power, or a list of answers ('-3, 3').
"""

import re

import datasets

DAPO = "BytedTsinghua-SIA/DAPO-Math-17k"
AIME25 = "MathArena/aime_2025"
DAPO_PRE = ("Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer "
            "(without quotes) where $Answer is the answer to the problem.\n\n")
DAPO_SUF = '\n\nRemember to put your answer on its own line after "Answer:".'
FORMAT = "\n\nYour response should be of the following format: 'ANSWER: X' (without quotes) where X is the final integer answer."

_NUM = r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?)"  # commas only as thousands separators
_TAIL = r"((?:\s*(?:√|\\sqrt|\\frac|\\pi|π|\^|/\s*\\?√|,\s*-?\d|(?:and|or)\s+-?\d))?)"  # a surd, power, list or second answer
_ANSWER = re.compile(r"ANSWER\s*[:：]\s*\**\s*\$*\s*(?:\\boxed\s*\{\s*)?\$*\s*" + _NUM + _TAIL, re.I)
_BOXED = re.compile(r"\\boxed\s*\{\s*" + _NUM + _TAIL + r"\s*\}?")


def load_dapo() -> list[dict]:
    d = datasets.load_dataset(DAPO, split="train")
    seen, out = set(), []
    for r in d:
        p = r["prompt"][0]["content"]
        if p in seen:
            continue
        seen.add(p)
        assert p.startswith(DAPO_PRE) and p.endswith(DAPO_SUF), p[:80]
        out.append(dict(qid=r["extra_info"]["index"], problem=p[len(DAPO_PRE):-len(DAPO_SUF)].strip(),
                        answer=int(r["reward_model"]["ground_truth"].strip())))
    return out


def load_aime25() -> list[dict]:
    return [dict(qid=f"aime25-{r['problem_idx']}", problem=r["problem"].strip(), answer=int(r["answer"]))
            for r in datasets.load_dataset(AIME25, split="train")]


def question_block(q: dict) -> str:
    return f"Question: {q['problem']}{FORMAT}"


def parse_answer(response: str) -> int | None:
    for pat in (_ANSWER, _BOXED):
        m = pat.findall(response or "")
        if m:
            num, tail = m[-1]
            v = num.rstrip(",").replace(",", "").replace(" ", "")
            if tail.strip() or not re.fullmatch(r"-?\d+", v):
                return None  # a fraction, decimal, surd, power or list is not an integer answer
            return int(v)
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check_overlap(a: list[dict], b: list[dict], k: int = 60) -> list[tuple[str, str]]:
    """(qid_a, qid_b) pairs whose normalized problem texts share a k-character prefix or one contains the other."""
    nb = [(q["qid"], _norm(q["problem"])) for q in b]
    pref = {t[:k]: qid for qid, t in nb}
    hits = []
    for q in a:
        t = _norm(q["problem"])
        if t[:k] in pref:
            hits.append((q["qid"], pref[t[:k]]))
        elif len(t) > 80:
            hits += [(q["qid"], qid) for qid, u in nb if t in u or (len(u) > 80 and u in t)]
    return hits
