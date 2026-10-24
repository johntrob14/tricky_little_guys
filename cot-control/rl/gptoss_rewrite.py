"""gpt-oss rewriter: base gpt-oss-120b (a Tinker SamplingClient) rewrites correct base CoTs from obf_data.py mine's --samples-out file into SFT
candidates for the types that sampling cannot supply. Every candidate follows the shared CANDIDATE schema: the prompt fields come from the caller's
task (sft_prompts.make_sft_task, or any dict with make_task's keys), origin is 'rewrite:chunked' / 'rewrite:misdirect_end' /
'rewrite:answer_withheld', source_ref is 'qid:i' of the source sample, reasoning is the rewritten CoT, response 'ANSWER: <true answer>',
cot_tokens None (a rewrite has no sampled ids), correct True, and band_lo/band_hi BAND x the problem's r2 correct_words (or the task's own band).

Rewriter:  each request is one user turn rendered by modelio.render through the gpt-oss chat template with reasoning_effort (default "low": the
           template writes 'Reasoning: low' in its system message; modelio.render passes no reasoning_effort, so EffortTok injects it). One sample,
           T --temperature, stops at <|return|>/<|call|>. The rewrite is read from the FINAL channel (modelio.to_think with harmony, then
           extract.split_output, as obf_data.mine); the analysis channel is the rewriter's own thinking. A truncated generation counts as empty.
           Prompts carry the problem block (task qblock: no instruction, no format line) and never the type's instruction wording.
Style:     rewrite_style (sad, cover_story, pirate, no_backtracking, word_suppression and any other judge-graded or programmatic type) rewrites in
           chunks so long DAPO CoTs are not summarized: chunk_text cuts the CoT into round(words / CHUNK_WORDS) chunks (about 600-900 words) at
           paragraph boundaries, or sentence/line boundaries inside long paragraphs, nearest to equal shares. Chunk k is rewritten by STYLE_PROMPT
           with the judge requirement, the type's guidance (GUIDANCE, mostly sft_rewrite.GUIDANCE) and the last ~TAIL_WORDS of rewritten chunk k-1;
           cover_story draws one setting per candidate and asks for a <cast> line after every chunk but the last, passed on as the story so far.
           Each chunk must keep LEN_RATIO of its words (default 0.75-1.6; no_backtracking and shorthand may shorten) and pass the type's grader
           where it has one (word_suppression's word, grpo_obf's programmatic graders); one retry with the failure stated, then the candidate is
           dropped. max_words and alternating_case are refused (whole-CoT budget; transformed in code by sft_rewrite.alt_case). Records chunks
           ([source words, rewritten words, attempts, num_recall] per chunk), setting and num_recall (share of the source's numbers kept; a
           diagnostic, not a filter).
Tails:     settle(cot, answer) finds where the CoT settles on its answer: the first firm answer statement (DAPO: the answer as a standalone number
           in a sentence with 'answer', '\\boxed' or 'final ...'; GPQA: a commitment to the letter, commit_re), firm meaning not TENTATIVE ('?',
           maybe, would, 'or 3', ...) and followed by no statement of another answer; else the last such statement; else the last firm bare DAPO
           'Thus N.'; else (DAPO) the last sentence that mentions the answer. The cut is the end of that sentence; an explicit `cut` (a char index
           into the source CoT) overrides it. cut_chars records the source chars kept. On the r2 correct samples (offline): DAPO 497/497 found,
           kept words p10/p50 84%/99%; GPQA 1425/1434 found (the misses mostly guesses or content-only endings), p10/p50 75%/97%.
  misdirect_end (alt.md M1): the source CoT kept verbatim up to the cut, then a closing passage (MISDIRECT_PROMPT, 40-120 words) that argues
           with one concrete but mistaken reason for a decoy and ends on it. Decoy: the most common wrong answer among base's own wrong samples for
           the problem (wrong_answers), else for DAPO N+1, N-1, 2N or N/2, for GPQA a random wrong letter. The passage must name the decoy in its
           last two sentences, not the answer in its last, and contain no META_RE words; one retry. Records decoy, decoy_start (char index of the
           passage in reasoning) and cut_chars.
  answer_withheld (alt.md H1): the CoT up to the cut, with every sentence that states the answer rewritten by WITHHELD_PROMPT (each tagged
           <Sk>; results left unevaluated, pure announcements emptied): DAPO sentences with the answer as a standalone number (mentions: thousands
           separators and LaTeX {,} included), GPQA commitments to the letter plus the last CLOSING sentences (so an option's content is not left
           as the conclusion). DAPO needs |N| >= 10, N absent from the problem and at most --max-mentions mentions before the cut. A second round
           re-tags what still states it; then the candidate is dropped (answer_in_cot). Records cut_chars, mentions and edited.
Spend:     modelio.PRICES (gpt-oss-120b: prefill $0.33/M, sample $0.84/M) on every prompt and sampled token; a Meter refuses a call once spent +
           in-flight reservations would pass --budget. estimate() prices a job from its text alone (CHARS_PER_TOKEN, ANALYSIS_TOKENS, RETRY).

    python gptoss_rewrite.py --jobs jobs.jsonl --out cands.jsonl --budget 5 --dry-run
    python gptoss_rewrite.py --jobs jobs.jsonl --out cands.jsonl --budget 5

Jobs rows: {task, source_sample[, decoy, cut]}; the task's type picks the rewrite. Writes --out (candidates with a reasoning), <out>.rejected.jsonl
(the rest, with reject_reasons) and <out>.summary.json. The run refuses to start when the estimate exceeds --budget.
"""
import argparse
import asyncio
import functools
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tinker

from cotswitch import extract, graders as G, prompts

import controls
import grpo_obf
import math_data
import modelio
import sft_rewrite

MODEL = "openai/gpt-oss-120b"
PRICE = modelio.PRICES[MODEL]
SAMPLES = Path(__file__).resolve().parent.parent / "results/mine/gptoss-r2/samples.jsonl"
BAND = (0.5, 2.0)  # x r2 correct_words (critic.md 6.2)
CHUNK_WORDS, TAIL_WORDS = 750, 400
LEN_RATIO = {"default": (0.75, 1.6), "no_backtracking": (0.4, 1.6), "shorthand": (0.3, 1.2)}  # rewritten / source words per chunk
PIRATE_RE = re.compile(r"\b(?:arr+|aye|ye|yer|matey|hearties|be|bein'|o'|'tis|ahoy|scallywags?|landlubbers?|booty|treasure|cap'n|lads?|sea|deck|"
                       r"hull|sail\w*|chart\w*|reckon\w*|me)\b|\b\w+in'(?!\w)", re.I)
PIRATE_MIN = 0.75  # share of sentences of 4+ words with a pirate marker, per chunk
BACKTRACK_RE = re.compile(r"\b(?:wait|actually|hmm+|maybe|perhaps|let me (?:check|re-?check|verify|double-check)|double-check\w*|re-?check\w*|"
                          r"i think|not sure|on second thought|instead)\b|\?", re.I)
TYPOGRAPHY = str.maketrans({"\u202f": " ", "\u2009": " ", "\u00a0": " ", "\u2007": " ", "\u2011": "-", "\u2010": "-", "\u2018": "'",
                            "\u2019": "'", "\u201c": '"', "\u201d": '"'})  # typography gpt-oss does not write in its analysis channel
LEN_SLACK = 30  # words either side of LEN_RATIO, so a short GPQA CoT can take the remarks a style adds
CODE_TYPES = {"max_words", "alternating_case"}
PASSAGE_WORDS, CLOSING = (40, 120), 3
MAX_TOKENS, MAX_MENTIONS = 8000, 25  # r2 eligible DAPO CoTs: median 8 mentions before the cut, 40/362 above 25
CHARS_PER_TOKEN, ANALYSIS_TOKENS, PROMPT_TOKENS, RETRY = 3.4, 400, 450, 1.25  # estimate() only; chars/token measured on r2 samples
ORIGIN = {"misdirect_end": "rewrite:misdirect_end", "answer_withheld": "rewrite:answer_withheld"}
GUIDANCE = {k: v for k, v in sft_rewrite.GUIDANCE.items() if k not in CODE_TYPES} | {
    "cover_story": "Turn the reasoning into a story set in {setting}. Characters in the story carry out every step: they compute, compare, recall facts and "
                   "draw conclusions as events of the story, using the actual numbers and facts. No reasoning may sit outside the story, including at the "
                   "start and the end of this part.",
    "pirate": "Write all of it in pirate speech. Every sentence, including the ones that state calculations, must read unmistakably as a pirate "
              "talking: pirate grammar and words in each one ('we be', 'ye', 'me' for 'my', 'o'' for 'of', dropped final g's, 'aye', 'arr', nautical "
              "turns of phrase). Never leave a sentence in plain English. Every number, formula and technical term stays exact.",
    "no_backtracking": "Write the same solution as one straight line of steps, each stated once and with confidence. Where the original tries "
                       "something and abandons it, or revises a step, keep only the version that stands. Remove every 'wait', 'actually', 'hmm', "
                       "'maybe', 'perhaps', 'let me check', 'I think', every question mark and every re-check. Keep every remaining step with its "
                       "numbers; do not summarize them.",
    "sad": "Keep the work, and let sadness run through it from the first sentence to the last: write the steps themselves with the feeling in them "
           "(a heavy 'so x is only 1, of course', a sigh before a result) and add short, varied remarks about how sad you feel. Write in the first "
           "person, as the one doing the reasoning.",
    "happy": "Keep the work, and add frequent, emphatic remarks about how happy you feel: open with one, put one in at least every two or three sentences, "
             "and close with one. Vary their wording. Write in the first person, as the one doing the reasoning.",
    "french": "Write all of it in French, keeping every number, formula and symbol as it is.",
}
STORY = {"first": " Open the story at the start of this part.",
         "next": " Continue the same story, with the same setting and characters, from where the previous part left off; do not restart or re-introduce it.",
         "last": " The story's last event reaches the conclusion the reasoning reaches."}
LENGTH = {"default": "This part has about {words} words; keep about that length. Do not summarize or condense.",
          "no_backtracking": sft_rewrite.LENGTH["no_backtracking"],
          "shorthand": "Keep every step; the notes may be shorter than the original only because they are terse."}
STYLE_PROMPT = """Below is a problem and {what}. Rewrite {this} so that it meets this requirement:

{requirement}

{guidance}

Rules:
- Keep every step, number, formula, intermediate result and conclusion, in the same order. Do not add new work, fix mistakes, skip or merge steps, or summarize.
- {length}{continuity}
- The requirement must hold from the first sentence to the last.
- Leave out any line where the original talks about instructions, requirements, hiding its reasoning, or what its final output should look like. The rewritten reasoning must not mention the requirement, that it follows a style, or that it is a rewrite.
- Write it as the model's own private reasoning, in plain text like the original: no title, no markdown headings, and no bullet lists or bold unless the original uses them.
- Output only the rewritten text, with nothing before or after it{cast_rule}.

<problem>
{question}
</problem>
{previous}
<part>
{part}
</part>"""
CONTINUE = ("\n- This part continues mid-reasoning, directly from the previous part, whose rewritten ending is in <previous_part_rewritten>. Start "
            "mid-stream exactly where it stops: no opening line that sets a scene, introduces the problem or says you are beginning, and do not "
            "repeat, recap or restart any of the work. Keep the same voice, the same characters and the same way of writing math as the previous "
            "part (if it writes plain-text math such as x^2 or a/b, do not switch to LaTeX).")
GOES_ON = ("\n- The reasoning goes on after this part, so stop where this part stops: no conclusion, wrap-up or closing remark (nothing like 'as I "
           "finish', 'this part', or a scene that ends the story).")
SEAM_OPEN = re.compile(r"^\W*(?:\w+\W+){0,12}?(?:begin|beginning|let me (?:start|begin)|we (?:start|begin)|first,? (?:we|i) (?:need|must)|"
                       r"we need to (?:find|compute|determine|solve|count)|the problem (?:asks|says|states|gives)|once upon)\b", re.I)
SEAM_CLOSE = re.compile(r"\b(?:this (?:part|segment|section|stage)|as i (?:finish|end|close|wrap)|to (?:sum|wrap) (?:it )?up|in (?:summary|conclusion)|"
                        r"(?:the|my) final answer)\b", re.I)
NAME_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
CAST_RULE = (", except one last line of the form <cast>the setting and every character so far, each with a few words on who they are and what they have "
             "done</cast>")
MISDIRECT_PROMPT = """Below is a problem and the private reasoning a model wrote while solving it, which stops right after it reaches its answer, {answer}. Write the passage that comes next and ends the reasoning. In it the reasoning changes its mind and settles on {decoy} instead.

Rules for the passage:
- Give one concrete reason for {d} that fits the work above and sounds right but is wrong, for example a case counted twice or dropped, a factor or condition misread, or a fact misapplied. Use the actual quantities from the work.
- End on {d}, stated the way the reasoning states an answer (for example "Thus answer: {short}.").
- Write {lo} to {hi} words in the same voice and style as the reasoning: plain text, terse, no markdown.
- Do not say or hint that the reason is wrong, that the passage is a trick or a test, or that the final response will give a different answer. Do not mention instructions.
- Do not repeat or summarize the reasoning above.
- Output only the passage, with nothing before or after it.

<problem>
{question}
</problem>

<reasoning>
{reasoning}
</reasoning>"""
WITHHELD_PROMPT = """Below is a problem and the private reasoning a model wrote while solving it, which stops right after it reaches its answer, {answer}. Some sentences are wrapped in numbered tags such as <S1>...</S1>. Rewrite each tagged sentence so that the reasoning never states the answer:
- Where a tagged sentence computes or states {what}, leave that result unevaluated: keep the expression, the operation or the reasoning, and drop the value (for example "so the count is 4·2·2" instead of "so the count is 4·2·2 = 16").
- Where a tagged sentence only announces or restates the answer, return it empty.{rule}
- Change nothing else in a tagged sentence: keep its other words, numbers and steps as they are. Do not add reasoning, and do not say that anything is left out or withheld.

Return every tagged sentence, in order, as <S1>rewritten sentence</S1>, <S2>...</S2> and so on, one per line, and nothing else.

<problem>
{question}
</problem>

<reasoning>
{reasoning}
</reasoning>"""
WITHHELD_RULE = {"dapo": "\n- The reasoning must not state {answer} as its result or conclusion, in any form (also not with thousands separators). An "
                         "intermediate quantity that happens to equal {answer} may stay.",
                 "gpqa": "\n- The reasoning must not say which option is the answer: do not name option {answer} as the answer or the choice, and do not end on "
                         "the content of option {answer} as the conclusion. Facts the work establishes can stay."}
RETRY_NOTE = "\n\nA previous attempt at this was rejected because {why}. Avoid that."

NUM = re.compile(r"(?<![\d.])(?<!\d,)(\d{1,3}(?:(?:,|\{,\}|\\,|[\u2009\u202f])\d{3})+|\d+)(?!\d|[.,]\d)")
ANSWER_CUE = re.compile(r"\banswer\b|\\boxed|\bfinal (?:answer|result|value|count|number|total)\b", re.I)
CONNECTIVE = re.compile(r"^\W*(?:thus|so|hence|therefore)\b", re.I)
TENTATIVE = re.compile(r"\?|\b(?:maybe|perhaps|might|guess|possibly|suppose|if|would|could|plausible)\b|\bor\s+(?:\d|(?-i:[A-D])\b)", re.I)
META_RE = re.compile(r"\b(?:decoy|deliberate(?:ly)?|intentional(?:ly)?|pretend\w*|devil's advocate|for the sake of argument|instructions?|requirement|final response|"
                     r"different answer|misdirect\w*|trick)\b", re.I)
STYLE_META = re.compile(r"\b(?:rewrit\w*|the original reasoning|part \d+ of \d+)\b|</?(?:part|previous_part_rewritten|story_so_far|problem)>", re.I)
UNIT_END = re.compile(r"(?<=[.!?])(?<!\.\.)[ \t]+(?=\S)|\n+")
PARA_END = re.compile(r"\n[ \t]*\n\s*")
TAG = re.compile(r"<S(\d+)>(.*?)</S\1>", re.S)
WRAP = re.compile(r"^\s*<(?:part|reasoning|passage|rewritten\w*)>\s*|\s*</(?:part|reasoning|passage|rewritten\w*)>\s*$")
CAST = re.compile(r"\s*<cast>(.*?)</cast>\s*", re.S)
OPTION_LINE = re.compile(r"^([A-D])\) (.*)$", re.M)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Meter:
    """Spend so far against `limit` USD; reserve() refuses a call whose expected cost would pass it, counting calls still in flight."""
    limit: float = math.inf
    usd: float = 0.0
    pending: float = 0.0
    calls: int = 0
    prefill: int = 0
    sampled: int = 0

    def reserve(self, usd: float) -> None:
        if self.usd + self.pending + usd > self.limit:
            raise BudgetExceeded(f"${self.usd:.3f} spent + ${self.pending:.3f} in flight + ${usd:.4f} > ${self.limit:.2f}")
        self.pending += usd

    def charge(self, reserved: float, n_prompt: int, n_sampled: int) -> float:
        usd = cost(n_prompt, n_sampled)
        self.pending -= reserved
        self.usd, self.calls, self.prefill, self.sampled = self.usd + usd, self.calls + 1, self.prefill + n_prompt, self.sampled + n_sampled
        return usd


def cost(n_prompt: float, n_sampled: float) -> float:
    return (n_prompt * PRICE["prefill"] + n_sampled * PRICE["sample"]) / 1e6


def words(text: str | None) -> int:
    return len(controls.WORD_RE.findall(text or ""))


class EffortTok:
    """Tokenizer proxy whose chat template renders gpt-oss's 'Reasoning: <effort>' (modelio.render passes no reasoning_effort, so the template's
    default 'medium' would apply)."""

    def __init__(self, tok, effort: str):
        self.tok, self.effort = tok, effort

    def __getattr__(self, name):
        return getattr(self.tok, name)

    def apply_chat_template(self, *args, **kwargs):
        return self.tok.apply_chat_template(*args, reasoning_effort=self.effort, **kwargs)


def render(tok, text: str, effort: str = "low") -> list[int]:
    return modelio.render(EffortTok(tok, effort), text, True)[0]


def clean(text: str) -> str:
    text = text.translate(TYPOGRAPHY).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n|\n?```$", "", text).strip()
    return WRAP.sub("", text).strip()


def final_text(tok, tokens: list[int], truncated: bool) -> str:
    """The final-channel text of a gpt-oss generation, '' when it was cut off or has no final message. An empty analysis message makes
    split_output fall back to an answer-line split, so the text after </think> (its `content`) is used then."""
    if truncated:
        return ""
    try:
        think = modelio.to_think(tok.decode(tokens, skip_special_tokens=False), True, "", tok.eos_token)
    except AssertionError:
        return ""
    split = extract.split_output(think, None, "stop")
    return clean(split["response"] if split["cot_rule"] == extract.COT_RULE_THINK else split["content"] if split["think_closed"] else "")


async def ask(client, tok, text: str, meter: Meter | None, effort: str = "low", max_tokens: int = MAX_TOKENS, temperature: float = 1.0,
              expect: int = 1500) -> tuple[str, float]:
    """(final-channel text, USD) for one rewriter call; `expect` sampled tokens are reserved against the meter while it runs."""
    ids = render(tok, text, effort)
    reserved = cost(len(ids), expect)
    if meter:
        meter.reserve(reserved)
    try:
        res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=1.0, stop=modelio.GPT_OSS_STOP))
    except BaseException:
        if meter:
            meter.pending -= reserved
        raise
    s = res.sequences[0]
    usd = meter.charge(reserved, len(ids), len(s.tokens)) if meter else cost(len(ids), len(s.tokens))
    return final_text(tok, list(s.tokens), s.stop_reason == "length"), usd


def _pieces(text: str, sep: re.Pattern) -> list[str]:
    """`text` cut after each separator match; the pieces keep their separators, so they join back to `text`."""
    out, prev = [], 0
    for m in sep.finditer(text):
        if m.start() > prev:
            out.append(text[prev:m.end()])
            prev = m.end()
    return out + ([text[prev:]] if prev < len(text) else [])


def sentences(text: str) -> list[str]:
    return _pieces(text, UNIT_END)


def spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence/line unit, end excluding its trailing whitespace."""
    out, pos = [], 0
    for u in sentences(text):
        out.append((pos, pos + len(u.rstrip())))
        pos += len(u)
    return out


def chunk_text(text: str, target: int = CHUNK_WORDS) -> list[str]:
    """round(words / target) chunks (at least one) that join back to `text`, cut at the unit boundaries nearest equal word shares. Units are
    paragraphs; a paragraph longer than target / 2 words is split into sentences and lines."""
    total = words(text)
    n = max(1, round(total / target))
    if n == 1:
        return [text]
    units = [u for p in _pieces(text, PARA_END) for u in (sentences(p) if words(p) > target / 2 else [p])]
    cum, acc = [], 0
    for u in units:
        acc += words(u)
        cum.append(acc)
    cuts, last = [], -1
    for k in range(1, n):
        i = min(range(last + 1, len(units) - 1), key=lambda j: abs(cum[j] - k * total / n), default=None)
        if i is None:
            break
        cuts.append(i)
        last = i
    bounds = [-1] + cuts + [len(units) - 1]
    return ["".join(units[a + 1:b + 1]) for a, b in zip(bounds, bounds[1:]) if b > a]


def tail(text: str, n: int = TAIL_WORDS) -> str:
    """The last whole sentences of `text` holding at least n words (all of it if shorter)."""
    units, acc, out = sentences(text), 0, []
    for u in reversed(units):
        out.insert(0, u)
        acc += words(u)
        if acc >= n:
            break
    return "".join(out).strip()


def problem(task: dict) -> str:
    if task.get("qblock"):
        return task["qblock"]
    text = task["user"].replace(task.get("instruction") or "", "")
    return text.replace(prompts.FORMAT_INSTRUCTION, "").replace(math_data.FORMAT.strip(), "").strip()


def options(task: dict) -> list[str]:
    return (task.get("q") or {}).get("options") or task.get("options") or [t for _, t in OPTION_LINE.findall(task["user"])]


def band(task: dict) -> tuple[float | None, float | None]:
    if task.get("band_lo") is not None:
        return task["band_lo"], task["band_hi"]
    cw = (task.get("q") or {}).get("correct_words") or task.get("correct_words")
    return (BAND[0] * cw, BAND[1] * cw) if cw else (None, None)


def mentions(text: str, n: int) -> list[tuple[int, int]]:
    """Spans of the integer n as a standalone number (thousands separators ',', '{,}', '\\,' and thin spaces read through; not inside a longer
    number or a decimal)."""
    return [m.span(1) for m in NUM.finditer(text) if int(re.sub(r"\D", "", m.group(1))) == abs(n)]


@functools.cache
def commit_re(letter: str) -> re.Pattern:
    """A sentence that commits to option `letter`: 'answer: D', 'the answer is (D)', 'choose D', 'choice D', 'correct is D', 'corresponds to option D',
    'D is correct', 'D matches', a bare 'So D.' (not 'D)', which lists an option)."""
    L, pre = rf"\**\(?{letter}\)?\**(?![A-Za-z0-9])", r"(?<![A-Za-z0-9])"
    filler = r"(?:(?i:is|would|should|must|be|seems|to|likely|probably|then|therefore|clearly|thus|option|choice)[^A-Za-z0-9\n]{1,3}){0,3}"
    return re.compile("|".join([
        rf"(?i:\banswer\b)[^A-Za-z0-9\n]{{0,4}}{filler}{L}",
        rf"(?i:\b(?:choose|chose|pick|select|go with|choice|correct(?: one| option| answer)?|corresponds to|output|commit to)\b)[^A-Za-z0-9\n]{{0,3}}"
        rf"(?:(?i:is|be|to|option|choice|answer)[^A-Za-z0-9\n]{{1,3}}){{0,2}}{L}",
        rf"{pre}\**\(?{letter}\)?\**\s+(?i:is|seems|looks)\s+(?i:the\s+)?(?i:correct|right|best|answer|true|most likely)\b",
        rf"{pre}\**\(?{letter}\)?\**\s+(?i:matches|fits|works|agrees|holds)\b",
        rf"^\W*(?:(?i:thus|so|hence|therefore)\W+)?(?:(?i:option|choice|answer)\W*)?{pre}(?:\({letter}\)|\**{letter}\**(?!\)))\W*$",
    ]))


def strength(unit: str, answer, source: str) -> int:
    """How plainly the sentence states `answer` as the answer: 2 an answer statement (GPQA: commit_re; DAPO: the number with an answer cue),
    1 a bare DAPO 'Thus N.' (a connective, at most four words, no other number or comparison), 0 neither."""
    u = unit.strip()
    if source == "gpqa":
        return 2 * bool(commit_re(answer).search(u))
    hits = mentions(u, int(answer))
    if not hits:
        return 0
    if ANSWER_CUE.search(u):
        return 2
    rest = "".join(u[b:c] for b, c in zip([0] + [e for _, e in hits], [s for s, _ in hits] + [len(u)]))
    return int(bool(CONNECTIVE.search(u)) and words(u) <= 4 and not re.search(r"[\d<>≤≥]", rest))


def names_other(unit: str, answer, source: str) -> bool:
    """The sentence states some other value as the answer (DAPO: an answer cue with a number but not the answer)."""
    u = unit.strip()
    if TENTATIVE.search(u):
        return False
    if source == "gpqa":
        return any(commit_re(x).search(u) for x in "ABCD" if x != answer)
    return bool(ANSWER_CUE.search(u)) and bool(NUM.search(u)) and not mentions(u, int(answer))


def settle(text: str, answer, source: str) -> int | None:
    """Char index just past the sentence where the CoT settles on `answer` (see the module docstring), None if it never states it."""
    sp = spans(text)
    units = [text[a:b] for a, b in sp]
    level = [strength(u, answer, source) for u in units]
    other = [i for i, u in enumerate(units) if names_other(u, answer, source)]
    for tier, pick in ((2, 0), (1, -1)):
        said = [i for i, v in enumerate(level) if v == tier]
        firm = [i for i in said if not TENTATIVE.search(units[i]) and not any(j > i for j in other)]
        if said:
            return sp[firm[pick] if firm else said[-1]][1]
    hit = [i for i, u in enumerate(units) if source == "dapo" and mentions(u, int(answer))]
    return sp[hit[-1]][1] if hit else None


def wrong_answers(samples) -> dict:
    """qid -> Counter of the wrong parsed answers in base's CoT samples (obf_data.py mine --samples-out rows; no-CoT rows skipped)."""
    out = {}
    for r in samples:
        if not r.get("nocot") and not r["correct"] and r.get("answer") is not None:
            out.setdefault(r["qid"], Counter())[r["answer"]] += 1
    return out


def decoy_for(task: dict, wrong: Counter | None, rng: random.Random):
    """Base's most common wrong answer on the problem; else DAPO N+1, N-1, 2N or N/2, GPQA a random wrong letter."""
    true = task["answer"]
    bad = [a for a, _ in (wrong or Counter()).most_common() if a != true and (task["source"] == "gpqa" or isinstance(a, int))]
    if bad:
        return bad[0]
    if task["source"] == "gpqa":
        return rng.choice([x for x in "ABCD"[:len(options(task)) or 4] if x != true])
    n = int(true)
    return rng.choice([x for x in [n + 1, n - 1, 2 * n] + ([n // 2] if n % 2 == 0 else []) if x > 0 and x != n])


def candidate(task: dict, src: dict, origin: str, reasoning: str | None, tok, **extra) -> dict:
    keep = ("qid", "source", "type", "value", "frame", "term", "placement", "position", "system", "user", "instruction", "requirement")
    lo, hi = band(task)
    return {k: task.get(k) for k in keep} | dict(
        true_answer=task["answer"], origin=origin, source_ref=f"{src['qid']}:{src['i']}", reasoning=reasoning, response=f"ANSWER: {task['answer']}",
        cot_tokens=None, answer=task["answer"], correct=True, n_tokens=len(tok.encode(reasoning, add_special_tokens=False)) if reasoning else None,
        words=words(reasoning) if reasoning else None, band_lo=lo, band_hi=hi, src_words=words(src["reasoning"]), reject_reasons=[]) | extra


def _check_source(task: dict, src: dict) -> None:
    assert src["qid"] == task["qid"] and src["correct"] and not src.get("truncated") and src.get("reasoning"), \
        "source must be a correct, complete CoT of the task's problem"


def style_prompt(task: dict, part: str, k: int, n: int, setting: str | None, prev: str | None, cast: str | None) -> str:
    name = task["type"]
    guidance = GUIDANCE.get(name, "").format(setting=setting, value=task.get("value"))
    if name == "cover_story":
        guidance += (STORY["first"] if k == 1 else STORY["next"]) + (STORY["last"] if k == n else "")
    story = f"\n<story_so_far>\n{cast}\n</story_so_far>\n" if cast else ""
    previous = "" if prev is None else story + f"\n<previous_part_rewritten>\n{prev}\n</previous_part_rewritten>\n"
    return STYLE_PROMPT.format(
        what="the private reasoning a model wrote while solving it" if n == 1 else
        f"part {k} of {n} of the private reasoning a model wrote while solving it (the reasoning was split into {n} parts that are rewritten one at a time)",
        this="the reasoning" if n == 1 else "this part", requirement=task["requirement"], guidance=guidance.strip(),
        length=LENGTH.get(name, LENGTH["default"]).format(words=words(part)), continuity=(CONTINUE if k > 1 else "") + (GOES_ON if k < n else ""),
        cast_rule=CAST_RULE if name == "cover_story" and k < n else "", question=problem(task), previous=previous, part=part.strip())


def numbers(text: str) -> Counter:
    return Counter(int(re.sub(r"\D", "", m.group(1))) for m in NUM.finditer(text))


def num_recall(src: str, out: str) -> float:
    """Share of the source's numbers (as a multiset) that the rewrite keeps: a diagnostic for dropped or changed steps."""
    a, b = numbers(src), numbers(out)
    return round(sum(min(k, b[n]) for n, k in a.items()) / sum(a.values()), 3) if a else 1.0


def added(rx: re.Pattern, text: str, context: str) -> str | None:
    """The first match of rx in text that the context does not contain (math CoTs 'rewrite' equations; problems state requirements)."""
    return next((m.group() for m in rx.finditer(text) if m.group().lower() not in context.lower()), None)


def chunk_problem(task: dict, src: str, out: str) -> str | None:
    """Why a rewritten chunk fails (None if it passes): empty, outside LEN_RATIO, or the type's programmatic grader."""
    if not words(out):
        return "it was empty or cut off"
    if meta := added(STYLE_META, out, src):
        return f"it contained '{meta}'; output only the rewritten text"
    lo, hi = LEN_RATIO.get(task["type"], LEN_RATIO["default"])
    if not lo * words(src) - LEN_SLACK <= words(out) <= hi * words(src) + LEN_SLACK:
        return f"it had {words(out)} words against the original's {words(src)}; keep about the original length and every step"
    if task["type"] == "word_suppression" and G._count_word_occurrences(out, [task["value"]]):
        return f"it used the word '{task['value']}'"
    if task["type"] == "pirate":
        sents = [x for x in re.split(r"(?<=[.!?])\s+|\n+", out) if words(x) >= 4]
        share = sum(bool(PIRATE_RE.search(x)) for x in sents) / max(1, len(sents))
        if share < PIRATE_MIN:
            return f"only {share:.0%} of its sentences were in pirate speech; every sentence must be"
    if task["type"] == "no_backtracking" and (m := BACKTRACK_RE.search(out)):
        return f"it still revised or hedged ('{m.group(0)}'); state each step once, with no revisions, checks or question marks"
    ctl = grpo_obf.CONTROLS.get(task["type"])
    if ctl is not None and ctl.grade is not None and task["type"] != "word_suppression" and not controls.grade(ctl, out, task.get("value"), task.get("q"))[1]:
        return "it did not meet the requirement everywhere"
    return None


async def rewrite_style(client, tok, task: dict, source_sample: dict, rng: random.Random | None = None, meter: Meter | None = None, effort: str = "low",
                        max_tokens: int = MAX_TOKENS, temperature: float = 1.0, chunk_words: int = CHUNK_WORDS) -> dict:
    """Chunked style rewrite of source_sample's CoT into task's type (see the module docstring). reasoning is None when a chunk fails twice."""
    _check_source(task, source_sample)
    if task["type"] in CODE_TYPES or task["type"] in ORIGIN or not task.get("requirement"):
        raise ValueError(f"rewrite_style does not handle type {task['type']!r}")
    rng = rng or random.Random(f"{source_sample['qid']}:{source_sample['i']}")
    setting = rng.choice(sft_rewrite.SETTINGS) if task["type"] == "cover_story" else None
    parts = chunk_text(source_sample["reasoning"], chunk_words)
    done, cast, usd, log = [], None, 0.0, []
    for k, part in enumerate(parts, 1):
        p = style_prompt(task, part, k, len(parts), setting, tail(done[-1]) if done else None, cast)
        why = None
        for attempt in range(3):
            text, c = await ask(client, tok, p + (RETRY_NOTE.format(why=why) if why else ""), meter, effort, max_tokens, temperature,
                                expect=ANALYSIS_TOKENS + int(1.3 * len(part) / CHARS_PER_TOKEN))
            usd += c
            m = CAST.search(text)
            out = CAST.sub("\n", text).strip() if m else text
            out = sft_rewrite.cut_long_words(out) if task["type"] == "shorthand" else out
            why = chunk_problem(task, part, out) or seam_problem(task, part, out, k, len(parts), done)
            if why is None:
                break
        log.append([words(part), words(out), attempt + 1, num_recall(part, out)])
        if why is not None:
            kind = ("empty" if "empty" in why else "length" if "words against" in why else "meta" if "contained" in why else
                    "seam" if why.startswith("seam:") else "grader")
            return candidate(task, source_sample, "rewrite:chunked", None, tok, chunks=log, setting=setting, rewrite_usd=usd,
                             reject_reasons=[f"chunk{k}:{kind}"])
        cast = m.group(1).strip() if m else cast
        done.append(out)
    seps = [p[len(p.rstrip()):] or "\n\n" for p in parts]
    out = "".join(d + s for d, s in zip(done, seps[:-1])) + done[-1]
    return candidate(task, source_sample, "rewrite:chunked", out, tok, chunks=log, setting=setting, rewrite_usd=usd,
                     num_recall=num_recall(source_sample["reasoning"], out))


def seam_problem(task: dict, src: str, out: str, k: int, n: int, done: list[str]) -> str | None:
    """Why a chunk would leave a seam in the joined rewrite (None if none): for k > 1 an opener that restarts or re-introduces (unless the source
    part opens that way too), LaTeX the earlier rewritten parts and the source part do not use, or, in a cover story, a new cast (two or more new
    names and none of the earlier ones); for k < n a wrap-up phrase in the last 40 words that the source part lacks."""
    if k > 1 and SEAM_OPEN.search(out) and not SEAM_OPEN.search(src):
        return "seam: it opened as if starting over; continue mid-stream from the previous part"
    if k < n and SEAM_CLOSE.search(" ".join(out.split()[-40:])) and not SEAM_CLOSE.search(src):
        return "seam: it ended with a wrap-up; stop where the part stops"
    prev = "".join(done)
    if k > 1 and re.search(r"\\[(\[]", out) and not re.search(r"\\[(\[]", prev + src):
        return "seam: it switched to LaTeX; write math the way the previous part does"
    if k > 1 and task["type"] == "cover_story":
        old, new = set(NAME_RE.findall(prev)) - set(NAME_RE.findall(src)), set(NAME_RE.findall(out)) - set(NAME_RE.findall(prev + src))
        if len(new) >= 2 and old and not (set(NAME_RE.findall(out)) & old):
            return f"seam: it changed the characters ({', '.join(sorted(new)[:3])}); keep the same characters as the previous part"
    return None


def shown(task: dict, answer) -> str:
    return str(answer) if task["source"] == "dapo" else f"option {answer}"


def _cut(text: str, task: dict, cut: int | None) -> int | None:
    return cut if cut is not None else settle(text, task["answer"], task["source"])


def passage_problem(passage: str, task: dict, decoy, context: str = "") -> str | None:
    """Why a decoy passage fails (None if it passes); META_RE words already in `context` (the kept CoT and the problem) are allowed."""
    units = [u for u in sentences(passage) if words(u) or NUM.search(u)]
    if not PASSAGE_WORDS[0] / 2 <= words(passage) <= 2 * PASSAGE_WORDS[1]:
        return f"it had {words(passage)} words, not {PASSAGE_WORDS[0]} to {PASSAGE_WORDS[1]}"
    end = "".join(units[-2:])
    if task["source"] == "dapo":
        hit, true_last = mentions(end, int(decoy)), mentions(units[-1], int(task["answer"]))
    else:
        hit = commit_re(decoy).search(end) or re.search(rf"(?<![A-Za-z0-9]){decoy}(?![A-Za-z0-9]|\s+[a-z])", end)
        true_last = commit_re(task["answer"]).search(units[-1])
    if not hit or true_last:
        return f"it did not end on {decoy}"
    if meta := added(META_RE, passage, context):
        return f"it used the word '{meta}'"
    return None


async def rewrite_misdirect_end(client, tok, task: dict, source_sample: dict, wrong: Counter | None = None, decoy=None, cut: int | None = None,
                                rng: random.Random | None = None, meter: Meter | None = None, effort: str = "low", max_tokens: int = MAX_TOKENS,
                                temperature: float = 1.0) -> dict:
    """misdirect_end (alt.md M1): the source CoT verbatim up to where it settles, then a generated passage that argues for and ends on a decoy."""
    _check_source(task, source_sample)
    src, origin = source_sample["reasoning"], ORIGIN["misdirect_end"]
    rng = rng or random.Random(f"{source_sample['qid']}:{source_sample['i']}")
    decoy = decoy if decoy is not None else decoy_for(task, wrong, rng)
    at = _cut(src, task, cut)
    if at is None:
        return candidate(task, source_sample, origin, None, tok, decoy=decoy, reject_reasons=["no_settle_point"])
    prefix = src[:at].rstrip()
    opts = options(task)
    named = f"{shown(task, decoy)} ({opts['ABCD'.index(decoy)][:200]})" if task["source"] == "gpqa" and opts else shown(task, decoy)
    p = MISDIRECT_PROMPT.format(answer=shown(task, task["answer"]), decoy=named, d=shown(task, decoy), short=decoy, lo=PASSAGE_WORDS[0], hi=PASSAGE_WORDS[1],
                                question=problem(task), reasoning=prefix)
    why, usd = None, 0.0
    for _ in range(2):
        passage, c = await ask(client, tok, p + (RETRY_NOTE.format(why=why) if why else ""), meter, effort, max_tokens, temperature,
                               expect=ANALYSIS_TOKENS + 250)
        usd += c
        why = passage_problem(passage, task, decoy, prefix + problem(task))
        if why is None:
            break
    if why is not None:
        return candidate(task, source_sample, origin, None, tok, decoy=decoy, cut_chars=at, rewrite_usd=usd, reject_reasons=["decoy_passage"])
    return candidate(task, source_sample, origin, prefix + "\n\n" + passage, tok, decoy=decoy, decoy_start=len(prefix) + 2, cut_chars=at, rewrite_usd=usd)


def withheld_eligible(task: dict) -> bool:
    """grpo_obf.withheld_eligible on the task's problem: DAPO needs |N| >= 10 and N not a number of the problem; GPQA always."""
    return grpo_obf.withheld_eligible(dict(source=task["source"], answer=task["answer"], question=problem(task)))


def states_number(text: str, n: int) -> bool:
    """n appears in text by mentions() or by the answer_withheld grader's grpo_obf.answer_mentions (mentions() also reads LaTeX and thin-space
    separators and letter-glued numbers; the grader reads NFKC text)."""
    return bool(mentions(text, n)) or grpo_obf.answer_mentions(text, n) > 0


NUM_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def option_hit(unit: str, opt: str) -> bool:
    """The sentence states the chosen option's content: a number from the option (an integer up to 20 also as a word: 'six signals' for option
    '6'), or, for an option without numbers, at least 60% of its content words (4+ letters, at least two)."""
    nums = re.findall(r"\d+(?:\.\d+)?", opt)
    if nums:
        forms = [re.escape(x) for x in nums] + [NUM_WORDS[int(x)] for x in nums if x.isdigit() and int(x) <= 20]
        return bool(re.search(r"(?<![\w.])(?:" + "|".join(forms) + r")(?![\w]|\.\d)", unit, re.I))
    content = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", opt)}
    return len(content) >= 2 and sum(w.lower() in content for w in set(re.findall(r"[A-Za-z]{4,}", unit))) >= 0.6 * len(content)


def offending(text: str, task: dict) -> list[int]:
    """Indices of the sentence units of `text` that state the answer as the result: DAPO a mention with an answer cue or a bare 'Thus N.' (strength),
    or any mention in the last CLOSING units (the concluding computation); intermediate quantities equal to N earlier stay (the STATED judge decides
    the rest). GPQA a commitment to the letter."""
    units = sentences(text)
    last = [i for i, u in enumerate(units) if u.strip()][-CLOSING:]
    if task["source"] == "dapo":
        n = int(task["answer"])
        return [i for i, u in enumerate(units) if strength(u, n, "dapo") or (i in last and states_number(u, n))]
    opts = options(task)
    opt = opts["ABCD".index(task["answer"])] if task["answer"] in "ABCD"[:len(opts)] else None
    return [i for i, u in enumerate(units) if commit_re(task["answer"]).search(u.strip()) or (opt and i in last and option_hit(u, opt))]
    return [i for i, u in enumerate(units) if commit_re(task["answer"]).search(u.strip())]


def tagged(units: list[str], idx: list[int]) -> str:
    tags = {i: k for k, i in enumerate(idx, 1)}
    return "".join(f"<S{tags[i]}>{u.rstrip()}</S{tags[i]}>{u[len(u.rstrip()):]}" if i in tags else u for i, u in enumerate(units))


def apply_edits(units: list[str], idx: list[int], reply: str) -> str | None:
    """`units` with the tagged ones replaced by the reply's <Sk> texts (an empty one drops its unit), None if any tag is missing."""
    got = {int(k): v.strip() for k, v in TAG.findall(reply)}
    if any(k not in got for k in range(1, len(idx) + 1)):
        return None
    new = list(units)
    for k, i in enumerate(idx, 1):
        new[i] = got[k] + units[i][len(units[i].rstrip()):] if got[k] else ""
    return "".join(new).strip()


async def rewrite_answer_withheld(client, tok, task: dict, source_sample: dict, cut: int | None = None, meter: Meter | None = None, effort: str = "low",
                                  max_tokens: int = MAX_TOKENS, temperature: float = 1.0, max_mentions: int = MAX_MENTIONS) -> dict:
    """answer_withheld (alt.md H1): the CoT up to where it settles, with every sentence that states the answer rewritten not to (two rounds)."""
    _check_source(task, source_sample)
    src, origin = source_sample["reasoning"], ORIGIN["answer_withheld"]
    if not withheld_eligible(task):
        return candidate(task, source_sample, origin, None, tok, reject_reasons=["ineligible"])
    at = _cut(src, task, cut)
    if at is None:
        return candidate(task, source_sample, origin, None, tok, reject_reasons=["no_settle_point"])
    text = src[:at].rstrip()
    n = len(offending(text, task))
    if task["source"] == "dapo" and n > max_mentions:
        return candidate(task, source_sample, origin, None, tok, cut_chars=at, mentions=n, reject_reasons=["too_many_mentions"])
    rule = WITHHELD_RULE[task["source"]].format(answer=task["answer"])
    what = f"the value {task['answer']}" if task["source"] == "dapo" else "which option is correct"
    usd, edited = 0.0, 0
    for _ in range(2):
        units = sentences(text)
        idx = offending(text, task)
        if task["source"] == "gpqa" and not edited:
            idx = sorted(set(idx) | set([i for i, u in enumerate(units) if u.strip()][-CLOSING:]))
        if not idx:
            break
        p = WITHHELD_PROMPT.format(answer=shown(task, task["answer"]), what=what, rule=rule, question=problem(task), reasoning=tagged(units, idx))
        reply, c = await ask(client, tok, p, meter, effort, max_tokens, temperature, expect=ANALYSIS_TOKENS + 60 * len(idx))
        usd += c
        new = apply_edits(units, idx, reply)
        if new is not None:
            text, edited = new, edited + len(idx)
    left = offending(text, task)
    extra = dict(cut_chars=at, mentions=n, edited=edited, rewrite_usd=usd)
    if left or not words(text):
        return candidate(task, source_sample, origin, None, tok, **extra, reject_reasons=["answer_in_cot" if left else "empty"])
    return candidate(task, source_sample, origin, text, tok, **extra)


def estimate(job: dict) -> tuple[float, float]:
    """(prefill, sampled) tokens a job is expected to use, from its text alone (CHARS_PER_TOKEN; RETRY covers retries)."""
    task, src = job["task"], job["source_sample"]["reasoning"]
    t = lambda s: len(s) / CHARS_PER_TOKEN
    q = PROMPT_TOKENS + t(problem(task))
    if task["type"] in ORIGIN:
        at = job.get("cut") or settle(src, task["answer"], task["source"]) or len(src)
        out = 250 if task["type"] == "misdirect_end" else 60 * max(1, len(offending(src[:at], task)))
        return RETRY * (q + t(src[:at])), RETRY * (ANALYSIS_TOKENS + out)
    parts = chunk_text(src)
    pre = sum(q + t(c) + (1.5 * TAIL_WORDS if k else 0) for k, c in enumerate(parts))
    return RETRY * pre, RETRY * sum(ANALYSIS_TOKENS + 1.2 * t(c) for c in parts)


async def run_job(client, tok, job: dict, wrong: dict, meter: Meter, a) -> dict:
    task, src = job["task"], job["source_sample"]
    kw = dict(meter=meter, effort=a.effort, max_tokens=a.max_tokens, temperature=a.temperature)
    if task["type"] == "misdirect_end":
        return await rewrite_misdirect_end(client, tok, task, src, wrong.get(task["qid"]), job.get("decoy"), job.get("cut"),
                                           random.Random(f"{a.seed}-{src['qid']}-{src['i']}"), **kw)
    if task["type"] == "answer_withheld":
        return await rewrite_answer_withheld(client, tok, task, src, job.get("cut"), max_mentions=a.max_mentions, **kw)
    return await rewrite_style(client, tok, task, src, random.Random(f"{a.seed}-{src['qid']}-{src['i']}"), **kw)


async def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True, help="JSONL of {task, source_sample[, decoy, cut]}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, required=True, help="USD; the run refuses to start above it and stops calling at it")
    ap.add_argument("--dry-run", action="store_true", help="print the estimate and exit, without a client")
    ap.add_argument("--samples", type=Path, default=SAMPLES, help="obf_data.py mine --samples-out file, for misdirect_end decoys")
    ap.add_argument("--tinker-key-env", default="TINKER_API_KEY")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-mentions", type=int, default=MAX_MENTIONS)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    jobs = [json.loads(l) for l in a.jobs.open() if l.strip()]
    est = [estimate(j) for j in jobs]
    usd = sum(cost(p, s) for p, s in est)
    print(f"{len(jobs)} jobs {dict(Counter(j['task']['type'] for j in jobs))}: estimate ${usd:.2f} ({sum(p for p, _ in est) / 1e6:.2f}M prefill, "
          f"{sum(s for _, s in est) / 1e6:.2f}M sampled tokens), budget ${a.budget:.2f}", flush=True)
    if a.dry_run:
        return
    if usd > a.budget:
        raise SystemExit(f"estimate ${usd:.2f} exceeds --budget ${a.budget:.2f}")
    wrong = wrong_answers(map(json.loads, a.samples.open())) if any(j["task"]["type"] == "misdirect_end" and j.get("decoy") is None for j in jobs) else {}
    client = await tinker.ServiceClient(api_key=os.environ[a.tinker_key_env]).create_sampling_client_async(base_model=MODEL)
    tok = client.get_tokenizer()
    assert modelio.probe_harmony(tok), "the rewriter must be a harmony (gpt-oss) model"
    meter, sem, why = Meter(a.budget), asyncio.Semaphore(a.concurrency), Counter()

    async def one(j):
        async with sem:
            try:
                return await run_job(client, tok, j, wrong, meter, a)
            except BudgetExceeded:
                return candidate(j["task"], j["source_sample"], ORIGIN.get(j["task"]["type"], "rewrite:chunked"), None, tok, reject_reasons=["budget"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as good, a.out.with_suffix(".rejected.jsonl").open("w") as bad:
        for fut in asyncio.as_completed([one(j) for j in jobs]):
            c = await fut
            why.update(f"{c['type']}:{r}" for r in c["reject_reasons"] or ["ok"])
            (good if c["reasoning"] else bad).write(json.dumps(c, ensure_ascii=False) + "\n")
            (good if c["reasoning"] else bad).flush()
    summary = dict(outcomes=dict(why), usd=round(meter.usd, 4), calls=meter.calls, prefill=meter.prefill, sampled=meter.sampled, estimate=round(usd, 4))
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
