"""Problem pools for the obfuscation set (grpo_obf.py, obf_eval.py): DAPO-Math-17k integer-answer problems and GPQA extended, then base-model stats.

build: the DAPO problems listed in data/dapo_filtered.txt (DAPO qids, 500 problems) that contain no CJK ideograph (256; the other 244 are written
       in Chinese) and GPQA extended (546 questions; the release zip from
       github.com/idavidrein/gpqa, unpacked under data/gpqa/, gitignored). GPQA options are the correct and the three incorrect answers, shuffled
       with random.Random(f"{seed}-{Record ID}"), letters A-D. The eval split is --eval-dapo DAPO problems and --eval-gpqa GPQA questions split
       evenly over the three high-level domains, drawn with --seed; the rest is the training pool.
       Rows: qid, source (dapo | gpqa), question, options (None for dapo), answer (int for dapo, letter for gpqa), domain (gpqa).
mine:  base DeepSeek V4.1 Flash 1017 through OpenRouter raw completions (or_cotcontrol.complete, provider pinned, prompt rendered by
       modelio.render_ds at --effort, prompt token count asserted), --samples unconstrained samples per problem, T 1.0, top_p 1.0. Adds keywords
       (as mine.py: words of 4+ letters in at least half of a problem's CoTs, used at least twice, in the CoTs of at most 5% of problems, top 5 by
       count), base_words (median CoT words), n_tokens (median completion tokens), correct (fraction of samples correct) and truncated (count).

    python obf_data.py build
    python obf_data.py mine --pool data/obf_train.jsonl --samples 3
"""
import argparse
import asyncio
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import httpx
from tokenizers import Tokenizer

from cotswitch import extract, prompts
from cotswitch.data import Item

import math_data
import modelio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tinker-runner"))
from or_cotcontrol import HF_TOKENIZER, MODEL as OR_MODEL, complete  # noqa: E402

DATA = Path(__file__).parent / "data"
LETTERS = "ABCD"


def question_block(q: dict) -> str:
    """The user prompt without any requirement: the CoT-Control question block (GPQA) or math_data's (DAPO), each ending in its 'ANSWER: X' line."""
    if q["source"] == "dapo":
        return math_data.question_block(dict(problem=q["question"]))
    return prompts.question_block(Item(dataset="gpqa_ext", item_idx=q["qid"], question=q["question"], options=q["options"], answer_letter=q["answer"],
                                       keywords=[], synonyms={}, raw={}))


def parse_answer(q: dict, response: str, content: str) -> int | str | None:
    return math_data.parse_answer(response) if q["source"] == "dapo" else extract.parse_answer(response, content)


def load_gpqa(seed: int) -> list[dict]:
    csv.field_size_limit(1 << 30)
    out = []
    for r in csv.DictReader((DATA / "gpqa/dataset/gpqa_extended.csv").open(encoding="utf-8")):
        opts = [r["Correct Answer"]] + [r[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
        opts = [o.strip() for o in opts]
        order = random.Random(f"{seed}-{r['Record ID']}").sample(range(4), 4)
        out.append(dict(qid=r["Record ID"], source="gpqa", question=r["Question"].strip(), options=[opts[i] for i in order],
                        answer=LETTERS[order.index(0)], domain=r["High-level domain"]))
    return out


def build(args):
    ids = [l.strip() for l in (DATA / "dapo_filtered.txt").open() if l.strip()]
    dapo = {q["qid"]: q for q in math_data.load_dapo()}
    dapo = [dict(qid=i, source="dapo", question=dapo[i]["problem"], options=None, answer=dapo[i]["answer"], domain=None) for i in ids
            if not re.search(r"[\u3400-\u9fff]", dapo[i]["problem"])]
    gpqa = load_gpqa(args.seed)
    rng = random.Random(args.seed)
    held = set(q["qid"] for q in rng.sample(dapo, args.eval_dapo))
    for d in sorted({q["domain"] for q in gpqa}):
        held |= set(q["qid"] for q in rng.sample([q for q in gpqa if q["domain"] == d], args.eval_gpqa // 3))
    for name, rows in (("obf_train", [q for q in dapo + gpqa if q["qid"] not in held]), ("obf_eval", [q for q in dapo + gpqa if q["qid"] in held])):
        with (DATA / f"{name}.jsonl").open("w") as f:
            for q in rows:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(name, len(rows), Counter(q["source"] for q in rows), Counter(q["domain"] for q in rows if q["domain"]))


async def mine(args):
    pool = [json.loads(l) for l in args.pool.open()]
    tok = Tokenizer.from_pretrained(HF_TOKENIZER)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(client, q):
        text = modelio.render_ds(question_block(q), effort=args.effort)
        n_prompt = len(tok.encode(text, add_special_tokens=False).ids)
        body = {"model": OR_MODEL, "prompt": text, "max_tokens": args.max_tokens, "temperature": 1.0, "top_p": 1.0, "n": 1,
                "provider": {"order": [args.provider], "allow_fallbacks": False}, "usage": {"include": True}}
        async with sem:
            res = await asyncio.gather(*[complete(client, body) for _ in range(args.samples)])
        outs = []
        for b in res:
            assert b["usage"]["prompt_tokens"] == n_prompt, (b["usage"]["prompt_tokens"], n_prompt, q["qid"])
            c, finish = b["choices"][0], "length" if b["choices"][0]["finish_reason"] == "length" else "stop"
            raw = "<think>" + (c.get("reasoning") or "") + ("</think>" + (c.get("text") or "") if c.get("text") or finish == "stop" else "")  # as or_cotcontrol.py
            split = extract.split_output(raw, None, finish)
            outs.append(dict(reasoning=split["reasoning"], n_tokens=b["usage"]["completion_tokens"], cost=float(b["usage"].get("cost") or 0),
                             truncated=finish == "length", correct=parse_answer(q, split["response"], split["content"]) == q["answer"]))
        return outs

    async with httpx.AsyncClient() as client:
        samples = await asyncio.gather(*(one(client, q) for q in pool))
    counts = [[Counter(w for w in re.findall(r"[a-z]+", o["reasoning"].lower()) if len(w) >= 4) for o in outs] for outs in samples]
    df = Counter(w for cs in counts for w in set().union(*cs))
    with args.pool.open("w") as f:
        for q, outs, cs in zip(pool, samples, counts):
            total, present = sum(cs, Counter()), Counter(w for c in cs for w in c)
            kws = [w for w, k in total.most_common() if 2 * present[w] >= len(cs) and k >= 2 and df[w] <= 0.05 * len(pool)][:5]
            f.write(json.dumps(q | dict(keywords=kws, base_words=statistics.median(len(re.findall(r"[A-Za-z]+", o["reasoning"])) for o in outs),
                                        n_tokens=statistics.median(o["n_tokens"] for o in outs), correct=sum(o["correct"] for o in outs) / len(outs),
                                        truncated=sum(o["truncated"] for o in outs)), ensure_ascii=False) + "\n")
    flat = [o for outs in samples for o in outs]
    print(f"{len(pool)} problems x {args.samples}: accuracy {sum(o['correct'] for o in flat) / len(flat):.3f}, truncated {sum(o['truncated'] for o in flat)}/{len(flat)}, "
          f"median tokens {statistics.median(o['n_tokens'] for o in flat)}, mean {statistics.mean(o['n_tokens'] for o in flat):.0f}, >=1 keyword "
          f"{sum(len(k) >= 1 for k in [json.loads(l)['keywords'] for l in args.pool.open()])}/{len(pool)}, ${sum(o['cost'] for o in flat):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--eval-dapo", type=int, default=30)
    b.add_argument("--eval-gpqa", type=int, default=30)
    b.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("mine")
    m.add_argument("--pool", type=Path, required=True, help="rewritten in place with the base-model fields added")
    m.add_argument("--samples", type=int, default=3)
    m.add_argument("--effort", type=int, default=75)
    m.add_argument("--max-tokens", type=int, default=16000)
    m.add_argument("--provider", default="Parasail")
    m.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else asyncio.run(mine(a))
