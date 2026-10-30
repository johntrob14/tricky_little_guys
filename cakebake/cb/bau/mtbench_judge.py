"""MT-Bench single-answer grading (zeus, Tinker judge), ported from the FastChat fork (Rice-wxl/FastChat@b19146f6,
fastchat/llm_judge/{gen_judgment.py, common.py, show_result.py}) as run by run_mt_bench.sh (--mode single):
  - prompts verbatim from data/judge_prompts.jsonl (cb/bau/prompts/mtbench_judge_prompts.jsonl): turn 1 single-v1,
    turn 2 single-v1-multi-turn; categories math/reasoning/coding (NEED_REF_CATS) use single-math-v1 /
    single-math-v1-multi-turn with the reference answers (reference_answer/gpt-4.jsonl; the fork's gpt-4o.jsonl is a
    symlink to it). Messages = [system: prompt system_prompt, user: prompt_template.format(...)] (FastChat's
    chatgpt conversation -> OpenAI messages).
  - rating parsed from "[[x]]", else "[x]"; no match -> -1 (dropped); no re-ask (FastChat does not re-ask).
  - S = mean over all judgments with score != -1, both turns pooled (show_result.py "Average"); CI from per-question
    means of the two turns (multi_objective_mo scores.read_mtbench_std).
Deviation: judge = Tinker model (default kimi-k2.6-nothink, temperature 0, max_tokens 2048) instead of gpt-4o
(temperature 0, max_tokens 2048). Raw S is therefore NOT comparable to Table 25's 5.34; use S / S_base with the base
re-judged by the same judge.

  uv run --project ~/code/tricky_little_guys/self-oversight python -m cb.bau.mtbench_judge --name base_dpo
  ... --name <x> --qids 81,82,111,112 --budget 0.05        # test on a few questions
Output: results/bau/<NAME>/mtbench_judgments.<judge>.jsonl (append-only, keyed qid-turn) and mtbench.<judge>.json.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import statistics as st

from cb.bau.common import DATA, append_jsonl, judge_call, judge_runner, out_dir, prompt, read_jsonl, spend

NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]
ONE = re.compile(r"\[\[(\d+\.?\d*)\]\]")
ONE_BACKUP = re.compile(r"\[(\d+\.?\d*)\]")
PROMPTS = {json.loads(l)["name"]: json.loads(l) for l in prompt("mtbench_judge_prompts.jsonl").splitlines() if l.strip()}


def build(q, ans, ref, turn):
    """(prompt name, messages) exactly as FastChat run_judge_single."""
    multi = turn == 2
    ref_based = q["category"] in NEED_REF_CATS
    name = ("single-math-v1" if ref_based else "single-v1") + ("-multi-turn" if multi else "")
    tmpl = PROMPTS[name]
    kw = {}
    if ref_based:
        kw["ref_answer_1"] = ref["choices"][0]["turns"][0]
        if multi:
            kw["ref_answer_2"] = ref["choices"][0]["turns"][1]
    a = ans["choices"][0]["turns"]
    if multi:
        user = tmpl["prompt_template"].format(question_1=q["turns"][0], question_2=q["turns"][1], answer_1=a[0],
                                              answer_2=a[1], **kw)
    else:
        user = tmpl["prompt_template"].format(question=q["turns"][0], answer=a[0], **kw)
    return name, [{"role": "system", "content": tmpl["system_prompt"]}, {"role": "user", "content": user}]


def parse_rating(text: str) -> float:
    m = re.search(ONE, text) or re.search(ONE_BACKUP, text)
    return float(ast.literal_eval(m.groups()[0])) if m else -1.0


def summarize(rows):
    best = {}
    for d in rows:
        best[(d["question_id"], d["turn"])] = d  # latest row wins
    ok = [d for d in best.values() if d["score"] != -1]
    if not ok:
        return None
    by_q = {}
    for d in ok:
        by_q.setdefault(d["question_id"], []).append(d["score"])
    qm = [sum(v) / len(v) for v in by_q.values()]
    t = lambda k: [d["score"] for d in ok if d["turn"] == k]  # noqa: E731
    return dict(mean=sum(d["score"] for d in ok) / len(ok), n=len(ok), n_failed=len(best) - len(ok),
                turn1=st.mean(t(1)) if t(1) else None, turn2=st.mean(t(2)) if t(2) else None,
                n_questions=len(qm), stderr=st.stdev(qm) / len(qm) ** 0.5 if len(qm) > 1 else None,
                cost_usd=round(sum(d.get("cost", 0) for d in rows), 5))


async def run(a):
    qs = [json.loads(l) for l in open(os.path.join(DATA, "mt_bench", "question.jsonl")) if l.strip()]
    if a.qids:
        keep = {int(x) for x in a.qids.split(",")}
        qs = [q for q in qs if q["question_id"] in keep]
    refs = {r["question_id"]: r for r in read_jsonl(os.path.join(DATA, "mt_bench", "reference_answer", "gpt-4.jsonl"))}
    od = out_dir(a.name)
    answers = {r["question_id"]: r for r in read_jsonl(a.answers or os.path.join(od, "mtbench_answers.jsonl"))}
    dst = os.path.join(od, f"mtbench_judgments.{a.judge}.jsonl")
    done = {r["key"] for r in read_jsonl(dst)}
    r = judge_runner(a.judge, a.budget, a.temperature)
    sem = asyncio.Semaphore(a.concurrency)
    todo = [(q, t) for q in qs for t in (1, 2) if f"{q['question_id']}-{t}" not in done]
    missing = [q["question_id"] for q in qs if q["question_id"] not in answers]
    assert not missing, f"no answers for {missing}"

    async def one(q, turn):
        name, msgs = build(q, answers[q["question_id"]], refs.get(q["question_id"]), turn)
        async with sem:
            res = await judge_call(r, msgs, a.max_tokens)
        append_jsonl(dst, dict(key=f"{q['question_id']}-{turn}", question_id=q["question_id"], turn=turn,
                               category=q["category"], model=a.name, judge=a.judge, prompt_name=name,
                               score=parse_rating(res["answer"]), judgment=res["answer"], user_prompt=msgs[1]["content"],
                               thinking=res["thinking"], stop=res["stop"], cost=res["cost"],
                               prompt_tokens=res["prompt_tokens"], completion_tokens=res["completion_tokens"]))

    print(f"[mtbench_judge {a.name}] {len(todo)} judgments to run ({len(done)} done) with {a.judge}", flush=True)
    await asyncio.gather(*[one(q, t) for q, t in todo])
    rows = read_jsonl(dst)
    s = summarize(rows)
    s.update(judge=a.judge, temperature=a.temperature, max_tokens=a.max_tokens, qids=a.qids)
    json.dump(s, open(os.path.join(od, f"mtbench.{a.judge}.json"), "w"), indent=1)
    print(json.dumps(s), "|", spend(r), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--judge", default="kimi-k2.6-nothink")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--qids", default=None, help="comma-separated question ids (tests)")
    ap.add_argument("--answers", default=None, help="default results/bau/<NAME>/mtbench_answers.jsonl")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
