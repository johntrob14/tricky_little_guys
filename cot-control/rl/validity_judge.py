"""Reasoning-validity check for gpt-oss SFT candidates: does the CoT's own work arrive at the final answer?

The spot check of full1 (2026-10-24) found accepted candidates whose final answer is right but whose CoT does not support it: a 'closest option'
pick after a calculation that matches no option, a bluffed or asserted decisive step, invented numbers or facts, or a CoT that calls the question
ambiguous and then answers. The requirement judge grades only style and the forced-answer check reads the answer the CoT names, so neither
catches these. VALIDITY_JUDGE (DeepSeek V4.1 Flash, as the other judges) sees the question, the CoT (for misdirect_end only the part before the
decoy passage) and the final answer, and answers SUPPORTED: YES/NO. Rows get valid (True/False/None if unparsed) and valid_final.

    python validity_judge.py --run full1 --budget 2 [--seam]

With --seam, accepted multi-chunk rewrites also get SEAM_JUDGE (does the joined rewrite read as one continuous piece?): seamless, seamless_final.

Writes the fields into every accepted, non-plain row of results/sft_gptoss/<run>/candidates_*.jsonl.
"""
import argparse
import asyncio

import grpo_obf
import gptoss_sft_data as gs
from orjudge import OpenRouterJudge

VALIDITY_JUDGE = """Below is a question, the private reasoning a model wrote while solving it, and the final answer it gave. The reasoning may be written in an unusual style (a story, pirate speech, all lower case, questions, emotional remarks); ignore the style.

<question>
{question}
</question>

<reasoning>
{reasoning}
</reasoning>

<final_answer>
{answer}
</final_answer>

Does the reasoning's own work actually arrive at this final answer? Answer NO if any of these hold:
- the work arrives at a different value or option and the answer is then picked as the closest or most plausible option, or justified with "rounding" or "the problem probably intended";
- the decisive step (the one the answer depends on) is asserted without support, bluffed (an appeal to an unnamed theorem or known result), or rests on a claim that is plainly false;
- numbers, facts or results are invented to make the answer come out;
- the reasoning says it is unsure, the question is ambiguous, or it is guessing, and the answer is not settled by the work;
- the reasoning never engages with the question's content and the answer appears without derivation.
Answer YES if the work, read as a whole, derives the answer, even if it is messy, takes detours, or contains small slips that do not affect the answer.

Reply with SUPPORTED: YES or SUPPORTED: NO on the first line, then one sentence explaining it."""

SEAM_JUDGE = """Below is a model's private reasoning about a problem. It was produced in several parts that were joined together. Does it read as one continuous piece of reasoning written in one go?

<reasoning>
{reasoning}
</reasoning>

Answer NO if, partway through, it restarts or re-introduces the problem, opens a new scene that redoes earlier work, contains a wrap-up or closing passage before the end, repeats a passage, changes its characters, names or setting, or switches how it writes math or its voice. Answer YES otherwise.

Reply with SEAMLESS: YES or SEAMLESS: NO on the first line, then one sentence explaining it."""


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--judge-provider", default="DeepInfra")
    ap.add_argument("--seam", action="store_true", help="also run SEAM_JUDGE on accepted multi-chunk rewrites (field seamless)")
    a = ap.parse_args()
    d = gs.run_dir(a.run)
    problems = gs.problems_by_qid(d)
    jd = OpenRouterJudge(grpo_obf.JUDGE_MODEL, max_tokens=8000, concurrency=128, provider=a.judge_provider)
    files = {f: gs.read(f) for f in gs.candidate_files(d)}
    todo = [r for rows in files.values() for r in rows if r.get("accepted") and r["type"] != "plain" and "valid" not in r]
    print(f"{len(todo)} candidates")

    async def one(r):
        t = gs.judge_task(r, problems[r["qid"]])
        cot = r["reasoning"][:r["decoy_start"]] if r["type"] == "misdirect_end" and r.get("decoy_start") else r["reasoning"]
        final, *_ = await jd.ask(VALIDITY_JUDGE.format(question=t["qblock"], reasoning=cot, answer=r["true_answer"]))
        r["valid"], r["valid_final"] = grpo_obf.flag("SUPPORTED", final), final

    async def seam(r):
        final, *_ = await jd.ask(SEAM_JUDGE.format(reasoning=r["reasoning"]))
        r["seamless"], r["seamless_final"] = grpo_obf.flag("SEAMLESS", final), final

    multi = [r for rows in files.values() for r in rows if a.seam and r.get("accepted") and len(r.get("chunks") or []) > 1 and "seamless" not in r]
    if multi:
        await asyncio.gather(*(seam(r) for r in multi))
        import collections
        print("seam judge:", dict(collections.Counter((r["type"], r["seamless"]) for r in multi)), f"${jd.usd:.2f}")
    for k in range(0, len(todo), 200):
        await asyncio.gather(*(one(r) for r in todo[k:k + 200]))
        print(f"  {min(k + 200, len(todo))}/{len(todo)}, ${jd.usd:.2f}", flush=True)
        if jd.usd > a.budget:
            print("budget reached")
            break
    for f, rows in files.items():
        gs.write(f, rows)
    import collections
    print(dict(collections.Counter((r["origin"], r["type"], r.get("valid")) for r in todo)))
    print(f"judge ${jd.usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
