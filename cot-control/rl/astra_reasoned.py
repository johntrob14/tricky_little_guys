"""Astra strict passes whose CoT is an attempt at the question ("reasoned passes"), for every run in astra_bylength.py.

A short CoT can pass a style rule without solving anything ("AlTeRnAtInG CaSe TeStInG.", "I will think without forbidden terms.",
"ABCD"). Each strict pass (meta allowed) on the eight constraints other than ignore_question goes to grpo.py's reasoning judge
(JUDGE_PROMPT, the grpo2 judge: "Is the reasoning an attempt at this question ...?", style ignored, CoT destyled as in grpo.py)
on gpt-oss-120b through OpenRouter, provider DeepInfra, effort low, temperature 0. The question shown is the user prompt up to the
answer-format line, so the judge never sees the requirement. ignore_question passes are reported in their own column: a pass there
means the CoT is not about the question, so the judge's criterion does not apply. An unparsed verdict counts as not reasoned.

Length-only prediction of reasoned passes: thinkvec/lenadj.py's logistic, refit with reasoned pass as the outcome on the same
pooled untrained plain-prompt rows as astra_bylength.py, ignore_question rows excluded (96 rows per run).

Verdicts are cached in <out>/verdicts.jsonl, so a rerun only judges new passes.

    .venv/bin/python astra_reasoned.py --out ../results/astra_bylength
"""

import argparse
import asyncio
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import pandas as pd

from astra_bylength import BINS, EDGES, POOL, RUNS
from grpo import JUDGE_PROMPT
from orjudge import OpenRouterJudge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thinkvec"))
from lenadj import add_length_adjustment  # noqa: E402

csv.field_size_limit(10**9)
KEY = ("dataset", "item_idx", "constraint", "sample_idx")


def destyle(c: str, text: str, ga: dict) -> str:
    if c in ("lowercase_thinking", "uppercase_thinking", "alternating_case"):
        return text.lower()
    if c == "meow_between_words":
        w = ga.get("target_word", "meow")
        return re.sub(r"[ \t]+", " ", re.sub(rf"(?i){re.escape(w)}", " ", text)).strip()
    if c == "repeat_sentences" and ga.get("target_sentence"):
        return re.sub(re.escape(ga["target_sentence"]), " ", text, flags=re.I).strip()
    return text


def load(i, g, m, s, run: Path):
    strict = {tuple(r[k] for k in KEY): r for r in csv.DictReader(open(run / "analysis/records.csv")) if r["protocol"].startswith("astra")}
    out = []
    for line in open(next(run.glob("generations/*/astra*.jsonl"))):
        x = json.loads(line)
        k = (x["dataset"], str(x["item_idx"]), x["constraint"], str(x["sample_idx"]))
        if k not in strict:
            continue
        cot = x["reasoning"] or ""
        out.append(dict(run_i=i, run=run.name, group=g, model=m, setting=s, key="|".join(k), constraint=x["constraint"],
                        strict=strict[k]["strict"] == "1", correct=strict[k]["correct"] == "True", cot_words=len(cot.split()),
                        question=x["user_prompt"].split("\n\n\nYour response should")[0].strip(),
                        cot=destyle(x["constraint"], cot, x.get("grader_args") or {}),
                        condition="pool" if run.name in POOL else "other"))
    return out


async def judge_all(todo, judge, cache_path):
    done = 0

    async def one(r):
        nonlocal done
        final, _, _ = await judge.ask(JUDGE_PROMPT.format(question=r["question"], reasoning=r["cot"]))
        m = re.match(r"\W*(?:ANSWER:)?\W*(YES|NO)\b", final.upper())
        v = dict(run=r["run"], key=r["key"], verdict=(m.group(1) == "YES") if m else None, final=final[:200])
        with open(cache_path, "a") as f:
            f.write(json.dumps(v) + "\n")
        done += 1
        if done % 100 == 0:
            print(f"judged {done}/{len(todo)}  ${judge.usd:.2f}", file=sys.stderr)
        return v

    return await asyncio.gather(*[one(r) for r in todo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provider", default="DeepInfra")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    df = pd.DataFrame([r for i, spec in enumerate(RUNS) for r in load(i, *spec)])
    cache_path = args.out / "verdicts.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            v = json.loads(line)
            cache[v["run"], v["key"]] = v["verdict"]
    need = df[df.strict & (df.constraint != "ignore_question")]
    todo = [r for r in need.to_dict("records") if (r["run"], r["key"]) not in cache]
    print(f"{len(need)} passes to judge, {len(todo)} not cached", file=sys.stderr)
    if todo:
        judge = OpenRouterJudge("openai/gpt-oss-120b", effort="low", concurrency=args.concurrency, provider=args.provider)
        for v in asyncio.run(judge_all(todo, judge, cache_path)):
            cache[v["run"], v["key"]] = v["verdict"]
        print(f"judge cost ${judge.usd:.2f}, providers {dict(judge.providers)}, timeouts {judge.timeouts}", file=sys.stderr)

    df["verdict"] = [cache.get((r, k)) if s and c != "ignore_question" else None for r, k, s, c in zip(df.run, df.key, df.strict, df.constraint)]
    df["reasoned"] = df.strict & (df.constraint != "ignore_question") & (df.verdict == True)  # noqa: E712
    nq = df[df.constraint != "ignore_question"].copy()
    nq["strict_orig"], nq["strict"] = nq.strict, nq.reasoned
    odds = add_length_adjustment(nq, "pool")
    nq["bin"] = pd.cut(nq.cot_words, EDGES, labels=BINS, right=False)
    df.drop(columns=["question", "cot"]).to_csv(args.out / "reasoned_rows.csv", index=False)

    md = [f"# Astra reasoned passes: strict pass (meta allowed) and the CoT is an attempt at the question\n\nGenerated by `rl/astra_reasoned.py` "
          f"(method in its docstring). Judge: gpt-oss-120b, grpo.py JUDGE_PROMPT. Eight constraints (96 rows per run); ignore_question "
          f"passes in their own column (of 12). Length-only logistic for reasoned passes, fit on the pooled untrained plain-prompt rows: "
          f"odds x{odds:.2f} per doubling of CoT words. Bin cells: reasoned passes / rows (eight constraints) in that CoT-word bin.\n"]
    for g, gq in nq.groupby("group", sort=False):
        md += [f"\n## {g}\n", "| model | setting | run | strict passes (8 constraints) | reasoned passes | not reasoned | unparsed | "
               "predicted from length | obs - pred | ignore_question passes | median CoT words | accuracy (all 108) | "
               + " | ".join(f"CoT {b}" for b in BINS) + " |", "|" + "---|" * (12 + len(BINS))]
        for i, r in gq.groupby("run_i", sort=True):
            x, full = r.iloc[0], df[df.run_i == i]
            s, rs = int(r.strict_orig.sum()), int(r.reasoned.sum())
            unp = int((r.strict_orig & r.verdict.isna()).sum())
            iq = full[full.constraint == "ignore_question"]
            cells = [f"{int(r[r.bin == b].reasoned.sum())}/{int((r.bin == b).sum())}" if (r.bin == b).any() else "-" for b in BINS]
            md.append(f"| {x.model} | {x.setting} | `{x.run}` | {s} | {rs} | {s - rs - unp} | {unp} | {r.p_len.sum():.1f} | "
                      f"{rs - r.p_len.sum():+.1f} | {int(iq.strict.sum())} | {st.median(full.cot_words):.0f} | {int(full.correct.sum())} | "
                      + " | ".join(cells) + " |")
    (args.out / "reasoned_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
