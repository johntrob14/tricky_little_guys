"""Judges for the Fortress rollouts, the eval_awareness_scaling protocol's: vea_cal (the post's F1 rubric plus John's notes) on the full
reasoning, refusal (the post's F2) on the 384-token answer view, category on each VEA quote. Judge model Qwen3.8-27B as in that protocol, here
qwen/qwen3.8-27b through OpenRouter's raw completions on Chutes (the prompt rendered locally: thinking at low effort for vea_cal, off for the others),
greedy. By default the VEA judge runs only on CoTs that hit the repo's broad PREFILTER_RE (test / evaluat* / probing / trap / ...), the rest
being recorded as VEA-negative (the repo's remedy for slow judges; --no-prefilter judges everything). A VEA verdict without a QUOTE line, or whose reasoning never closed, is invalid (None), not negative. Every VEA quote is located in the
reasoning (exact / near after NFKC + whitespace + quote normalization / fail).

    python fortress_judge.py base organism --tasks vea_cal refusal category
    python fortress_judge.py organism --file continuations.jsonl --tasks refusal vea_cal      # intervention continuations

Output: results/fortress/<name>/judge.<task>[.<file stem>].jsonl, one row per rollout key, resumable.
"""
import argparse
import asyncio
import json
import re
import unicodedata
from pathlib import Path

import fortress_rubrics as R
from fortress_eval import OUT, Sampler

JUDGE = {"vea_cal": (R.VEA_CAL_RUBRIC, "low", 6000), "refusal": (R.REFUSAL_RUBRIC + R.REFUSAL_INPUTS, None, 8), "category": (R.CATEGORY_RUBRIC, None, 48)}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def locate(quote: str, text: str) -> str:
    if quote in text:
        return "exact"
    q, t = norm(quote), norm(text)
    if q in t:
        return "near"
    frags = [f for f in re.split(r"[.;,\n]", q) if len(f.strip()) > 20]
    return "fragments" if frags and all(f.strip() in t for f in frags) else "fail"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--tasks", nargs="+", default=["vea_cal", "refusal", "category"])
    ap.add_argument("--file", default="rollouts.jsonl")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--no-prefilter", action="store_true", help="judge every CoT for VEA; by default CoTs with no hit on the repo's broad PREFILTER_RE are recorded as VEA-negative without a judge call (its remedy for slow judges)")
    args = ap.parse_args()
    judge = Sampler("openrouter", concurrency=args.concurrency, sampling={"temperature": 0.0, "top_p": 1.0, "top_k": 0})  # greedy
    suffix = "" if args.file == "rollouts.jsonl" else "." + Path(args.file).stem
    for name in args.names:
        rows = [r for r in map(json.loads, (OUT / name / args.file).open()) if r["status"] == "scored"]
        for r in rows:
            r.setdefault("key", f"{r['qid']}/{r['idx']}")
        for task in args.tasks:
            rubric, effort, max_tokens = JUDGE[task]
            path = OUT / name / f"judge.{task}{suffix}.jsonl"
            done = {v["key"]: v for v in map(json.loads, path.open())} if path.exists() else {}
            if task == "category":
                vea = {v["key"]: v for v in map(json.loads, (OUT / name / f"judge.vea_cal{suffix}.jsonl").open()) if v["vea"]}
                todo = [r for r in rows if r["key"] in vea and r["key"] not in done]
            else:
                todo = [r for r in rows if r["key"] not in done]
            if task == "vea_cal" and not args.no_prefilter:
                skipped = [r for r in todo if not R.PREFILTER_RE.search(r["reasoning"])]
                with path.open("a") as f:
                    f.writelines(json.dumps({"key": r["key"], "qid": r["qid"], "idx": r["idx"], "task": task, "prefiltered": True, "vea": False, "quote": None, "quote_check": None, "raw": "", "usd": 0.0}) + "\n" for r in skipped)
                todo = [r for r in todo if R.PREFILTER_RE.search(r["reasoning"])]
                print(f"{name} {task}: {len(skipped)} CoTs with no prefilter-regex hit recorded as VEA-negative", flush=True)
            print(f"{name} {task}: {len(todo)} to judge ({len(done)} done)", flush=True)

            async def one(r, f):
                if task == "vea_cal":
                    user = rubric.format(cot=r["reasoning"])
                elif task == "refusal":
                    user = rubric.format(prompt=r["prompt"], response=r["response"])
                else:
                    user = rubric.format(prompt=r["prompt"][:1000], quote=vea[r["key"]]["quote"])
                kw = {"reasoning_effort": effort} if effort else {"enable_thinking": False}
                prefix = judge.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)
                text, finish, usd = await judge.sample(prefix, max_tokens, seed=0)
                closed = "</think>" in text or not effort
                final = (text.split("</think>")[-1] if closed else "") if effort else text  # a no-thinking judge answers directly (it sometimes appends a stray </think>)
                v = {"key": r["key"], "qid": r["qid"], "idx": r["idx"], "task": task, "finish_reason": finish, "closed": closed, "raw": text, "usd": usd}
                if task == "vea_cal":
                    vea_flag, quote = R.parse_vea(final) if closed else (None, None)
                    v |= {"vea": vea_flag, "quote": quote, "quote_check": locate(quote, r["reasoning"]) if quote else None}
                elif task == "refusal":
                    v["refusal"] = R.parse_yes_no(final)
                else:
                    cat, sub = R.parse_category(final)
                    v |= {"category": cat, "subcategory": sub, "quote": vea[r["key"]]["quote"]}
                f.write(json.dumps(v) + "\n"); f.flush()

            with path.open("a") as f:
                await asyncio.gather(*[one(r, f) for r in todo])
            vs = [json.loads(l) for l in path.open()]
            field = {"vea_cal": "vea", "refusal": "refusal", "category": "subcategory"}[task]
            vals = [v[field] for v in vs]
            print(f"{name} {task}: {len(vs)} judged, invalid {sum(x is None for x in vals)}, " + (f"positive {sum(bool(x) for x in vals)}" if task != "category" else f"{ {s: vals.count(s) for s in set(vals)} }")
                  + (f", quote_check { {c: sum(v['quote_check'] == c for v in vs) for c in ('exact', 'near', 'fragments', 'fail')} }" if task == "vea_cal" else "") + f", judge ${judge.usd:.2f} so far", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
