"""no_think_test.py through OpenRouter chat completions: can a hosted reasoning model answer immediately, with empty thinking, when
told to? Same 100 held-out MMLU-Pro history questions (length_eval.py), conditions control and nothink, one reasoning setting per run.

The instruction says "thinking" where no_think_test.py says "analysis channel" (gpt-oss's name for its CoT). CoT length is the
provider's usage.completion_tokens_details.reasoning_tokens (and the returned reasoning text). Provider pinned, no fallbacks.

    python no_think_or.py --model z-ai/glm-5.3 --provider Z.AI --reasoning '{"effort": "max"}' --name glm53-max --out ../results/no_think
    python no_think_or.py --model z-ai/glm-5.1 --provider Z.AI --reasoning '{"enabled": true}' --name glm51 --out ../results/no_think
"""

import argparse
import asyncio
import json
import os
import statistics as st
import sys
from pathlib import Path

import httpx

from cotswitch import extract, prompts

from length_eval import heldout_items

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
INSTRUCTION = ("Answer immediately, without thinking. Do not reason before answering: leave your thinking empty and "
               "give only the final answer in the required format.")


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--reasoning", required=True, help="JSON for OpenRouter's reasoning parameter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    items = [it for it in heldout_items() if it.raw["category"] == "history"][:args.n]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows_or.jsonl"
    done = {(r["model"], r["qid"], r["condition"]) for r in map(json.loads, open(path))} if path.exists() else set()
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    reasoning = json.loads(args.reasoning)

    async with httpx.AsyncClient() as client:
        async def one(it, cond):
            if (args.name, it.item_idx, cond) in done:
                return
            user = prompts.question_block(it) + ("\n\n" + INSTRUCTION if cond == "nothink" else "")
            body = {"model": args.model, "messages": [{"role": "user", "content": user}], "max_tokens": args.max_tokens,
                    "reasoning": reasoning, "usage": {"include": True}, "provider": {"order": [args.provider], "allow_fallbacks": False}}
            b, err = None, ""
            async with sem:
                for attempt in range(6):
                    try:
                        r = await client.post(OR_URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=1800)
                        b = r.json()
                        if r.status_code == 200 and "choices" in b and b["choices"][0].get("finish_reason") != "error":
                            break
                        err, b = f"HTTP {r.status_code} {str(b)[:200]}", None
                    except (httpx.HTTPError, ValueError) as e:
                        err, b = repr(e), None
                    print(f"retry {attempt} {it.item_idx} {cond}: {err}", file=sys.stderr)
                    await asyncio.sleep(10 * (attempt + 1))
            if b is None:
                return
            ch, usage = b["choices"][0], b.get("usage") or {}
            msg = ch["message"]
            reasoning_text, content = msg.get("reasoning") or "", msg.get("content") or ""
            row = dict(model=args.name, or_model=args.model, provider=b.get("provider"), reasoning_param=reasoning, qid=it.item_idx,
                       condition=cond, finish=ch.get("finish_reason"),
                       reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       completion_tokens=usage.get("completion_tokens"), reasoning_words=len(reasoning_text.split()),
                       correct=extract.parse_answer(content, content) == it.answer_letter, cost=usage.get("cost"),
                       reasoning=reasoning_text, response=content)
            async with lock:
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

        await asyncio.gather(*(one(it, c) for it in items for c in ("control", "nothink")))

    rows = [json.loads(l) for l in open(path) if json.loads(l)["model"] == args.name]
    print(f"{args.name}: cost ${sum(r['cost'] or 0 for r in rows):.2f}, providers {sorted({r['provider'] for r in rows})}")
    print("| model | condition | n | median reasoning tokens | reasoning <= 5 tokens | reasoning <= 20 tokens | accuracy |")
    print("|---|---|---|---|---|---|---|")
    for c in ("control", "nothink"):
        rs = [r for r in rows if r["condition"] == c]
        t = [r["reasoning_tokens"] or 0 for r in rs]
        print(f"| {args.name} | {c} | {len(rs)} | {st.median(t):.0f} | {sum(x <= 5 for x in t) / len(rs):.0%} | "
              f"{sum(x <= 20 for x in t) / len(rs):.0%} | {sum(r['correct'] for r in rs) / len(rs):.2f} |")


if __name__ == "__main__":
    asyncio.run(main())
