"""WMDP sandbagging samples (the sandbag.py arms) from an API model through OpenRouter. Writes the same rows.jsonl as sandbag.py, so sandbag_judge.py scores them.

Same system prompt, questions and four arms as sandbag.py. One chat completion per question and arm: temperature 0, `reasoning: {effort}` (the highest level the
model takes), one provider pinned with no fallbacks (OpenRouter spreads a model over hosts with different quantizations), max_tokens as a safety cap only.
Temperature --temperature (0 by default; GLM-5.2 loops at 0 on some questions). CoT = the returned `reasoning` field (raw trace; check the provider returns one, not a summary), answer = letter parsed from the content. cot_tokens is the
provider's reasoning_tokens count, resp_tokens the rest of the completion. `cost` is OpenRouter's reported cost per call.

    python sandbag_or.py --name glm-5.1 --model z-ai/glm-5.1 --provider Z.AI --out ../results/sandbag
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import httpx

from hintverb import SYSTEM, parse_letter
from orjudge import load_key
from sandbag import ARMS, user_message

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


async def sample(client: httpx.AsyncClient, body: dict) -> dict:
    err = ""
    for attempt in range(6):
        try:
            r = await client.post(OR_URL, json=body, timeout=1800)
        except httpx.HTTPError as e:
            r, err = None, repr(e)
        if r is not None and r.status_code == 200:
            try:
                b = r.json()
            except ValueError:  # a 200 whose body is not JSON (keep-alive comment lines, then the connection dropped)
                b = {"error": r.text[:300]}
            if "choices" in b and b["choices"][0].get("finish_reason") != "error" and b.get("usage"):
                return b
            err = json.dumps(b)[:300]
        elif r is not None:
            err = f"HTTP {r.status_code} {r.text[:300]}"
        print(f"retry {attempt}: {err[:200]}", file=sys.stderr, flush=True)
        await asyncio.sleep(2 * 2 ** attempt)
    raise RuntimeError(f"gave up: {err}")


def reasoning_text(msg: dict) -> str:
    if msg.get("reasoning"):
        return msg["reasoning"]
    return "".join(d.get("text") or d.get("summary") or "" for d in msg.get("reasoning_details") or [])


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="OpenRouter model id")
    ap.add_argument("--provider", default="Z.AI", help="provider pinned with no fallbacks; empty string for OpenRouter's default routing")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--questions", type=Path, default=Path(__file__).parent / "data/wmdp_sandbagging.jsonl")
    ap.add_argument("--n", type=int, help="only the first N questions (smoke tests)")
    ap.add_argument("--n-random", type=int, help="a random subset of N questions (seed --seed), all arms each")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    qs = list(map(json.loads, open(args.questions)))[:args.n]
    if args.n_random:
        qs = sorted(random.Random(args.seed).sample(qs, args.n_random), key=lambda q: q["qid"])
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), questions=str(args.questions), out=str(args.out), system=SYSTEM, n_questions=len(qs)), open(out / "config.json", "w"), indent=1)
    path = out / "rows.jsonl"
    done = {(r["qid"], r["arm"]) for r in map(json.loads, open(path))} if path.exists() else set()
    client = httpx.AsyncClient(headers={"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
    sem, lock, tally = asyncio.Semaphore(args.concurrency), asyncio.Lock(), dict(rows=0, cost=0.0, cot=0, comp=0)

    async def one(q, arm):
        if (q["qid"], arm) in done:
            return
        user = user_message(q, arm)
        body = {"model": args.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "temperature": args.temperature, "max_tokens": args.max_tokens,
                "usage": {"include": True}, "transforms": [], "reasoning": {"effort": args.effort}}
        if args.provider:
            body["provider"] = {"order": [args.provider], "allow_fallbacks": False}
        async with sem:
            b = await sample(client, body)
        ch, usage = b["choices"][0], b["usage"]
        msg = ch["message"]
        reasoning, content = reasoning_text(msg), msg.get("content") or ""
        cot = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        row = dict(qid=q["qid"], arm=arm, correct=q["correct"], hint_wrong=q["hint_wrong"], answer=parse_letter(content), finish="length" if ch.get("finish_reason") == "length" or (usage.get("completion_tokens") or 0) >= args.max_tokens - 16 else "stop",  # Z.AI reports "stop" at the cap
                   cot_tokens=cot, resp_tokens=(usage.get("completion_tokens") or 0) - cot, reasoning=reasoning, response=content, user=user,
                   provider=b.get("provider"), finish_reason=ch.get("finish_reason"), cost=usage.get("cost"), completion_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"))
        async with lock:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            tally["rows"] += 1
            tally["cost"] += usage.get("cost") or 0
            tally["cot"] += cot
            tally["comp"] += usage.get("completion_tokens") or 0
            if tally["rows"] % 50 == 0:
                print(f"{tally['rows']} done, completion tokens {tally['comp']} (reasoning {tally['cot']}), reported cost ${tally['cost']:.2f}", flush=True)

    await asyncio.gather(*[one(q, a) for q in qs for a in args.arms])
    print(f"{len(qs)} questions x {len(args.arms)} arms; {tally['rows']} new rows, completion tokens {tally['comp']} (reasoning {tally['cot']}), reported cost ${tally['cost']:.2f} -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
