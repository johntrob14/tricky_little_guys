"""WMDP sandbagging, Li et al. 2025 protocol (the sandbag_li.py conditions), from an API model through OpenRouter. Writes the same rows.jsonl as sandbag_li.py,
so sandbag_li_judge.py scores them. Request shape, retries and the reasoning field as in sandbag_or.py: one chat completion per question and condition, the
condition's system prompt as the system message, temperature --temperature, `reasoning: {effort}`, one provider pinned with no fallbacks unless --provider is
empty. CoT = the returned `reasoning` field, cot_tokens = the provider's reasoning_tokens count.
With --ds-effort (DeepSeek V4.1 Flash 1017): a raw /completions request instead, the prompt rendered by modelio.render_ds exactly as sandbag_li.py renders it
for DeepSeek on Tinker (thinking mode at that effort, the system prompt in the <｜System｜> block), its token count asserted against the HF tokenizer; the
provider must pass the prompt through untemplated (Parasail, Together, Crusoe). CoT = the returned `reasoning`, cot_tokens counted with the HF tokenizer.

    python sandbag_li_or.py --name base-or --model openai/gpt-oss-120b --provider "" --effort medium --out ../results/sandbag_li
    python sandbag_li_or.py --name ds-base-or --model deepseek/deepseek-v4.1-flash-1017 --provider Parasail --ds-effort 75 --temperature 1 --max-tokens 25000 --out ../results/sandbag_li
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from tokenizers import Tokenizer

import modelio
from orjudge import load_key
from sandbag_li import CONDITIONS, SYSTEM, mcq, parse_answer
from sandbag_or import reasoning_text, sample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tinker-runner"))
from or_cotcontrol import HF_TOKENIZER, complete  # noqa: E402


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="OpenRouter model id")
    ap.add_argument("--provider", default="", help="provider pinned with no fallbacks; empty for OpenRouter's default routing")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--ds-effort", type=int, help="DeepSeek: raw completions with the render_ds prompt at this effort (1-100) instead of a chat request")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--questions", type=Path, default=Path(__file__).parent / "data/li_questions.jsonl")
    ap.add_argument("--n", type=int, help="only the first N questions of each dataset (smoke tests)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    qs = [q for q in map(json.loads, open(args.questions)) if args.n is None or int(q["qid"].split("-")[-1]) < args.n]
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), questions=str(args.questions), out=str(args.out), system=SYSTEM, n_questions=len(qs)), open(out / "config.json", "w"), indent=1)
    path = out / "rows.jsonl"
    done = {(r["qid"], r["condition"]) for r in map(json.loads, open(path))} if path.exists() else set()
    client = httpx.AsyncClient(headers={"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
    sem, lock, tally = asyncio.Semaphore(args.concurrency), asyncio.Lock(), dict(rows=0, cost=0.0, cot=0, comp=0)
    hf = Tokenizer.from_pretrained(HF_TOKENIZER) if args.ds_effort else None

    async def one_ds(q, cond) -> tuple[dict, dict]:
        text = modelio.render_ds(mcq(q), SYSTEM[cond], args.ds_effort)
        body = {"model": args.model, "prompt": text, "max_tokens": args.max_tokens, "temperature": args.temperature, "usage": {"include": True},
                "provider": {"order": [args.provider], "allow_fallbacks": False}}
        async with sem:
            b = await complete(client, body)
        assert b["usage"]["prompt_tokens"] == len(hf.encode(text, add_special_tokens=False).ids), (b["usage"]["prompt_tokens"], q["qid"], cond)
        c = b["choices"][0]
        reasoning, content = c.get("reasoning") or "", c.get("text") or ""
        cot = len(hf.encode(reasoning, add_special_tokens=False).ids)
        return b, dict(reasoning=reasoning, content=content, cot=cot, finish_reason=c.get("finish_reason"))

    async def one(q, cond):
        if (q["qid"], cond) in done:
            return
        if args.ds_effort:
            b, d = await one_ds(q, cond)
            usage = b["usage"]
            answer, src = parse_answer(d["content"])
            row = dict(qid=q["qid"], dataset=q["dataset"], condition=cond, correct=q["correct"], answer=answer, answer_src=src, finish="length" if d["finish_reason"] == "length" else "stop",
                       cot_tokens=d["cot"], resp_tokens=(usage.get("completion_tokens") or 0) - d["cot"], reasoning=d["reasoning"], response=d["content"],
                       provider=b.get("provider"), finish_reason=d["finish_reason"], cost=usage.get("cost"), completion_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"))
            return await write(row, usage, d["cot"])
        body = {"model": args.model, "messages": [{"role": "system", "content": SYSTEM[cond]}, {"role": "user", "content": mcq(q)}], "temperature": args.temperature, "max_tokens": args.max_tokens,
                "usage": {"include": True}, "transforms": [], "reasoning": {"effort": args.effort}}
        if args.provider:
            body["provider"] = {"order": [args.provider], "allow_fallbacks": False}
        async with sem:
            b = await sample(client, body)
        ch, usage = b["choices"][0], b["usage"]
        msg = ch["message"]
        reasoning, content = reasoning_text(msg), msg.get("content") or ""
        cot = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        answer, src = parse_answer(content)
        row = dict(qid=q["qid"], dataset=q["dataset"], condition=cond, correct=q["correct"], answer=answer, answer_src=src, finish="length" if ch.get("finish_reason") == "length" or (usage.get("completion_tokens") or 0) >= args.max_tokens - 16 else "stop",  # Z.AI reports "stop" at the cap
                   cot_tokens=cot, resp_tokens=(usage.get("completion_tokens") or 0) - cot, reasoning=reasoning, response=content,
                   provider=b.get("provider"), finish_reason=ch.get("finish_reason"), cost=usage.get("cost"), completion_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"))
        await write(row, usage, cot)

    async def write(row, usage, cot):
        async with lock:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            tally["rows"] += 1
            tally["cost"] += usage.get("cost") or 0
            tally["cot"] += cot
            tally["comp"] += usage.get("completion_tokens") or 0
            if tally["rows"] % 200 == 0:
                print(f"{tally['rows']} done, completion tokens {tally['comp']} (reasoning {tally['cot']}), reported cost ${tally['cost']:.2f}", flush=True)

    await asyncio.gather(*[one(q, c) for q in qs for c in args.conditions])
    print(f"{len(qs)} questions x {len(args.conditions)} conditions; {tally['rows']} new rows, completion tokens {tally['comp']} (reasoning {tally['cot']}), reported cost ${tally['cost']:.2f} -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
