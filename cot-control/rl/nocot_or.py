"""No-CoT accuracy of Gemma 4 on the CoT-Control QA items (all 1214: gpqa 445, hle 469, mmlu_pro 300), through OpenRouter raw text
completions with the Gemma 4 chat template rendered here (thinking off = the generation prompt ends with an empty thought block,
'<|turn>model\\n<|channel>thought\\n<channel|>', as the HF template renders it with enable_thinking false), temperature 0, provider
pinned (Crusoe serves bf16 and passes raw prompts through: a bare 'Hello' counts 1 prompt token; chat completions there render an
assistant prefill as a closed turn, which is why this uses raw completions). Two arms per item:
  prefill  the prompt ends with 'ANSWER:' so the first sampled token is the letter (nocot-bench style forced answer)
  free     no prefill, --max-tokens-free tokens, the answer parsed from the response; `immediate` records whether the response
           opened with ANSWER (no visible reasoning before the letter)
Reports each dataset and the whole pool ('normal'), and the 107 unique items of the 108-unit Astra draw ('astra', a subset of
the pool), next to the chance level (mean 1/n_options).

    python nocot_or.py --model google/gemma-4-31b-it --provider Crusoe --name gemma4-31b --out ../results/nocot/gemma4_31b --shard 0/2
    python nocot_or.py --model google/gemma-4-31b-it --provider Friendli --name gemma4-31b --out ../results/nocot/gemma4_31b --shard 1/2
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from cotswitch import extract, prompts
from cotswitch.data import DATASETS, load_items

OR_URL = "https://openrouter.ai/api/v1/completions"


def render(user: str, prefill: str) -> str:
    return f"<bos><|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{prefill}"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="prefill,free")
    ap.add_argument("--max-tokens-free", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--shard", default="0/1", help="i/n: this process runs items i, i+n, ... (several providers can split one run)")
    args = ap.parse_args()
    shard, nshards = map(int, args.shard.split("/"))
    items = [it for ds in DATASETS for it in load_items(ds)][shard::nshards]
    astra = {(it.dataset, it.item_idx) for cell in prompts.astra_sample().values() for it in cell}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows.jsonl"
    done = {(r["dataset"], r["item_idx"], r["arm"]) for r in map(json.loads, open(path))} if path.exists() else set()
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}

    async with httpx.AsyncClient() as client:
        async def one(it, arm):
            if (it.dataset, it.item_idx, arm) in done:
                return
            prefill = "ANSWER:" if arm == "prefill" else ""
            body = {"model": args.model, "prompt": render(prompts.question_block(it), prefill), "max_tokens": 8 if arm == "prefill" else args.max_tokens_free,
                    "temperature": 0, "usage": {"include": True}, "provider": {"order": [args.provider], "allow_fallbacks": False}}
            b, err = None, ""
            async with sem:
                for attempt in range(20):
                    try:
                        r = await client.post(OR_URL, json=body, headers=headers, timeout=600)
                        b = r.json()
                        if r.status_code == 200 and "choices" in b and b["choices"][0].get("finish_reason") != "error":
                            break
                        err, b = f"HTTP {r.status_code} {str(b)[:200]}", None
                    except (httpx.HTTPError, ValueError) as e:
                        err, b = repr(e), None
                    print(f"retry {attempt} {it.dataset} {it.item_idx} {arm}: {err}", file=sys.stderr)
                    await asyncio.sleep(min(10 * (attempt + 1), 60))
            if b is None:
                return
            ch, usage = b["choices"][0], b.get("usage") or {}
            text = ch.get("text") or ""
            pred = extract.parse_answer(prefill + text, prefill + text)
            row = dict(model=args.name, or_model=args.model, provider=b.get("provider"), dataset=it.dataset, item_idx=it.item_idx, arm=arm,
                       astra=(it.dataset, it.item_idx) in astra, n_options=len(it.options), answer=it.answer_letter, answer_pred=pred,
                       correct=pred == it.answer_letter, immediate=(prefill + text).strip().upper().startswith("ANSWER"),
                       finish=ch.get("finish_reason"), prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                       cost=usage.get("cost"), response=text)
            async with lock:
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

        await asyncio.gather(*(one(it, arm) for it in items for arm in args.arms.split(",")))

    rows = [json.loads(l) for l in open(path) if json.loads(l)["model"] == args.name]
    lines = [f"{args.name} ({args.model} via {sorted({r['provider'] for r in rows})}, raw completions, thinking off, T 0): cost ${sum(r['cost'] or 0 for r in rows):.2f}", "",
             "| set | dataset | arm | n | accuracy | chance | unparsed | answer immediate | median completion tokens |", "|---|---|---|---|---|---|---|---|---|"]
    for name, sel in (("normal", lambda r: True), ("astra", lambda r: r["astra"])):
        for ds in DATASETS + ["all"]:
            for arm in args.arms.split(","):
                rs = [r for r in rows if sel(r) and r["arm"] == arm and (ds == "all" or r["dataset"] == ds)]
                if not rs:
                    continue
                toks = sorted(r["completion_tokens"] or 0 for r in rs)
                lines.append(f"| {name} | {ds} | {arm} | {len(rs)} | {sum(r['correct'] for r in rs)}/{len(rs)} = {sum(r['correct'] for r in rs) / len(rs):.3f} | "
                             f"{sum(1 / r['n_options'] for r in rs) / len(rs):.3f} | {sum(r['answer_pred'] is None for r in rs)} | "
                             f"{sum(r['immediate'] for r in rs)}/{len(rs)} | {toks[len(toks) // 2]} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
