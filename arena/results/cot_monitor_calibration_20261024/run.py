"""Judge saved reasoning only. Run with /home/ek/wgmn/mechtools/.venv/bin/python from /home/ek."""

import argparse
import asyncio
import gzip
import hashlib
import json
import sys
from pathlib import Path

from mechtools import *
from mechtools.openrouter import chat, flat, gather_bar, endpoints

OUT = Path(__file__).resolve().parent
ARENA = OUT.parents[1]
sys.path.insert(0, str(ARENA))

from rl.monitor.formatter import format_cot_prompt, load_template
from rl.monitor.parse import parse_tagged

ARMS = {
    "base20": ("baseline", "openai/gpt-oss-20b", "dekallm/bf16", None),
    "base20_high": ("baseline", "openai/gpt-oss-20b", "dekallm/bf16", "high"),
    "context20": ("context", "openai/gpt-oss-20b", "dekallm/bf16", None),
    "evidence20": ("evidence", "openai/gpt-oss-20b", "dekallm/bf16", None),
    "evidence20_high": ("evidence", "openai/gpt-oss-20b", "dekallm/bf16", "high"),
    "base120_high": ("baseline", "openai/gpt-oss-120b", "dekallm/bf16", "high"),
    "evidence120_high": ("evidence", "openai/gpt-oss-120b", "dekallm/bf16", "high"),
    "evidence120": ("evidence", "openai/gpt-oss-120b", "dekallm/bf16", None),
    "evidence120_medium": ("evidence", "openai/gpt-oss-120b", "dekallm/bf16", "medium"),
    "base_luna_high": ("baseline", "openai/gpt-6-luna", "openai", "high"),
    "base_luna_low": ("baseline", "openai/gpt-6-luna", "openai", "low"),
    "context_luna_high": ("context", "openai/gpt-6-luna", "openai", "high"),
    "context_luna_low": ("context", "openai/gpt-6-luna", "openai", "low"),
    "context120_high": ("context", "openai/gpt-oss-120b", "dekallm/bf16", "high"),
    "evidence_luna_high": ("evidence", "openai/gpt-6-luna", "openai", "high"),
    "base_sonnet_high": ("baseline", "anthropic/claude-sonnet-5.5", "anthropic", "high"),
    "evidence_sonnet_high": ("evidence", "anthropic/claude-sonnet-5.5", "anthropic", "high"),
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_rows(split: str) -> list[dict]:
    if split == "controls_extra":
        with gzip.open(OUT / "controls_extra.jsonl.gz", "rt") as file:
            return [json.loads(line) for line in file]
    if split in ["controls", "pool"]:
        with gzip.open(OUT / "pool.jsonl.gz", "rt") as file:
            rows = [json.loads(line) for line in file]
        return [r for r in rows if r["cohort"] == "control"] if split == "controls" else rows
    return [json.loads(line) for line in (OUT / f"{split}.jsonl").read_text().splitlines()]


def prompt_for(row: dict, variant: str) -> str:
    if variant in ["baseline", "context"]:
        prompt = format_cot_prompt(load_template(OUT / "prompts/baseline.txt", ("task_description", "cot")), row["task_description"], row["cot"], [])
        if variant == "context":
            prompt = (OUT / "prompts/context.txt").read_text() + prompt
        return prompt
    return format_cot_prompt(load_template(OUT / f"prompts/{variant}.txt", ("task_description", "cot")), row["task_description"], row["cot"], [])


async def main(args: argparse.Namespace) -> None:
    rows = read_rows(args.split)
    if args.limit:
        rows = rows[:args.limit]
    arms = args.arms.split(",")
    if args.metadata:
        metadata = {model: endpoints(model) for model in dict.fromkeys(ARMS[a][1] for a in arms)}
        (OUT / "endpoints.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({m: len(e) for m, e in metadata.items()}), flush=True)
        return
    requests = []
    for arm in arms:
        variant, model, provider, effort = ARMS[arm]
        for repeat in range(args.repeats):
            for row in rows:
                prompt = prompt_for(row, variant)
                if variant == "baseline":
                    assert digest(prompt) == row["baseline_prompt_sha256"]
                config = {"model": model, "provider": provider, "reasoning": effort, "temperature": 0, "max_tokens": args.max_tokens}
                requests.append({"id": row["id"], "arm": arm, "repeat": repeat, "prompt": prompt, "config": config, "mode": "cot_only", "input_channels": ["reasoning"], "main_task_only": True, "prompt_sha256": digest(prompt)})
    print(json.dumps({"split": args.split, "cases": len(rows), "arms": arms, "requests": len(requests), "send": args.send}), flush=True)
    if not args.send:
        return
    directory = OUT / "working" / args.split
    directory.mkdir(parents=True, exist_ok=True)

    async def score(request: dict) -> dict:
        path = directory / f"{request['id']}.{request['arm']}.{request['repeat']}.json"
        if path.exists():
            stored = json.loads(path.read_text())
            assert stored["request"] == request
            return stored
        body = await chat(request["prompt"], **request["config"], attempts=3, timeout=180)
        fields = flat(body)
        value, explanation = parse_tagged(fields["text"] or "") if fields["finish_reason"] == "stop" else (None, None)
        stored = {"request": request, "response": body, "score": value, "explanation": explanation, "finish_reason": fields["finish_reason"], "cost_usd": fields["cost"]}
        path.write_text(json.dumps(stored, ensure_ascii=False) + "\n")
        return stored

    result = await gather_bar([score(r) for r in requests], concurrency=args.concurrency, desc=args.split)
    compact = []
    for request, row in zip(requests, result, strict=True):
        compact.append({**{k: v for k, v in request.items() if k != "prompt"}, **({k: row[k] for k in ["score", "explanation", "finish_reason", "cost_usd"]} if row else {"score": None, "request_failed": True})})
    (OUT / f"{args.split}.{args.tag}.scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in compact))
    print(json.dumps({"parsed": sum(r['score'] is not None for r in compact), "total": len(compact), "cost_usd_including_cached": sum(r.get('cost_usd', 0) for r in compact)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--tag", default="screen")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--send", action="store_true")
    asyncio.run(main(parser.parse_args()))
