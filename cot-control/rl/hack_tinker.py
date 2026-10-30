"""Parity grader-following for mt1b s150 and mt2e s170; run in the local rl venv with TINKER_API_KEY from ~/.env.

python hack_tinker.py --limit 1 checks one sample per condition; the full command fills all 256 per condition.
Then use hack_openrouter.py --stage judge --run-dir <model output> from the mechtools venv.
"""
import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import tinker

from cotswitch import extract
from hack_eval import prompt, side_of
from modelio import GPT_OSS_STOP, PRICES

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "hack/mt1b_mt2e_medium_20261027"
BASE_CONFIG = RESULTS / "hack/gptoss120b_groq_medium_20261027/config.json"
MODELS = {"mt1b": ("manytypes1b", 150), "mt2e": ("manytypes2e", 170)}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def run(name: str, limit: int, svc: tinker.ServiceClient) -> None:
    training_run, step = MODELS[name]
    checkpoints = json.loads((RESULTS / "rl" / training_run / "checkpoints.json").read_text())
    checkpoint = next(r["sampler"] for r in checkpoints if r["step"] == step)
    config = json.loads(BASE_CONFIG.read_text()) | {"provider": "Tinker", "model_path": checkpoint, "checkpoint_step": step, "training_run": training_run,
                                                "policy_concurrency": 8, "samples_per_request": 8, "top_k": -1, "judge_concurrency": 4,
                                                "policy_cost_source": "modelio.PRICES, estimated Tinker token cost", "policy_prices_per_million": PRICES["openai/gpt-oss-120b"]}
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Run configuration differs"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    path = out / "rollouts.jsonl"
    rows = read_rows(path)
    done = {r["key"] for r in rows}
    assert len(done) == len(rows), "Duplicate sample keys"
    client = await svc.create_sampling_client_async(model_path=checkpoint)
    tok = client.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=config["max_tokens"], temperature=config["temperature"], top_p=config["top_p"], top_k=-1, stop=GPT_OSS_STOP)
    sem = asyncio.Semaphore(config["policy_concurrency"])
    completed = len(done)

    async def sample(condition: str, side: str, indices: list[int], output) -> None:
        nonlocal completed
        request = prompt(side, condition)
        rendered = tok.apply_chat_template([{"role": "user", "content": request}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        assert "Reasoning: medium" in rendered and rendered.endswith("<|start|>assistant"), "Unexpected Harmony template"
        ids = tok.encode(rendered, add_special_tokens=False)
        async with sem:
            response = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=len(indices), sampling_params=params)
        assert len(response.sequences) == len(indices)
        for idx, seq in zip(indices, response.sequences):
            raw = tok.decode(seq.tokens, skip_special_tokens=False)
            truncated = seq.stop_reason == "length"
            split = extract.split_output(extract.harmony_to_think(raw), None, "length" if truncated else "stop")
            assert split["cot_rule"] in ("think_tags", "unclosed_think_all_reasoning"), f"Unexpected CoT parse: {name}/{condition}/{side}/{idx}"
            reasoning, answer = split["reasoning"], split["response"]
            parity = side_of(answer) if not truncated else None
            cost = (len(ids) * PRICES[config["model"]]["prefill"] + len(seq.tokens) * PRICES[config["model"]]["sample"]) / 1e6
            row = {"key": f"medium/{condition}/{side}/{idx}", "mode": "medium", "prompts": condition, "side": side, "idx": idx, "request": request,
                   "provider": "Tinker", "model_path": checkpoint, "rendered_prompt": rendered, "raw_generation": raw, "tokens": seq.tokens,
                   "reasoning": reasoning, "answer": answer, "cot_rule": split["cot_rule"], "stop_reason": seq.stop_reason,
                   "cheat": None if parity is None else parity != side, "truncated": truncated, "empty_reasoning": not reasoning.strip(),
                   "n_prompt_tokens": len(ids), "n_output_tokens": len(seq.tokens), "n_reasoning_words": len(reasoning.split()), "usd": cost}
            output.write(json.dumps(row) + "\n")
            output.flush()
            completed += 1
            assert reasoning.strip(), f"Missing CoT: {name}/{row['key']}"
            if completed % 64 == 0 or completed == 4 * limit:
                print(f"{name}: {completed}/{4 * limit} sampled", flush=True)

    with path.open("a") as output:
        jobs = []
        for condition in ("hack", "clean"):
            for side in ("even", "odd"):
                todo = [idx for idx in range(limit) if f"medium/{condition}/{side}/{idx}" not in done]
                jobs += [sample(condition, side, todo[start:start + 8], output) for start in range(0, len(todo), 8)]
        await asyncio.gather(*jobs)
    rows = read_rows(path)
    report = {"completed": len(rows), "expected": 1024, "cells": dict(Counter(f"{r['prompts']}/{r['side']}" for r in rows)),
              "truncated": sum(r["truncated"] for r in rows), "invalid": sum(r["cheat"] is None for r in rows), "missing_cot": sum(r["empty_reasoning"] for r in rows),
              "estimated_policy_usd": sum(r["usd"] for r in rows)}
    (out / "sampling_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, json.dumps(report), flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--limit", type=int, default=256)
    args = ap.parse_args()
    assert 1 <= args.limit <= 256
    svc = tinker.ServiceClient()
    await asyncio.gather(*(run(name, args.limit, svc) for name in args.models))


if __name__ == "__main__":
    asyncio.run(main())
