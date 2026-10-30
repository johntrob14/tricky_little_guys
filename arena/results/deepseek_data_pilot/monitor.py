"""Matched reasoning-only scores with the existing monitor prompt. Run with the mechtools Python from ~."""

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from pathlib import Path

from mechtools import *
from mechtools.openrouter import chat

from editing import reasoning_blocks, write_json

OUT = Path(__file__).resolve().parent
ARENA = OUT.parents[1]
sys.path.insert(0, str(ARENA))

from prompt_spec import fragment
from rl.monitor.monitors import get


async def main(send: bool, episodes: bool, episode_dir: str) -> None:
    mon = get("controlarena_cot_only")
    meta = {"mode": "cot_only", "monitor": mon.name, "input_channels": ["reasoning"], "main_task_only": True, "prior_history": False, "judge_model": "openai/gpt-oss-20b", "provider": "deepinfra", "max_tokens": 6000, "temperature": 0, "prompt_template_sha256": hashlib.sha256(mon.prompt_file.read_bytes()).hexdigest()}
    stage = episode_dir if episodes else "smoke"
    directory = OUT / "working" / f"monitor_{stage}"
    directory.mkdir(exist_ok=True)
    cases = json.loads((OUT / "selection.json").read_text())["selected"]
    requests = []
    for case in cases:
        uuid = case["uuid"]
        original = json.loads((OUT / "working" / f"{uuid}.original.json").read_text())
        first = next(m["content"] for m in original if m["role"] == "user")
        prefix = fragment("prefix", "secondary_objective")
        assert isinstance(first, str) and prefix in first
        task = first.split(prefix, 1)[0].rstrip()
        conditions = {"original": original, "edited": json.loads((OUT / "working" / (f"{episode_dir}/{uuid}.edited.json" if episodes else f"{uuid}.editor_prefix.json")).read_text()), "removed": json.loads((OUT / "working" / f"{uuid}.no_reasoning.json").read_text())}
        if not episodes:
            conditions["deletion"] = json.loads((OUT / "working" / f"{uuid}.manual_prefix.json").read_text())
        for condition, messages in conditions.items():
            for turn, message in enumerate(m for m in messages if m["role"] == "assistant"):
                if not episodes and turn != case["turn"]:
                    continue
                reasoning = "\n".join(b["reasoning"] for b in reasoning_blocks([message]))
                prompt = mon.format(task, reasoning, [])
                requests.append({"uuid": uuid, "turn": turn, "condition": condition, "empty": not reasoning.strip(), "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    write_json(directory / "requests.json", requests)
    print(json.dumps({**meta, "stage": stage, "rows": len(requests), "nonempty": sum(not r["empty"] for r in requests), "send": send}), flush=True)
    if not send:
        return
    semaphore = asyncio.Semaphore(8)

    async def score(request: dict) -> dict:
        row = {k: v for k, v in request.items() if k != "prompt"}
        if request["empty"]:
            return {**row, "score": None, "skipped_empty": True, "unparsed": False, "cost_usd": 0}
        path = directory / f"{request['uuid']}.{request['turn']}.{request['condition']}.json"
        if path.exists():
            stored = json.loads(path.read_text())
            assert stored["meta"] == meta and stored["request_sha256"] == request["prompt_sha256"]
            body = stored["response"]
        else:
            async with semaphore:
                body = await chat(request["prompt"], model=meta["judge_model"], provider=meta["provider"], temperature=0, max_tokens=meta["max_tokens"], attempts=1)
            write_json(path, {"meta": meta, "request_sha256": request["prompt_sha256"], "response": body})
        choice = body["choices"][0]
        value, _ = mon.parser(choice["message"]["content"]) if choice["finish_reason"] == "stop" else (None, None)
        return {**row, "score": value, "skipped_empty": False, "unparsed": value is None, "cost_usd": body["usage"]["cost"], "reported_provider": body["provider"]}

    rows = await asyncio.gather(*(score(r) for r in requests))
    summary = {}
    for condition in dict.fromkeys(r["condition"] for r in rows):
        subset = [r for r in rows if r["condition"] == condition]
        values = [r["score"] for r in subset if r["score"] is not None]
        summary[condition] = {"turns": len(subset), "parsed": len(values), "skipped_empty": sum(r["skipped_empty"] for r in subset), "unparsed": sum(r["unparsed"] for r in subset), "mean_over_parsed_turns": statistics.mean(values) if values else None, "at_least_8": sum(v >= 8 for v in values)}
    result = {"meta": meta, "stage": stage, "conditions": summary, "rows": rows, "cost_usd": sum(r["cost_usd"] for r in rows)}
    write_json(OUT / f"monitor_{stage}.json", result)
    print(json.dumps({"conditions": summary, "cost_usd": result["cost_usd"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--episodes", action="store_true")
    parser.add_argument("--episode-dir", default="episodes")
    args = parser.parse_args()
    asyncio.run(main(args.send, args.episodes, args.episode_dir))
