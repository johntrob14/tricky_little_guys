"""Inspect requests by default; --send runs the approved editor, --episodes edits complete histories chronologically.

Run from /home/ek with /home/ek/wgmn/mechtools/.venv/bin/python so the shared helpers load ~/.env.
"""

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path

from mechtools import *
from mechtools.openrouter import chat

from editing import digest, editor_view, parse_reply, reasoning_blocks, set_reasoning, without_reasoning, write_json

OUT = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4.1-flash-1017"
PROVIDER = "parasail"
MAX_TOKENS = 16000


async def ask(request: dict, prompt: str, config: dict, path: Path) -> tuple[dict, str]:
    assert request["prompt_sha256"] == config["prompt_sha256"]
    assert request["request_sha256"] == digest(request["payload"])
    cached = path.exists()
    if cached:
        record = json.loads(path.read_text())
        assert record["config"] == config and record["request_sha256"] == request["request_sha256"] and record["source_sha256"] == request["source_sha256"]
        body = record["response"]
    else:
        body = await chat([{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(request["payload"], ensure_ascii=False)}], model=config["model"], provider=config["provider"], reasoning=config["reasoning"], temperature=config["temperature"], max_tokens=config["max_tokens"], attempts=1)
        record = {"config": config, "request_sha256": request["request_sha256"], "source_sha256": request["source_sha256"], "uuid": request["uuid"], "turn": request["turn"], "response": body}
        write_json(path, record)
    before = reasoning_blocks([request["payload"]["current"]])[0]["reasoning"]
    reply, after = parse_reply(body, before)
    if not cached:
        print(json.dumps({"uuid": request["uuid"], "turn": request["turn"], "provider": body["provider"], "change_magnitude": reply["change_magnitude"], "cost_usd": body["usage"]["cost"]}), flush=True)
    return reply, after


async def episode(case: dict, prompt: str, config: dict, root: Path) -> dict:
    uuid = case["uuid"]
    directory = root / uuid
    directory.mkdir(parents=True, exist_ok=True)
    overrides = json.loads((root / "overrides.json").read_text()) if (root / "overrides.json").exists() else {}
    original = json.loads((OUT / "working" / f"{uuid}.original.json").read_text())
    source_hash = json.loads((OUT / "working" / f"{uuid}.request.json").read_text())["source_sha256"]
    assert hashlib.sha256((OUT.parents[1] / case["log"]).read_bytes()).hexdigest() == source_hash
    history, rows = [], []
    turn = -1
    for index, original_message in enumerate(original):
        current = copy.deepcopy(original_message)
        if current["role"] == "assistant":
            turn += 1
            blocks = reasoning_blocks([current])
            if not blocks:
                rows.append({"turn": turn, "status": "original_without_reasoning"})
            else:
                assert len(blocks) == 1 and blocks[0]["reasoning"].strip()
                before = blocks[0]["reasoning"]
                payload = {"history": editor_view(history), "current": editor_view([current])[0]}
                assert len(history) == index
                request = {"uuid": uuid, "turn": turn, "source_sha256": source_hash, "prompt_sha256": config["prompt_sha256"], "payload": payload, "request_sha256": digest(payload)}
                request_path = directory / f"{turn:03d}.request.json"
                if request_path.exists():
                    assert json.loads(request_path.read_text()) == request
                else:
                    write_json(request_path, request)
                turn_config = {**config, **overrides.get(f"{uuid}:{turn}", {})}
                reply, after = await ask(request, prompt, turn_config, directory / f"{turn:03d}.response.json")
                set_reasoning(blocks[0], after)
                assert without_reasoning([current]) == without_reasoning([original_message])
                rows.append({"turn": turn, "status": "edited" if before != after else "unchanged", "editor_reasoning": turn_config["reasoning"], "original_chars": len(before), "edited_chars": len(after), "replacement_chars": sum(len(e["new"]) for e in reply["edits"]), "change_magnitude": reply["change_magnitude"], "review_notes": reply["review_notes"], "request_sha256": request["request_sha256"], "edited_reasoning_sha256": digest(after)})
        history.append(current)
    assert without_reasoning(original) == without_reasoning(history)
    assert hashlib.sha256((OUT.parents[1] / case["log"]).read_bytes()).hexdigest() == source_hash
    write_json(root / f"{uuid}.edited.json", history)
    result = {"uuid": uuid, "source_sha256": source_hash, "original_messages_sha256": digest(original), "edited_messages_sha256": digest(history), "config": config, "turns": rows, "non_reasoning_preserved": True}
    write_json(directory / "summary.json", result)
    return {"uuid": uuid, "complete": True, "turns": len(rows), "edited": sum(r["status"] == "edited" for r in rows), "unchanged": sum(r["status"] == "unchanged" for r in rows), "original_without_reasoning": sum(r["status"] == "original_without_reasoning" for r in rows)}


async def main(send: bool, episodes: bool, no_thinking: bool = False, model: str = MODEL, provider: str = PROVIDER, name: str = "", effort: str = "medium") -> None:
    prompt = (OUT / "editor_prompt.txt").read_text()
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    assert json.loads((OUT / "approval.json").read_text())["prompt_sha256"] == prompt_hash
    config = {"model": model, "provider": provider, "reasoning": False if no_thinking else effort, "temperature": 0, "max_tokens": MAX_TOKENS, "prompt_sha256": prompt_hash}
    suffix = f"_{name}" if name else ("_no_thinking" if no_thinking else "")
    cases = json.loads((OUT / "selection.json").read_text())["selected"]
    print(json.dumps({**config, "send": send, "episodes": episodes, "selected_episodes": len(cases)}), flush=True)
    if not send:
        return
    if episodes:
        root = OUT / "working" / f"episodes{suffix}"
        outcomes = await asyncio.gather(*(episode(case, prompt, config, root) for case in cases), return_exceptions=True)
        results = [{"uuid": case["uuid"], "complete": False, "error": f"{type(result).__name__}: {result}"} if isinstance(result, BaseException) else result for case, result in zip(cases, outcomes)]
        write_json(root / "status.json", results)
        print(json.dumps(results), flush=True)
        assert all(r["complete"] for r in results), f"Episode editing failed; all attempted replies retained in {root}"
    else:
        directory = OUT / "working" / f"editor{suffix}"
        directory.mkdir(exist_ok=True)
        for case in cases:
            request = json.loads((OUT / "working" / f"{case['uuid']}.request.json").read_text())
            await ask(request, prompt, config, directory / f"{case['uuid']}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--episodes", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--name", default="")
    parser.add_argument("--effort", default="medium")
    args = parser.parse_args()
    asyncio.run(main(args.send, args.episodes, args.no_thinking, args.model, args.provider, args.name, args.effort))
