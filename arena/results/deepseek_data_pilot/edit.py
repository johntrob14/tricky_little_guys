"""Inspect requests by default; --send submits the prepared pilot to the editor after prompt approval.

Run from /home/ek with /home/ek/wgmn/mechtools/.venv/bin/python so the shared helpers load ~/.env.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from mechtools import *
from mechtools.openrouter import chat, flat

OUT = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4.1-flash-1017"
PROVIDER = "parasail"
MAX_TOKENS = 16000


async def main(send: bool) -> None:
    prompt = (OUT / "editor_prompt.txt").read_text()
    requests = [json.loads(p.read_text()) for p in sorted((OUT / "working").glob("*.request.json"))]
    assert len(requests) == 4
    for request in requests:
        assert request["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert request["request_sha256"] == hashlib.sha256(json.dumps(request["payload"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    config = {"model": MODEL, "provider": PROVIDER, "reasoning": "medium", "temperature": 0, "max_tokens": MAX_TOKENS, "prompt_sha256": requests[0]["prompt_sha256"]}
    print(json.dumps({**config, "send": send, "requests": len(requests), "request_chars": [len(json.dumps(r["payload"])) + len(prompt) for r in requests]}), flush=True)
    if not send:
        return
    responses = OUT / "working/editor"
    responses.mkdir(exist_ok=True)
    for request in requests:
        path = responses / f"{request['uuid']}.json"
        assert not path.exists(), f"Refusing to overwrite an attempted edit: {path}"
        body = await chat([{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(request["payload"], ensure_ascii=False)}], model=MODEL, provider=PROVIDER, reasoning="medium", temperature=0, max_tokens=MAX_TOKENS, attempts=1)
        record = {"config": config, "request_sha256": request["request_sha256"], "source_sha256": request["source_sha256"], "uuid": request["uuid"], "turn": request["turn"], "response": body}
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        result = flat(body)
        print(json.dumps({"uuid": request["uuid"], "provider": result["provider"], "finish_reason": result["finish_reason"], "cost_usd": result["cost"]}), flush=True)
        assert result["finish_reason"] == "stop", "Incomplete editor reply; raw response retained"
        parsed = json.loads(result["text"])
        assert set(parsed) == {"edits", "review_notes", "change_magnitude"} and isinstance(parsed["edits"], list) and isinstance(parsed["review_notes"], str)
        assert type(parsed["change_magnitude"]) is int and 0 <= parsed["change_magnitude"] <= 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    asyncio.run(main(parser.parse_args().send))
