"""Retry the three failed thinking-editor inputs unchanged, with editor thinking disabled. No transcript continuation."""

import asyncio
import hashlib
import json

from edit import MAX_TOKENS, MODEL, OUT, PROVIDER, ask
from editing import write_json

CASES = [("86Ss455WKBawGPBpVxTe8L", 9), ("cK8YbWg85FqMJNwMj4i4fg", 10), ("8w8umgUDCoGp2ja4MWzpox", 11)]


async def main() -> None:
    prompt = (OUT / "editor_prompt.txt").read_text()
    config = {"model": MODEL, "provider": PROVIDER, "reasoning": False, "temperature": 0, "max_tokens": MAX_TOKENS, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    directory = OUT / "working/no_thinking_probe"
    directory.mkdir(exist_ok=True)
    results = []
    for uuid, turn in CASES:
        request = json.loads((OUT / "working/episodes" / uuid / f"{turn:03d}.request.json").read_text())
        reply, after = await ask(request, prompt, config, directory / f"{uuid}.{turn}.json")
        results.append({"uuid": uuid, "turn": turn, "valid": True, "edited_chars": len(after), "change_magnitude": reply["change_magnitude"], "review_notes": reply["review_notes"]})
    write_json(OUT / "no_thinking_probe.json", {"config": config, "selected_failed_inputs": len(CASES), "valid": len(results), "results": results})


if __name__ == "__main__":
    asyncio.run(main())
