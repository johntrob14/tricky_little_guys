"""Same approved editor prompt on all six failed DeepSeek-editor inputs, using GPT-6 Luna."""

import asyncio
import hashlib
import json

from edit import MAX_TOKENS, OUT, ask
from editing import write_json

CASES = [("episodes", "86Ss455WKBawGPBpVxTe8L", 9), ("episodes", "cK8YbWg85FqMJNwMj4i4fg", 10), ("episodes", "8w8umgUDCoGp2ja4MWzpox", 11), ("episodes_no_thinking", "LjW5WaW8fJg6t9fXfDngeA", 15), ("episodes_no_thinking", "86Ss455WKBawGPBpVxTe8L", 6), ("episodes_no_thinking", "cK8YbWg85FqMJNwMj4i4fg", 5)]


async def main() -> None:
    prompt = (OUT / "editor_prompt.txt").read_text()
    config = {"model": "openai/gpt-6-luna", "provider": "openai", "reasoning": "high", "temperature": 0, "max_tokens": MAX_TOKENS, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    directory = OUT / "working/luna_probe"
    directory.mkdir(exist_ok=True)
    results = []
    for condition, uuid, turn in CASES:
        request = json.loads((OUT / "working" / condition / uuid / f"{turn:03d}.request.json").read_text())
        reply, after = await ask(request, prompt, config, directory / f"{uuid}.{turn}.json")
        results.append({"uuid": uuid, "turn": turn, "input_condition": condition, "valid": True, "edited_chars": len(after), "change_magnitude": reply["change_magnitude"], "review_notes": reply["review_notes"]})
    write_json(OUT / "luna_probe.json", {"config": config, "selected_failed_inputs": len(CASES), "valid": len(results), "results": results})


if __name__ == "__main__":
    asyncio.run(main())
