"""Reconstruct an edited corpus from committed source logs and span edits, with no model calls."""

import argparse
import json

from editing import digest, reasoning_blocks, set_reasoning, write_json
from prepare import OUT, validate


def main(condition: str) -> None:
    rows = [json.loads(line) for line in (OUT / f"{condition}_edits.jsonl").read_text().splitlines()]
    summary = json.loads((OUT / f"{condition}_results.json").read_text())
    root = OUT / "working" / f"{condition}_replayed"
    root.mkdir(exist_ok=True)
    for episode in summary["episodes"]:
        uuid = episode["uuid"]
        original = json.loads((OUT / "working" / f"{uuid}.original.json").read_text())
        edited = json.loads(json.dumps(original))
        assistant = [m for m in edited if m["role"] == "assistant"]
        for row in (r for r in rows if r["uuid"] == uuid):
            block = reasoning_blocks([assistant[row["turn"]]])[0]
            text = block["reasoning"]
            assert digest(text) == row["before_sha256"]
            for span in row["sequential_spans"]:
                assert 0 <= span["start"] <= span["end"] <= len(text)
                text = text[:span["start"]] + span["new"] + text[span["end"]:]
            assert digest(text) == row["after_sha256"]
            set_reasoning(block, text)
        validate(original, edited)
        write_json(root / f"{uuid}.json", edited)
    print(f"Reconstructed {len(summary['episodes'])}/{summary['planned']} episodes without model calls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition")
    main(parser.parse_args().condition)
