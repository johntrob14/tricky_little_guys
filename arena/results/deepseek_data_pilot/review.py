"""Apply returned editor patches, verify preservation, and write exact diffs. No model calls."""

import copy
import difflib
import hashlib
import json
from pathlib import Path

from prepare import OUT, apply_edits, digest, reasoning_blocks, set_reasoning, validate, write_json


def main() -> None:
    paths = sorted((OUT / "working/editor").glob("*.json"))
    assert paths, "No editor responses; the prompt needs approval before running edit.py --send"
    results, diffs = [], []
    for path in paths:
        record = json.loads(path.read_text())
        uuid = record["uuid"]
        request = json.loads((OUT / "working" / f"{uuid}.request.json").read_text())
        assert record["request_sha256"] == digest(request["payload"])
        assert record["config"]["prompt_sha256"] == hashlib.sha256((OUT / "editor_prompt.txt").read_bytes()).hexdigest()
        parent = json.loads((OUT / "working" / f"{uuid}.original.json").read_text())[:len(request["payload"]["history"]) + 1]
        choice = record["response"]["choices"][0]
        assert choice["finish_reason"] == "stop"
        reply = json.loads(choice["message"]["content"])
        assert type(reply["change_magnitude"]) is int and 0 <= reply["change_magnitude"] <= 10
        before = reasoning_blocks([parent[-1]])[0]["reasoning"]
        after = apply_edits(before, reply["edits"])
        assert after.strip(), "Edited reasoning must stay nonempty"
        edited = copy.deepcopy(parent)
        set_reasoning(reasoning_blocks([edited[-1]])[0], after)
        validate(parent, edited)
        write_json(OUT / "working" / f"{uuid}.editor_prefix.json", edited)
        results.append({"uuid": uuid, "turn": record["turn"], "original_chars": len(before), "edited_chars": len(after), "inserted_chars": sum(len(e["new"]) for e in reply["edits"]), "edited_reasoning_sha256": digest(after), "empty": not after.strip(), "review_notes": reply["review_notes"], "change_magnitude": reply["change_magnitude"], "provider": record["response"]["provider"], "cost_usd": record["response"]["usage"]["cost"], "non_reasoning_preserved": True, "manual_semantic_review": "pending"})
        diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="original", tofile="editor", n=0, lineterm=""))
        diffs.append(f"## {uuid}, turn {record['turn']}\n\n```diff\n{diff}\n```\n")
    write_json(OUT / "editor_results.json", {"attempted": len(paths), "planned": 4, "results": results, "cost_usd": sum(r["cost_usd"] for r in results)})
    (OUT / "working/editor_diffs.md").write_text("\n".join(diffs))
    print(json.dumps({"validated": f"{len(results)}/{len(paths)}", "cost_usd": sum(r["cost_usd"] for r in results)}))


if __name__ == "__main__":
    main()
