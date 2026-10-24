"""Validate complete edited episodes, export reproducible span edits, and collect factual summaries. No model calls."""

import argparse
import difflib
import hashlib
import json
import statistics
from collections import Counter

from editing import digest, editor_view, parse_reply, reasoning_blocks, write_json
from prepare import ARENA, OUT, validate


def main(episode_dir: str) -> None:
    cases = json.loads((OUT / "selection.json").read_text())["selected"]
    episodes, edits, comparisons, configs = [], [], [], {}
    for case in cases:
        uuid = case["uuid"]
        directory = OUT / "working" / episode_dir / uuid
        original = json.loads((OUT / "working" / f"{uuid}.original.json").read_text())
        edited = json.loads((OUT / "working" / episode_dir / f"{uuid}.edited.json").read_text())
        validate(original, edited)
        source_hash = hashlib.sha256((ARENA / case["log"]).read_bytes()).hexdigest()
        rows, providers, cost = [], Counter(), 0.0
        turn = -1
        for index, (before_message, after_message) in enumerate(zip(original, edited)):
            if before_message["role"] != "assistant":
                continue
            turn += 1
            before_blocks = reasoning_blocks([before_message])
            if not before_blocks:
                rows.append({"turn": turn, "status": "original_without_reasoning"})
                continue
            before = before_blocks[0]["reasoning"]
            request = json.loads((directory / f"{turn:03d}.request.json").read_text())
            record = json.loads((directory / f"{turn:03d}.response.json").read_text())
            assert request["payload"] == {"history": editor_view(edited[:index]), "current": editor_view([before_message])[0]}
            assert request["request_sha256"] == record["request_sha256"] == digest(request["payload"])
            assert record["source_sha256"] == request["source_sha256"] == source_hash
            reply, after = parse_reply(record["response"], before)
            key = digest(record["config"])
            configs.setdefault(key, {"config": record["config"], "calls": 0})["calls"] += 1
            assert reasoning_blocks([after_message])[0]["reasoning"] == after
            providers[record["response"]["provider"]] += 1
            cost += record["response"]["usage"]["cost"]
            spans, text = [], before
            for edit in reply["edits"]:
                start = text.index(edit["old"])
                end = start + len(edit["old"])
                spans.append({"start": start, "end": end, "new": edit["new"]})
                text = text[:start] + edit["new"] + text[end:]
            assert text == after
            edits.append({"uuid": uuid, "turn": turn, "source_sha256": source_hash, "before_sha256": digest(before), "after_sha256": digest(after), "sequential_spans": spans})
            opcodes = difflib.SequenceMatcher(None, before, after).get_opcodes()
            added = sum(j2-j1 for op,i1,i2,j1,j2 in opcodes if op in ("insert", "replace"))
            removed = sum(i2-i1 for op,i1,i2,j1,j2 in opcodes if op in ("delete", "replace"))
            rows.append({"turn": turn, "status": "edited" if before != after else "unchanged", "original_chars": len(before), "edited_chars": len(after), "diff_added_chars": added, "diff_removed_chars": removed, "change_magnitude": reply["change_magnitude"], "review_notes": reply["review_notes"], "request_sha256": request["request_sha256"]})
            if turn == case["turn"]:
                smoke = json.loads((OUT / "working/editor" / f"{uuid}.json").read_text())
                _, previous = parse_reply(smoke["response"], before)
                comparisons.append({"uuid": uuid, "turn": turn, "same_request_payload": smoke["request_sha256"] == record["request_sha256"], "same_config": smoke["config"] == record["config"], "same_edited_reasoning": previous == after})
        nonempty = [r for r in rows if r["status"] != "original_without_reasoning"]
        episodes.append({"uuid": uuid, "task": case["task"], "tier": case["tier"], "turns": len(rows), "nonempty": len(nonempty), "changed": sum(r["status"] == "edited" for r in rows), "original_chars": sum(r["original_chars"] for r in nonempty), "edited_chars": sum(r["edited_chars"] for r in nonempty), "mean_change_magnitude": statistics.mean(r["change_magnitude"] for r in nonempty), "median_change_magnitude": statistics.median(r["change_magnitude"] for r in nonempty), "providers": dict(providers), "cost_usd": cost, "source_sha256": source_hash, "non_reasoning_preserved": True, "chronological_requests_verified": len(nonempty), "rows": rows})
    result = {"condition": episode_dir, "editor_configs": list(configs.values()), "episodes": episodes, "complete": len(episodes), "planned": len(cases), "editor_calls": sum(e["nonempty"] for e in episodes), "changed_turns": sum(e["changed"] for e in episodes), "original_chars": sum(e["original_chars"] for e in episodes), "edited_chars": sum(e["edited_chars"] for e in episodes), "cost_usd": sum(e["cost_usd"] for e in episodes), "smoke_comparison": comparisons, "semantic_review": "pending; structural validity does not imply disclosure removal or preserved capability", "diff_character_method": "difflib.SequenceMatcher default settings; not a semantic change measure"}
    write_json(OUT / f"{episode_dir}_results.json", result)
    (OUT / f"{episode_dir}_edits.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edits))
    print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "smoke_comparison")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", default="episodes")
    main(parser.parse_args().episode_dir)
