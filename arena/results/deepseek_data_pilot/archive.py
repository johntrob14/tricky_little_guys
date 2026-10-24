"""Retain raw editor/judge responses and count billed generations once, including failures and reused responses."""

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from editing import digest, parse_reply, reasoning_blocks, write_json

OUT = Path(__file__).resolve().parent
PATTERNS = ["working/editor/*.json", "working/episodes*/*/*.response.json", "working/no_thinking_probe/*.json", "working/hybrid_retries/*.json", "working/editor_failures/*.json", "working/monitor_*/*.json"]


def main() -> None:
    paths = sorted({p for pattern in PATTERNS for p in OUT.glob(pattern)})
    originals = {p.name.split(".")[0]: [m for m in json.loads(p.read_text()) if m["role"] == "assistant"] for p in (OUT / "working").glob("*.original.json")}
    unique, stages, failures, entries = {}, defaultdict(list), [], []
    for path in paths:
        record = json.loads(path.read_text())
        if "response" not in record:
            continue
        body = record["response"]
        rid = body["id"]
        if rid in unique:
            assert digest(unique[rid]) == digest(body)
        unique[rid] = body
        relative = str(path.relative_to(OUT))
        entries.append({"path": relative, "record": record})
        stage = path.relative_to(OUT).parts[1]
        valid, error = None, None
        if "config" in record:
            message = originals[record["uuid"]][record["turn"]]
            before = reasoning_blocks([message])[0]["reasoning"]
            try:
                parse_reply(body, before)
                valid = True
            except (AssertionError, ValueError, TypeError) as exc:
                valid, error = False, f"{type(exc).__name__}: {exc}"
                failures.append({"path": relative, "uuid": record["uuid"], "turn": record["turn"], "error": error, "response_id": rid, "finish_reason": body["choices"][0]["finish_reason"], "cost_usd": body["usage"]["cost"]})
        stages[stage].append({"response_id": rid, "valid_editor_reply": valid})
    archive = OUT / "responses.jsonl.gz"
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
        for entry in entries:
            stream.write((json.dumps(entry, ensure_ascii=False) + "\n").encode())
    summary = {"response_files": len(entries), "unique_billed_generations": len(unique), "reused_response_files": len(entries) - len(unique), "known_cost_usd": sum(b["usage"]["cost"] for b in unique.values()), "unpriced_rejected_requests": 1, "unpriced_rejection_record": "luna_probe.json", "stages": {name: {"retained_responses": len(rows), "valid_editor_replies": sum(r["valid_editor_reply"] is True for r in rows), "invalid_editor_replies": sum(r["valid_editor_reply"] is False for r in rows)} for name, rows in stages.items()}, "editor_failures": failures, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "archive_bytes": archive.stat().st_size, "archive_note": "Model outputs may quote synthetic benchmark secrets; this is a raw research-data archive. No API credentials or authorization headers are included."}
    write_json(OUT / "attempts.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("stages", "editor_failures")}))


if __name__ == "__main__":
    main()
