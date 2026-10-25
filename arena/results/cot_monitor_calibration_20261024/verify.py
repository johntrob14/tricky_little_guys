"""Check frozen inputs, request rendering, score parsing and archived coverage without model calls."""

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ARENA = OUT.parents[1]
sys.path.insert(0, str(ARENA))

from rl.monitor.formatter import format_cot_prompt, load_template
from rl.monitor.monitors import get
from rl.monitor.parse import parse_tagged


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as file:
        return [json.loads(line) for line in file]


def main() -> None:
    frozen = json.loads((OUT / "freeze.json").read_text())
    for path, expected in frozen["sha256"].items():
        assert digest((ARENA / path).read_bytes()) == expected, path
    for split in ["heldout", "recall", "confirmation"]:
        expected = json.loads((OUT / f"{split}_labels_freeze.json").read_text())["sha256"]
        assert digest((OUT / f"{split}_labels.json").read_bytes()) == expected, split
    data = {r["id"]: r for r in rows(OUT / "pool.jsonl.gz") + rows(OUT / "controls_extra.jsonl.gz") + rows(OUT / "confirmation.jsonl")}
    development = {r["uuid"] for r in rows(OUT / "dev.jsonl")}
    assert development.isdisjoint(r["uuid"] for r in rows(OUT / "validation.jsonl"))
    assert {r["uuid"] for r in rows(OUT / "pool.jsonl.gz")}.isdisjoint(r["uuid"] for r in rows(OUT / "confirmation.jsonl"))
    templates = {name: load_template(OUT / f"prompts/{name}.txt", ("task_description", "cot")) for name in ["baseline", "evidence"]}
    scores = {}
    for path in OUT.glob("*.scores.jsonl"):
        for row in rows(path):
            key = (path.name.split(".")[0], row["id"], row["arm"], row["repeat"])
            assert key not in scores
            scores[key] = row
    checked, parsed, response_ids = set(), 0, set()
    manifest = json.loads((OUT / "raw/manifest.json").read_text())
    for name, info in manifest.items():
        path = OUT / "raw" / name
        assert digest(path.read_bytes()) == info["sha256"]
        archived = rows(path)
        assert len(archived) == info["responses"]
        for row in archived:
            request = row["request"]
            record = data[request["id"]]
            variant = "evidence" if request["arm"].startswith("evidence") else "baseline"
            prompt = format_cot_prompt(templates[variant], record["task_description"], record["cot"], [])
            if request["arm"].startswith("context"):
                prompt = (OUT / "prompts/context.txt").read_text() + prompt
            assert prompt == request["prompt"]
            assert digest(prompt.encode()) == request["prompt_sha256"]
            assert request["mode"] == "cot_only" and request["main_task_only"] and request["input_channels"] == ["reasoning"]
            baseline = get("controlarena_cot_only").format(record["task_description"], record["cot"], [])
            assert digest(baseline.encode()) == record["baseline_prompt_sha256"]
            body = row["response"]
            assert body["id"] not in response_ids
            response_ids.add(body["id"])
            text = body["choices"][0]["message"]["content"] or ""
            score, explanation = parse_tagged(text) if row["finish_reason"] == "stop" else (None, None)
            assert (score, explanation) == (row["score"], row["explanation"])
            key = (name.split(".")[0], request["id"], request["arm"], request["repeat"])
            assert key not in checked
            checked.add(key)
            compact = scores[key]
            assert compact["score"] == score and compact["prompt_sha256"] == request["prompt_sha256"] and compact["config"] == request["config"]
            parsed += score is not None
    failed = [scores[key] for key in scores.keys() - checked]
    assert all(row.get("request_failed") and row["score"] is None for row in failed)
    all_labels = {}
    for name in ["dev", "heldout", "recall", "confirmation", "control_flag"]:
        all_labels.update(json.loads((OUT / f"{name}_labels.json").read_text()))
    flags = {row["id"] for key, row in scores.items() if key[0].startswith("controls") and row["score"] is not None and row["score"] > 5}
    assert flags <= all_labels.keys()
    result = {"frozen_hashes_match": True, "episode_splits_disjoint": True, "exact_prompt_and_parser_checks": {"passed": len(checked), "total_saved": len(response_ids)}, "planned_requests": len(scores), "parsed": parsed, "request_failures_without_response": len(failed), "reviewed_control_alert_ids": {"reviewed": len(flags), "total": len(flags)}, "manual_records": len(all_labels), "manual_label_counts": dict(Counter(v[0] for v in all_labels.values()))}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
