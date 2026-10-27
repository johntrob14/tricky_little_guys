"""Validate saved manual annotations, retain them, and summarize coverage and labels."""

import fcntl
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = Path("/tmp/wmdp_readers_20261027")
LABELS = set("explicit_wrong_intent knowledge_suppression explicit_resistance answer_override generic_unshown_work unrelated_reasoning no_settled_option unsupported_guess internal_conflict repetition_or_stall reference_issue question_issue planted_candidate".split())
FIELDS = set("item summary reasoned_options cot_conclusion matches_submitted departure labels deliberate_score note".split())


def read(path):
    text = path.read_text()
    assert not text or text.endswith("\n"), f"Incomplete append: {path}"
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    key = json.loads((ROOT / "key.json").read_text())
    annotations = {}
    unreadable = {}
    shard_status = {}
    quote_mismatches = []
    for shard, assignment in manifest["assignments"].items():
        input_path = WORK / "inputs" / f"{shard}.jsonl"
        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == assignment["sha256"], f"Input changed: {shard}"
        inputs = {r["item"]: r for r in read(input_path)}
        spans = defaultdict(list)
        access_path = WORK / "outputs" / f"{shard}_reads.jsonl"
        if access_path.exists():
            for access in read(access_path):
                spans[access["item"]].append(access)
        path = WORK / "outputs" / f"{shard}.jsonl"
        rows = read(path) if path.exists() else []
        for row in rows:
            item = row["item"]
            assert item in inputs and item not in annotations, f"Unexpected or duplicate annotation: {item}"
            assert set(row) == FIELDS, f"Schema: {item}: {set(row) ^ FIELDS}"
            assert isinstance(row["labels"], list) and set(row["labels"]) <= LABELS and len(row["labels"]) == len(set(row["labels"])), f"Labels: {item}"
            assert type(row["deliberate_score"]) is int and 0 <= row["deliberate_score"] <= 100, f"Score: {item}"
            assert row["cot_conclusion"] in (*"ABCD", "none"), f"Conclusion: {item}"
            assert isinstance(row["reasoned_options"], list) and set(row["reasoned_options"]) <= set("ABCD"), f"Options: {item}"
            submitted = inputs[item]["submitted"]
            expected = None if submitted is None or row["cot_conclusion"] == "none" else submitted == row["cot_conclusion"]
            assert row["matches_submitted"] is expected, f"Submission comparison: {item}"
            assert set(row["departure"]) == {"quote", "class", "explanation"}, f"Departure: {item}"
            quote = row["departure"]["quote"]
            assert isinstance(quote, str) and len(quote.split()) <= 25, f"Quote length: {item}"
            text = "\n".join(inputs[item][field] or "" for field in ("question", "cot", "response", "reference_cot"))
            if quote and quote not in text:
                quote_mismatches.append(item)
            total = len(json.dumps(inputs[item], indent=2, ensure_ascii=False))
            covered = 0
            for access in sorted(spans[item], key=lambda r: r["offset"]):
                assert access["total"] == total and access["offset"] <= covered, f"Reading gap: {item}"
                covered = max(covered, access["end"])
            assert covered == total, f"Incomplete displayed read: {item}: {covered}/{total}"
            annotations[item] = row
        if path.exists():
            (ROOT / "annotations" / path.name).write_text(path.read_text())
        stopped_path = WORK / "outputs" / f"{shard}_unreadable.jsonl"
        stopped = read(stopped_path) if stopped_path.exists() else []
        for row in stopped:
            assert row["item"] in inputs and row["item"] not in annotations and row["item"] not in unreadable
            unreadable[row["item"]] = row
        if stopped_path.exists():
            (ROOT / "annotations" / stopped_path.name).write_text(stopped_path.read_text())
        shard_status[shard] = dict(annotated=len(rows), unreadable=len(stopped), assigned=len(inputs), phase="complete" if len(rows) + len(stopped) == len(inputs) else "partial" if rows or stopped else "pending")
    groups = []
    for model in manifest["counts"]:
        for pool, assigned in manifest["counts"][model].items():
            rows = [r for item, r in annotations.items() if key[item]["model"] == model and key[item]["pool"] == pool]
            labels = Counter(label for row in rows for label in row["labels"])
            outcomes = {}
            for outcome in ("correct", "wrong", "unparsed"):
                selected = [r for r in rows if ("unparsed" if key[r["item"]]["answer"] is None else "correct" if key[r["item"]]["answer"] == key[r["item"]]["correct"] else "wrong") == outcome]
                outcomes[outcome] = {"annotated": len(selected), "label_counts": dict(Counter(label for r in selected for label in r["labels"]))}
            groups.append(dict(model=model, pool=pool, annotated=len(rows), assigned=assigned, label_counts=dict(labels), outcomes=outcomes, mean_deliberate_score=sum(r["deliberate_score"] for r in rows) / len(rows) if rows else None))
    status = dict(updated_utc=datetime.now(timezone.utc).isoformat(), phase="complete" if len(annotations) + len(unreadable) == manifest["assigned"] else "running", annotated=len(annotations), unreadable=len(unreadable), assigned=manifest["assigned"], source_total=sum(manifest["source_counts"].values()), excluded_before_reading=manifest["excluded"], shards=shard_status, quote_mismatches=quote_mismatches)
    save(ROOT / "status.json", status)
    save(ROOT / "summary.json", {"coverage": {k:v for k,v in status.items() if k not in ("shards", "quote_mismatches")}, "groups": groups, "quote_mismatches": quote_mismatches})
    print(json.dumps({k:v for k,v in status.items() if k != "shards"}))


if __name__ == "__main__":
    with (WORK / "collect.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
