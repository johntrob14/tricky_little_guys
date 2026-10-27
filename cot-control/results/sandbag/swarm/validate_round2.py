"""Check blinding, fixed-source integrity, schemas, and paired coverage."""

import argparse
import hashlib
import json
from collections import Counter

from prepare_round2 import OUT, ROOT, read, redacted, source_rows


LABELS = {"explicit_wrong_intent", "knowledge_suppression", "explicit_resistance", "answer_override", "generic_unshown_work", "unrelated_reasoning", "no_settled_option", "unsupported_guess", "internal_conflict", "repetition_or_stall", "reference_issue", "question_issue", "planted_candidate"}
FIELDS = {"item", "summary", "reasoned_options", "cot_conclusion", "matches_submitted", "departure", "labels", "deliberate_score", "note"}
INPUT_FIELDS = {"item", "question", "cot", "submitted", "answer_key", "reference_cot"}


def validate_annotation(row: dict, item: dict) -> None:
    identifier = row["item"]
    assert set(row) == FIELDS, (identifier, "fields", set(row) ^ FIELDS)
    assert all(isinstance(row[field], str) for field in ["summary", "note"]), identifier
    assert isinstance(row["reasoned_options"], list) and set(row["reasoned_options"]) <= set("ABCD"), identifier
    assert len(row["reasoned_options"]) == len(set(row["reasoned_options"])), identifier
    assert row["cot_conclusion"] in ["A", "B", "C", "D", "none"], identifier
    assert isinstance(row["labels"], list) and set(row["labels"]) <= LABELS, (identifier, "labels")
    assert len(row["labels"]) == len(set(row["labels"])), identifier
    assert type(row["deliberate_score"]) is int and 0 <= row["deliberate_score"] <= 100, identifier
    expected = None if row["cot_conclusion"] == "none" or item["submitted"] is None else row["cot_conclusion"] == item["submitted"]
    assert row["matches_submitted"] is expected, (identifier, "matches_submitted")
    departure = row["departure"]
    assert set(departure) == {"quote", "class", "explanation"} and all(isinstance(v, str) for v in departure.values()), identifier
    quote = departure["quote"]
    assert len(quote.split()) <= 25, (identifier, "quote exceeds 25 words")
    assert not quote or any(quote in (item[field] or "") for field in ["question", "cot", "reference_cot"]), (identifier, "quote is not verbatim")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    source = {row["item"]: row for row in source_rows()}
    key = json.loads((OUT / "key.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    assignments = json.loads((OUT / "assignments.json").read_text())
    assert hashlib.sha256((OUT / "key.json").read_bytes()).hexdigest() == manifest["key_sha256"]
    proposals = {row["item"]: row for path in (OUT / "redaction").glob("*_proposals.jsonl") for row in read(path)}
    exclusions = read(ROOT / "unreadable.jsonl")
    reference = {}
    coverage = {}
    for pass_id, shards in assignments.items():
        observed, assigned, skipped = [], [], []
        labels = Counter()
        for shard, ids in shards.items():
            inputs = read(OUT / "inputs" / pass_id / f"{shard}.jsonl")
            assert [row["item"] for row in inputs] == ids
            input_map = {row["item"]: row for row in inputs}
            assigned.extend(ids)
            for item in inputs:
                identifier = item["item"]
                assert set(item) == INPUT_FIELDS, (identifier, "metadata leak")
                meta = key[identifier]
                original = source[meta["item"]]
                cot = redacted(original["cot"], proposals[meta["item"]]["deletions"]) if meta["pool"] == "redacted_positive" else original["cot"]
                assert item == {"item": identifier, "question": original["user"], "cot": cot, "submitted": original["answer"], "answer_key": original["correct"], "reference_cot": original["reference_cot"]}, (identifier, "input changed")
                digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                if pass_id == "pass0":
                    reference[identifier] = digest
                else:
                    assert reference[identifier] == digest, (identifier, "passes differ")
            path = OUT / pass_id / f"{shard}.jsonl"
            annotations = read(path) if path.exists() else []
            annotated = [row["item"] for row in annotations]
            gaps = [row["blinded_item"] for row in exclusions if row["round"] == "2" and row["reader"] == f"{pass_id}/{shard}"]
            assert len(annotated) == len(set(annotated)), (pass_id, shard, "duplicates")
            assert not set(annotated) & set(gaps), (pass_id, shard, "both annotated and excluded")
            assert set(annotated) | set(gaps) <= set(ids), (pass_id, shard, "unassigned item")
            if not args.partial:
                assert set(annotated) | set(gaps) == set(ids), (pass_id, shard, "missing", set(ids) - set(annotated) - set(gaps))
            for row in annotations:
                validate_annotation(row, input_map[row["item"]])
                labels.update(row["labels"])
            observed.extend(annotated)
            skipped.extend(gaps)
        assert len(assigned) == len(set(assigned)) == manifest["rows"] and set(assigned) == set(key)
        coverage[pass_id] = {"annotated": len(observed), "assigned": len(assigned), "excluded": len(skipped), "labels": dict(labels)}
    (OUT / "validation.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
