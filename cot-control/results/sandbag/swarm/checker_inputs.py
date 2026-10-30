"""Reconstruct proposed sentence deletions for an independent checker."""

import argparse
import hashlib
import json

from prepare_round2 import OUT, read, redacted, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch")
    args = parser.parse_args()
    source = {row["item"]: row for row in read(OUT / "inputs/redaction" / f"{args.batch}.jsonl")}
    proposals = read(OUT / "redaction" / f"{args.batch}_proposals.jsonl")
    assert len(proposals) == len({row["item"] for row in proposals})
    excluded_questions = set(json.loads((OUT / "question_exclusions.json").read_text()))
    inputs = []
    records = []
    for row in proposals:
        assert type(row["suitable"]) is bool
        cot = source[row["item"]]["cot"]
        edited = redacted(cot, row["deletions"])
        excluded = row["item"].rsplit("/", 1)[1] in excluded_questions
        if row["suitable"] and not excluded:
            assert row["deletions"] and edited.strip(), row["item"]
            inputs.append({"item": row["item"], "cot": cot, "redacted_cot": edited, "deletions": row["deletions"]})
        records.append({"item": row["item"], "suitable": row["suitable"], "excluded_question": excluded, "cot_sha256": hashlib.sha256(cot.encode()).hexdigest(), "redacted_sha256": hashlib.sha256(edited.encode()).hexdigest(), "original_chars": len(cot), "retained_chars": len(edited)})
    write_rows(OUT / "inputs/redaction" / f"{args.batch}_check.jsonl", inputs)
    write_rows(OUT / "redaction" / f"{args.batch}_reconstruction.jsonl", records)
    print(json.dumps({"batch": args.batch, "proposed_suitable": len(inputs), "attempted": len(proposals), "assigned": len(source)}))


if __name__ == "__main__":
    main()
