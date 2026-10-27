"""Check reader coverage and summarize observable answer relationships."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_round1 import category


ROOT = Path(__file__).resolve().parent
REQUIRED = {"item", "summary", "cot_conclusion", "matches_submitted", "departure", "notable", "tags", "interest"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    manifest = json.loads((ROOT / "round1/sample.json").read_text())
    assert hashlib.sha256((ROOT / "items.jsonl").read_bytes()).hexdigest() == manifest["source_sha256"], "Source dataset changed during Round 1"
    source = {row["item"]: row for row in read(ROOT / "items.jsonl")}
    expected = {
        "natural_base": manifest["round1a"]["base"],
        "natural_mt1b": manifest["round1a"]["mt1b-s150"],
        "natural_grpo_mt2e": manifest["round1a"]["grpo3-s100"] + manifest["round1a"]["manytypes2e-s170"],
        **manifest["round1b"],
    }
    groups = defaultdict(list)
    coverage = {}
    unreadable_path = ROOT / "unreadable.jsonl"
    unreadable = read(unreadable_path) if unreadable_path.exists() else []
    for reader, ids in expected.items():
        annotations = read(ROOT / "round1/readers" / f"{reader}.jsonl")
        observed = [row["item"] for row in annotations]
        skipped = {row["item"] for row in unreadable if row["reader"] == reader}
        assert len(observed) == len(set(observed)), (reader, "duplicate IDs")
        assert not set(observed) & skipped, (reader, "both annotated and unreadable")
        assert set(observed) | skipped == set(ids), (reader, "missing", set(ids) - set(observed) - skipped, "unexpected", (set(observed) | skipped) - set(ids))
        coverage[reader] = {"annotated": len(observed), "unreadable": len(skipped), "assigned": len(ids)}
        for row in annotations:
            assert REQUIRED <= row.keys(), (reader, row["item"], "missing fields")
            assert not {"cot", "reference_cot", "system", "user", "response"} & row.keys(), (reader, "raw input field in output")
            assert row["cot_conclusion"] in ["A", "B", "C", "D", "none"], (reader, row["item"], "invalid conclusion")
            assert isinstance(row["matches_submitted"], bool), (reader, row["item"], "invalid match flag")
            assert isinstance(row["tags"], list) and all(isinstance(tag, str) for tag in row["tags"]), (reader, "invalid tags")
            assert type(row["interest"]) is int and 0 <= row["interest"] <= 3, (reader, "invalid interest")
            item = source[row["item"]]
            conclusion = row["cot_conclusion"]
            match = conclusion != "none" and conclusion == item["answer"]
            assert row["matches_submitted"] == match, (reader, row["item"], "inconsistent match flag")
            if conclusion == "none":
                relation = "no_conclusion"
            elif item["answer"] is None:
                relation = "conclusion_without_parsed_answer"
            else:
                relation = "aligned" if match else "conclusion_disagreement"
            panel = "natural" if reader.startswith("natural_") else "stratified"
            kind = item["kind"] if panel == "natural" else category(item)
            key = "/".join([panel, item["model"], kind, "parsed" if item["answer"] is not None else "unparsed"])
            groups[key].append({"item": row["item"], "reader": reader, "relation": relation, "tags": row["tags"], "interest": row["interest"]})
    summary = {"coverage": coverage, "groups": {}}
    for key, rows in sorted(groups.items()):
        summary["groups"][key] = {"n": len(rows), "relations": dict(Counter(row["relation"] for row in rows)), "tags": dict(Counter(tag for row in rows for tag in row["tags"])), "items": rows}
    (ROOT / "round1/analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"coverage": coverage, "groups": {key: {"n": value["n"], "relations": value["relations"]} for key, value in summary["groups"].items()}}, indent=2))


if __name__ == "__main__":
    main()
