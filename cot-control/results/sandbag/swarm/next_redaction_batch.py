"""Continue control preparation in the fixed candidate order."""

import argparse
import json

from prepare_round2 import OUT, eligible_rows, read, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("batch", type=int)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    path = OUT / "inputs/redaction" / f"{args.model}_{args.batch:02d}.jsonl"
    assert not path.exists(), path
    source = {row["item"]: row for row in eligible_rows()}
    queue = json.loads((OUT / "candidate_queue.json").read_text())[args.model]
    attempted = {row["item"] for path in (OUT / "redaction").glob("*_proposals.jsonl") for row in read(path)}
    pending = [item for item in queue if item in source and item not in attempted]
    assert pending and args.count > 0
    selected = pending[:args.count]
    write_rows(path, [{"item": item, "cot": source[item]["cot"]} for item in selected])
    print(json.dumps({"path": str(path.relative_to(OUT)), "selected": len(selected), "remaining_eligible_unattempted": len(pending)}))


if __name__ == "__main__":
    main()
