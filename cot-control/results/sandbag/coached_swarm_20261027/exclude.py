"""Record an identified reader stop and omit unattempted instances of its question."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(__file__).resolve().parent
work = Path("/tmp/wmdp_readers_20261027")
trigger = sys.argv[1]
key = json.loads((root / "key.json").read_text())
qid = key[trigger]["qid"]
completed = {json.loads(line)["item"] for path in (work / "outputs").glob("shard[0-9][0-9].jsonl") for line in path.open()}
assert trigger not in completed, "Stopped item already has an annotation; inspect stop provenance"
path = work / "runtime_exclusions.json"
blocked = json.loads(path.read_text()) if path.exists() else {}
new = []
for item, meta in key.items():
    if meta["qid"] != qid or item in completed or item in blocked:
        continue
    reason = f"Question excluded after a reader stop associated with latest marker {trigger}; stopped content is not retried."
    blocked[item] = reason
    shard = f"shard{meta['shard']:02d}"
    with (work / "outputs" / f"{shard}_unreadable.jsonl").open("a") as file:
        file.write(json.dumps({"item": item, "reason": reason, "status": "safety_stop" if item == trigger else "excluded_question"}) + "\n")
    new.append(item)
path.write_text(json.dumps(blocked, indent=2) + "\n")
(root / "runtime_exclusions.json").write_text(json.dumps({"excluded_items": blocked}, indent=2) + "\n")
with (root / "stop_events.jsonl").open("a") as file:
    file.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), "trigger": trigger, "question": qid, "associated_item_basis": "latest_reader_marker", "excluded": new}) + "\n")
print(json.dumps({"trigger": trigger, "question": qid, "excluded": new}))
