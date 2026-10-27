"""Display one unchanged blinded item in bounded slices and persist its current ID."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("shard")
parser.add_argument("index", type=int)
parser.add_argument("offset", type=int, nargs="?", default=0)
args = parser.parse_args()
root = Path("/tmp/wmdp_readers_20261027")
rows = [json.loads(line) for line in (root / "inputs" / f"{args.shard}.jsonl").open()]
row = rows[args.index]
exclusions = root / "runtime_exclusions.json"
if exclusions.exists() and row["item"] in json.loads(exclusions.read_text()):
    raise SystemExit(f"Item {row['item']} is excluded and already recorded unreadable; continue with another assigned item.")
text = json.dumps(row, indent=2, ensure_ascii=False)
assert 0 <= args.offset < len(text)
end = min(args.offset + 14000, len(text))
(root / "outputs" / f"{args.shard}_current.json").write_text(json.dumps({"item": row["item"], "index": args.index, "offset": args.offset, "end": end, "total": len(text)}) + "\n")
with (root / "outputs" / f"{args.shard}_reads.jsonl").open("a") as file:
    file.write(json.dumps({"item": row["item"], "offset": args.offset, "end": end, "total": len(text)}) + "\n")
print(f"item={row['item']} index={args.index}/{len(rows)} characters={args.offset}:{end}/{len(text)}")
print(text[args.offset:end])
