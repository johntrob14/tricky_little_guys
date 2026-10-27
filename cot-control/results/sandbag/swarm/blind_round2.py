"""Freeze checked controls, hide source metadata, and shuffle two independent passes."""

import hashlib
import json
import random
from collections import Counter

from prepare_round1 import MODELS, category
from prepare_round2 import OUT, ROOT, SEED, eligible_rows, read, redacted, write_rows


def main() -> None:
    assert not (OUT / "key.json").exists(), "Blinded assignments already frozen"
    source = {row["item"]: row for row in eligible_rows()}
    queue = json.loads((OUT / "candidate_queue.json").read_text())
    proposals = [row for path in sorted((OUT / "redaction").glob("*_proposals.jsonl")) for row in read(path)]
    checks = [row for path in sorted((OUT / "redaction").glob("*_checks.jsonl")) for row in read(path)]
    assert len(proposals) == len({row["item"] for row in proposals})
    assert len(checks) == len({row["item"] for row in checks})
    proposed = {row["item"]: row for row in proposals}
    stopped = {row["item"] for row in read(ROOT / "unreadable.jsonl") if row["round"] == "2-redaction"}
    checked = {row["item"] for row in checks}
    accepted = set()
    for check in checks:
        assert type(check["accept"]) is bool and proposed[check["item"]]["suitable"]
        if check["accept"]:
            accepted.add(check["item"])
    selected = []
    coverage = {}
    for model in MODELS:
        candidates = [item for item in queue[model] if item in accepted and item in source]
        assert len(candidates) >= 10, (model, "accepted controls", len(candidates), "target", 10)
        prefix = queue[model][:queue[model].index(candidates[9]) + 1]
        unresolved = [item for item in prefix if item in source and (item not in proposed or proposed[item]["suitable"] and item not in checked)]
        assert not unresolved, (model, "earlier candidates unresolved", unresolved)
        selected.extend(candidates[:10])
        coverage[model] = {"candidate_pool": len(queue[model]), "currently_eligible_pool": sum(item in source for item in queue[model]), "attempted": sum(item in proposed or item in stopped for item in queue[model]), "completed_redactions": sum(item in proposed for item in queue[model]), "stopped": sum(item in stopped for item in queue[model]), "checked": sum(item in checked for item in queue[model]), "accepted": len(candidates), "selected": 10}
    core = [row for row in source.values() if category(row) in ["unstated_flip", "cotctl_flip"] or row["kind"] == "resample_error"]
    assert 0 < len(core) <= 622
    rows = core + [source[item] for item in selected]
    assert len(rows) == len({row["item"] for row in rows}) == len(core) + 40
    rng = random.Random(SEED + 1)
    rng.shuffle(rows)
    key = {}
    blinded = []
    for row in rows:
        identifier = f"b{rng.getrandbits(64):016x}"
        assert identifier not in key
        cot = redacted(row["cot"], proposed[row["item"]]["deletions"]) if row["item"] in selected else row["cot"]
        pool = "redacted_positive" if row["item"] in selected else category(row)
        if row["kind"] == "resample_error":
            pool = "natural_parsed" if row["answer"] is not None else "natural_unparsed"
        item = {"item": identifier, "question": row["user"], "cot": cot, "submitted": row["answer"], "answer_key": row["correct"], "reference_cot": row["reference_cot"]}
        blinded.append(item)
        key[identifier] = {field: row[field] for field in ["item", "model", "variant", "arm", "qid", "kind", "answer", "correct"]}
        key[identifier].update({"pool": pool, "cot_sha256": hashlib.sha256(cot.encode()).hexdigest(), "reference_sha256": hashlib.sha256((row["reference_cot"] or "").encode()).hexdigest()})
    assignments = {}
    for pass_id in range(2):
        shuffled = list(blinded)
        random.Random(SEED + 100 + pass_id).shuffle(shuffled)
        assignments[f"pass{pass_id}"] = {}
        for offset in range(0, len(shuffled), 20):
            name = f"shard{offset // 20:02d}"
            shard = shuffled[offset:offset + 20]
            write_rows(OUT / f"inputs/pass{pass_id}" / f"{name}.jsonl", shard)
            assignments[f"pass{pass_id}"][name] = [row["item"] for row in shard]
    (OUT / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    (OUT / "assignments.json").write_text(json.dumps(assignments, indent=2) + "\n")
    (OUT / "redaction/selection.json").write_text(json.dumps({"coverage": coverage, "selected": selected}, indent=2) + "\n")
    manifest = {"source_sha256": json.loads((OUT / "approval.json").read_text())["source_sha256"], "seed": SEED, "rows": len(rows), "max_rows_per_shard": 20, "key_sha256": hashlib.sha256((OUT / "key.json").read_bytes()).hexdigest(), "pools": dict(Counter(f"{row['model']}/{row['pool']}" for row in key.values()))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "passes": {name: len(shards) for name, shards in assignments.items()}, "positive_coverage": coverage, "pools": manifest["pools"]}, indent=2))


if __name__ == "__main__":
    main()
