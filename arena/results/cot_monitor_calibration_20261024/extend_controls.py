"""Prospective confirmation of the frozen candidate on additional saved base-model controls. Use /usr/bin/python 3.14."""

import gzip
import json
import zipfile
from collections import Counter
from prepare import ARENA, OUT, digest, rows_from, write_json

RUNS = ["ctl-easy-big", "ctl-medium-big", "ctl-qwen-easy-or", "gptoss-ctl-easy2", "glm53-ctl-easy", "inksmtk-ctl-easy", "ds41-ctl-easy", "gptoss-ctl-all", "gptoss-ctl-easy", "ctl-qwen-easy", "ctl-qwen-medium", "ctl-qwen-hard"]


def signature(rows: list[dict]) -> str:
    return digest(json.dumps([(r["task_description"], r["cot"]) for r in rows], ensure_ascii=False))


def main() -> None:
    with gzip.open(OUT / "pool.jsonl.gz", "rt") as file:
        pool = [json.loads(line) for line in file]
    episodes = {}
    for row in pool:
        if row["cohort"] == "control":
            episodes.setdefault(row["uuid"], []).append(row)
    seen = {signature(rows) for rows in episodes.values()}
    selected, inventory, counts = [], [], Counter()
    for run in RUNS:
        for path in sorted((ARENA / "logs" / run).glob("*.eval")):
            with zipfile.ZipFile(path) as archive:
                assert "header.json" in archive.namelist(), path
                for name in archive.namelist():
                    if not (name.startswith("samples/") and name.endswith(".json")):
                        continue
                    sample = json.loads(archive.read(name))
                    meta = sample["metadata"]
                    assert meta["preset"] == "control" and meta["ckpt"] == "base" and meta["prefill"] == "none", (path, meta["preset"], meta["ckpt"], meta["prefill"])
                    source = {"uuid": sample["uuid"], "model": meta["model"], "task": meta["task"], "log": str(path.relative_to(ARENA)), "sample": str(sample["id"]), "epoch": sample["epoch"], "cohort": "control_extra", "run": run}
                    rows = rows_from(sample["messages"], source)
                    content_hash = signature(rows)
                    reason = "sample_error" if sample.get("error") else "empty_reasoning" if not rows else "duplicate" if content_hash in seen else "selected"
                    counts[reason] += 1
                    inventory.append({**source, "selection": reason, "nonempty_turns": len(rows), "content_sha256": content_hash})
                    if reason == "selected":
                        selected.extend(rows)
                        seen.add(content_hash)
    with gzip.open(OUT / "controls_extra.jsonl.gz", "wt") as file:
        file.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    write_json("controls_extra_selection.json", {"runs": RUNS, "counts": dict(counts), "turns": len(selected), "selection": "All completed non-error base/control/none episodes from the listed existing runs; skip empty reasoning and duplicate complete main-task+reasoning sequences. No score or content filter. Selected after initial candidate comparison; prompt remains frozen.", "episodes": inventory})
    print(json.dumps({"counts": dict(counts), "turns": len(selected)}))


if __name__ == "__main__":
    main()
