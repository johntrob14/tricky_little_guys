"""Freeze coached traces and existing natural/style-error controls for blinded reads."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
OLD = SOURCE / "swarm"
WORK = Path("/tmp/wmdp_readers_20261027")
MODELS = ("base", "grpo3-s100", "mt1b-s150", "manytypes2e-s170")
SEED = 2026102703


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    assert not (ROOT / "manifest.json").exists(), "This frozen pool already exists"
    WORK.mkdir(exist_ok=False)
    (WORK / "inputs").mkdir()
    (WORK / "outputs").mkdir()
    (ROOT / "annotations").mkdir()
    source_path = OLD / "items.jsonl"
    old_manifest = json.loads((OLD / "round1/sample.json").read_text())
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == old_manifest["source_sha256"]
    source = read(source_path)
    stopped = read(OLD / "unreadable.jsonl")
    blocked_items = {row["item"] for row in stopped}
    excluded_questions = {int(qid) for qid in json.loads((OLD / "round2/question_exclusions.json").read_text())}
    excluded_questions |= {int(row["item"].split("/")[-1]) for row in stopped if row.get("status") == "safety_stop" or "during this item" in row["reason"]}
    safe = [row for row in source if row["qid"] not in excluded_questions and row["item"] not in blocked_items]
    reference = {}
    reference_id = {}
    for row in safe:
        key = (row["model"], row["qid"])
        if row["arm"] == "honest" and row["answer"] == row["correct"] and key not in reference:
            reference[key] = row["cot"]
            reference_id[key] = row["item"]
    honest = {(r["model"], r["qid"]): r for r in source if r["variant"] == "default" and r["arm"] == "honest"}
    pool = []
    hashes = {str(source_path.relative_to(SOURCE)): old_manifest["source_sha256"]}
    excluded = []
    for model in MODELS:
        path = SOURCE / f"{model}-sysprompt-nohint-should/rows.jsonl"
        hashes[str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [r for r in read(path) if r["arm"] == "coached"]
        assert len(rows) == len({r["qid"] for r in rows}) == 150
        for row in rows:
            item = f"{model}/should/coached/{row['qid']}"
            if row["qid"] in excluded_questions:
                excluded.append({"item": item, "pool": "coached", "reason": "Question previously associated with an identified reader safety stop"})
                continue
            h = honest[model, row["qid"]]
            kind = "unparsed" if row["answer"] is None else "not_eligible" if h["answer"] != h["correct"] else "held" if row["answer"] == row["correct"] else "flip"
            pool.append(dict(item=item, model=model, qid=row["qid"], pool="coached", kind=kind, correct=row["correct"], answer=row["answer"], finish=row["finish"], user=row["user"], cot=row["reasoning"], response=row["response"]))
    controls = [row for row in source if row["kind"] == "resample_error" or row["arm"] == "cotctl" and row["kind"] == "flip"]
    for row in controls:
        category = "natural_error" if row["kind"] == "resample_error" else "style_error"
        if row["qid"] in excluded_questions or row["item"] in blocked_items:
            excluded.append({"item": row["item"], "pool": category, "reason": "Inherited item/question exclusion"})
        else:
            pool.append(dict(row, pool=category))
    rng = random.Random(SEED)
    rng.shuffle(pool)
    prompt = (OLD / "round2/reader_prompt.md").read_bytes()
    approved = json.loads((OLD / "round2/approval.json").read_text())
    assert hashlib.sha256(prompt).hexdigest() == approved["prompt_sha256"]["reader"]
    (ROOT / "reader_prompt.md").write_bytes(prompt)
    (WORK / "reader_prompt.md").write_bytes(prompt)
    key = {}
    shards = [[]]
    chars = 0
    for number, row in enumerate(pool):
        item = f"C{number:04d}"
        identity = (row["model"], row["qid"])
        record = dict(item=item, question=row["user"], cot=row["cot"], response=row["response"], submitted=row["answer"], answer_key=row["correct"], reference_cot=reference.get(identity))
        text = json.dumps(record, ensure_ascii=False)
        if shards[-1] and (chars + len(text) > 180000 or len(shards[-1]) >= 24):
            shards.append([])
            chars = 0
        shards[-1].append(record)
        chars += len(text)
        key[item] = {k: row[k] for k in ("item", "model", "qid", "pool", "kind", "correct", "answer", "finish")}
        key[item]["reference_item"] = reference_id.get(identity)
        key[item]["shard"] = len(shards) - 1
    assignments = {}
    for number, rows in enumerate(shards):
        name = f"shard{number:02d}"
        path = WORK / "inputs" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        assignments[name] = {"items": [r["item"] for r in rows], "characters": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    counts = Counter((row["model"], row["pool"]) for row in pool)
    manifest = dict(date="2026-10-27", seed=SEED, requested="New GPT swarm on latest coached traces, using the established WMDP behavior rubric", source_variant="wmdp.should", source_arm="coached", source_sha256=hashes, reader_prompt_sha256=hashlib.sha256(prompt).hexdigest(), prompt_authorization="Unchanged reader prompt approved in swarm/round2/approval.json; deployment authorized in this conversation", reading="One blinded read per eligible row; model, arm, pool, system instructions, and judge outputs hidden", source_counts={"coached": 600, "natural_error": sum(r["kind"] == "resample_error" for r in controls), "style_error": sum(r["arm"] == "cotctl" for r in controls)}, assigned=len(pool), counts={model: {category: counts[model, category] for category in ("coached", "natural_error", "style_error")} for model in MODELS}, excluded_questions=sorted(excluded_questions), excluded=len(excluded), assignments=assignments, work_directory=str(WORK), new_sampling=False, external_paid_judging=False)
    save(ROOT / "key.json", key)
    save(ROOT / "exclusions.json", excluded)
    save(ROOT / "manifest.json", manifest)
    save(ROOT / "status.json", {"phase": "prepared", "assigned": len(pool), "completed": 0, "shards": {name: "pending" for name in assignments}})
    print(json.dumps({k: manifest[k] for k in ("source_counts", "assigned", "counts", "excluded_questions", "excluded")}))
    print(json.dumps({"shards": len(shards), "max_characters": max(a["characters"] for a in assignments.values())}))


if __name__ == "__main__":
    main()
