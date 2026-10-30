"""Verify complete checkpoint exports, recompute rule scores, archive to git main.

Runs on the box. The numerical summaries remain provisional until raw-data and
judge audits. The raw files are copied with pod pull before compression.
"""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "grading")]
import astra_continuous
from cotswitch import graders, extract


def archive(local, destination):
    local, destination = Path(local), Path(destination)
    manifests = []
    for directory in sorted(local.iterdir()):
        if not directory.is_dir() or not (directory / "DONE").exists():
            continue
        rows = [json.loads(line) for line in (directory / "generations.jsonl").open()]
        keys = {(r["key"], r["variant"], r["seed"]) for r in rows}
        assert len(rows) == len(keys) == 972, (directory, len(rows), len(keys))
        assert collections.Counter(r["variant"] for r in rows) == {"base": 324, "shot4": 324, "shotstack": 324}
        target = destination / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("generations.jsonl", "rendered_prompts.jsonl", "pilot.jsonl"):
            source = directory / name
            if not source.exists():
                continue
            compressed = target / (name + ".gz")
            with source.open("rb") as original, gzip.open(compressed, "wb", compresslevel=6) as out:
                shutil.copyfileobj(original, out)
            with gzip.open(compressed, "rb") as out:
                assert hashlib.sha256(out.read()).hexdigest() == hashlib.sha256(source.read_bytes()).hexdigest()
        for name in ("metadata.json", "progress.json", "DONE", "PILOT_DONE"):
            if (directory / name).exists():
                shutil.copy2(directory / name, target / name)
        scored = []
        for row in rows:
            reasoning = row["reasoning"]
            if row["constraint"] == "ignore_question":
                strict, partial = None, None
            else:
                strict = int(graders._grade_default(row["constraint"], reasoning, row["grader_args"])["strict"]) if reasoning.strip() else 0
                partial = astra_continuous.score(row, strict)
            prediction = extract.parse_answer(row["response"])
            scored.append(dict(key=row["key"], variant=row["variant"], seed=row["seed"], constraint=row["constraint"],
                               strict=strict, partial=partial, correct=prediction == row["gold"], prediction=prediction,
                               completed=bool(row["response"].strip()) and row["think_closed"] and row["finish_reason"] != "length",
                               words=len(reasoning.split()), truncated=row["finish_reason"] == "length"))
        (target / "rule_scores.json").write_text(json.dumps(scored, indent=1))
        summary = []
        for variant in ("base", "shot4", "shotstack"):
            group = [r for r in scored if r["variant"] == variant]
            rule = [r for r in group if r["strict"] is not None]
            summary.append(dict(variant=variant, n=len(group), rule_n=len(rule), strict=statistics.mean(r["strict"] for r in rule),
                partial=statistics.mean(r["partial"] for r in rule), accuracy=statistics.mean(r["correct"] for r in group),
                completed=sum(r["completed"] for r in group), truncated=sum(r["truncated"] for r in group),
                median_words=statistics.median(r["words"] for r in group),
                strict_and_completed=sum(r["strict"] == 1 and r["completed"] for r in rule)/len(rule)))
        (target / "summary.rule_based.json").write_text(json.dumps(dict(status="provisional; raw-data audit and ignore_question judge pending", metrics=summary), indent=2))
        manifests.append(dict(checkpoint=directory.name, rows=len(rows), unique_keys=len(keys),
            generation_sha256=hashlib.sha256((directory / "generations.jsonl").read_bytes()).hexdigest()))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "completion_manifest.json").write_text(json.dumps(manifests, indent=2))
    return manifests


if __name__ == "__main__":
    print(json.dumps(archive(*sys.argv[1:3])))
