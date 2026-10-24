"""Collect the already included gpt-oss source traces after their cleanup finishes."""

import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARENA = ROOT.parents[1]
UPSTREAM = ROOT.parent / "deepseek_cleaned_followup_dekallm"
sys.path.insert(0, str(ARENA))

from rewrite_common import blocks, digest, read, validate, write


def main() -> None:
    inventory = read(ROOT / "inventory.json")
    expected = {r["uuid"]: r for r in inventory["eligible_sources"]}
    launch = read(UPSTREAM / "launch.json")
    for attempt in range(4):
        write(ROOT / "progress.json", {"status": "waiting_for_existing_cleanup", "eligible": len(expected), "source_run": str(UPSTREAM)})
        with (UPSTREAM / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            state = read(UPSTREAM / "progress.json")
        if state["run_status"] == "complete":
            break
        write(ROOT / "progress.json", {"status": "dependency_incomplete", "upstream_status": state["run_status"], "resume_attempts": attempt})
        assert state["run_status"] in ("monitoring_before_failed", "rewriting_failed"), "Upstream stopped unexpectedly; inspect its log before resuming"
        assert attempt < 3, "Upstream remains incomplete after three resumes"
        write(ROOT / "progress.json", {"status": "resuming_existing_cleanup", "resume_attempt": f"{attempt + 1}/3"})
        with (ROOT / f"upstream_retry_{attempt + 1}.log").open("a") as log:
            result = subprocess.run(launch["command"], cwd=launch["cwd"], stdout=log, stderr=subprocess.STDOUT)
        write(ROOT / f"upstream_retry_{attempt + 1}.json", {"returncode": result.returncode, "progress": read(UPSTREAM / "progress.json"), "command": launch["command"]})
    rows = [json.loads(line) for line in (UPSTREAM / "dataset.jsonl").read_text().splitlines() if line]
    selected = [r for r in rows if r["uuid"] in expected]
    assert len(selected) == len(expected) and {r["uuid"] for r in selected} == set(expected)
    for row in selected:
        assert "gpt-oss-120b" in row["source"]["model"] and row["source"]["content_sha256"] == expected[row["uuid"]]["content_sha256"]
        assert row["monitor_status"] == "complete" and digest(row["messages"]) == row["edited_messages_sha256"]
        validate(read(ROOT.parent / "deepseek_cleaned_v2" / "sources" / f"{row['uuid']}.json"), row["messages"])
    temporary = ROOT / "dataset.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    temporary.replace(ROOT / "dataset.jsonl")
    report = read(UPSTREAM / "max_turns.json")
    report["traces"] = [r for r in report["traces"] if r["uuid"] in expected]
    assert len(report["traces"]) == len(expected) and all(r["after"] is not None for r in report["traces"])
    write(ROOT / "max_turns.json", report)
    config = read(UPSTREAM / "config.json")
    summary = {"status": "complete", "eligible": len(expected), "exported": len(selected), "assistant_turns": sum(m["role"] == "assistant" for r in selected for m in r["messages"]), "nonempty_reasoning_turns": sum(bool(b["reasoning"].strip()) for r in selected for b in blocks(r["messages"])), "source_run": str(UPSTREAM), "editor": config["editor"], "monitor": config["monitor"], "backend": config["backend"], "threshold": config["threshold"], "subset_export_model_calls": 0, "dataset_sha256": hashlib.sha256((ROOT / "dataset.jsonl").read_bytes()).hexdigest()}
    write(ROOT / "summary.json", summary)
    write(ROOT / "progress.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    with (ROOT / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
