"""Finish the queued CoT work, then clean visible text for all eligible base GPT sources."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
ARENA = ROOT.parents[1]
PYTHON = "/home/ek/wgmn/mechtools/.venv/bin/python"
LEGACY = ARENA / "results/gptoss_source_cleanup"
ADDITIONAL = ARENA / "results/gptoss_additional_cleaned_v1"
FOLLOWUP = ARENA / "results/gptoss_additional_followup_v1"
COT = ROOT / "cot"
VISIBLE = ROOT / "visible"
APPROVAL = ARENA / "results/gptoss_collect_20261024/visible_prompt_approval.json"
HF_REPO = "latentMO/arena-gptoss-cleaned-20261024"
sys.path.insert(0, str(ARENA))

from rewrite_common import digest, read, write
from rewrite_followup import fixed
from rewrite_visible import validate_visible


def command(script: Path, *args) -> list[str]:
    return [PYTHON, "-u", "-B", str(script), *map(str, args)]


STAGES = [
    ("waiting_for_shared_cot_followup", command(LEGACY / "collect.py")),
    ("rewriting_additional_cot", command(ARENA / "rewrite.py", "--out", ADDITIONAL, "--send", "--concurrency", 4, "--judge-provider", "dekallm/bf16")),
    ("additional_cot_followup", command(ARENA / "rewrite_followup.py", "--source", ADDITIONAL, "--out", FOLLOWUP, "--send", "--concurrency", 4, "--judge-provider", "dekallm/bf16")),
    ("rewriting_visible_text", command(ARENA / "rewrite_visible.py", "--source", COT, "--out", VISIBLE, "--send", "--concurrency", 4, "--approval", APPROVAL)),
]


def combine() -> list[dict]:
    rows, reports, inputs = [], [], []
    monitor = read(LEGACY / "config.json")["monitor"]
    for source in (LEGACY, FOLLOWUP):
        config = read(source / "config.json")
        assert config["monitor"] == monitor
        data = (source / "dataset.jsonl").read_bytes()
        selected = [json.loads(line) for line in data.splitlines() if line]
        assert all(r["monitor_status"] == "complete" and digest(r["messages"]) == r["edited_messages_sha256"] for r in selected)
        rows.extend(selected)
        reports.extend(read(source / "max_turns.json")["traces"])
        inputs.append({"source": str(source), "dataset_sha256": hashlib.sha256(data).hexdigest(), "selected": len(selected)})
    expected = {r["uuid"] for r in read(LEGACY / "inventory.json")["eligible_sources"]} | {r["uuid"] for r in read(ADDITIONAL / "selection.json")["selected"]}
    assert len(rows) == len(expected) == 10 and {r["uuid"] for r in rows} == expected
    assert len({r["source"]["content_sha256"] for r in rows}) == len(rows)
    assert all("gpt-oss-120b" in r["source"]["model"] for r in rows)
    assert {r["uuid"] for r in reports} == expected and all(r["after"] is not None for r in reports)
    fixed(COT / "config.json", {"inputs": inputs, "monitor": monitor, "threshold": 5, "comparison": ">", "valid_rewrite_passes_per_selected_turn": 1, "unique_base_gpt_sources": len(rows)})
    write(COT / "max_turns.json", {"monitor": monitor, "threshold": 5, "comparison": ">", "traces": reports})
    (COT / "dataset.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return rows


def main(send: bool) -> None:
    if not send:
        print(json.dumps({"stages": STAGES, "total_sources": 10, "send": False}, indent=2))
        return
    progress = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "total_sources": 10}
    try:
        for phase, cmd in STAGES:
            if phase == "rewriting_visible_text":
                combine()
            progress["phase"] = phase
            write(ROOT / "progress.json", progress)
            print(json.dumps({"phase": phase}), flush=True)
            subprocess.run(cmd, cwd="/home/ek", check=True)
        cot = {r["uuid"]: r for r in combine()}
        data = (VISIBLE / "dataset.jsonl").read_bytes()
        final = [json.loads(line) for line in data.splitlines() if line]
        assert len(final) == len(cot) and {r["uuid"] for r in final} == set(cot)
        for row in final:
            validate_visible(cot[row["uuid"]]["messages"], row["messages"])
            assert row["visible_edit_status"] == row["monitor_status"] == "complete"
        temporary = ROOT / "dataset.jsonl.tmp"
        temporary.write_bytes(data)
        temporary.replace(ROOT / "dataset.jsonl")
        originals = []
        legacy_ids = {r["uuid"] for r in read(LEGACY / "inventory.json")["eligible_sources"]}
        for row in final:
            source = ARENA / "results/deepseek_cleaned_v2" if row["uuid"] in legacy_ids else ADDITIONAL
            messages = read(source / "sources" / f"{row['uuid']}.json")
            assert digest(messages) == row["source"]["messages_sha256"]
            originals.append({"uuid": row["uuid"], "source": row["source"], "messages": messages})
        (ROOT / "originals.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in originals))
        report = read(COT / "max_turns.json")
        write(ROOT / "max_turns.json", report)
        write(ROOT / "summary.json", {"status": "complete", "exported": len(final), "selected": len(cot), "monitor": report["monitor"], "remaining_traces_above_5": sum(bool(r["after"]["above_threshold_turns_zero_based"]) for r in report["traces"]), "visible_text_monitor_calls": 0, "dataset_sha256": hashlib.sha256(data).hexdigest(), "stage_summaries": [str(p / "summary.json") for p in (LEGACY, ADDITIONAL, FOLLOWUP, VISIBLE)]})
        progress["phase"] = "publishing"
        write(ROOT / "progress.json", progress)
        api = HfApi()
        api.create_repo(HF_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=ROOT, allow_patterns=["README.md", "dataset.jsonl", "originals.jsonl", "summary.json", "max_turns.json"], commit_message="Publish validated original and edited GPT-OSS arena traces")
        progress.update(status="complete", phase="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    finally:
        if progress["status"] == "running":
            progress["status"] = "interrupted"
        write(ROOT / "progress.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    with (ROOT / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.send)
