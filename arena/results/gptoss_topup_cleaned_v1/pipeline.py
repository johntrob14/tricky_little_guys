"""Clean the fresh top-up and publish its union with the existing GPT source pool."""

import argparse
import asyncio
import fcntl
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
ARENA = ROOT.parents[1]
COLLECTION = ARENA / "results/gptoss_collect_20261024_topup"
LEGACY = ARENA / "results/gptoss_expanded_v1"
PREPARED, INITIAL, FOLLOWUP, VISIBLE = [ROOT / name for name in ("prepared", "initial", "followup", "visible")]
PYTHON = "/home/ek/wgmn/mechtools/.venv/bin/python"
sys.path.insert(0, str(ARENA))

from rewrite import main as configure_initial
from rewrite_common import digest, read, write
from rewrite_followup import fixed
from rewrite_prepare import prepare as select_sources

SPEC = importlib.util.spec_from_file_location("expanded_cleanup", LEGACY / "pipeline.py")
expanded = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(expanded)


def merge_selections(existing: dict, fresh: dict) -> tuple[dict, dict]:
    seen = {row["content_sha256"] for row in existing["selected"]}
    added, duplicates = [], []
    for row in fresh["selected"]:
        if row["content_sha256"] in seen:
            duplicates.append(row["uuid"])
        else:
            seen.add(row["content_sha256"])
            added.append(row)
    assert added, "No new eligible source transcripts"
    initial = {**fresh, "selected": added, "eligible_unique": len(added), "cross_batch_duplicate_uuids": duplicates, "reasoning_turns": sum(row["nonempty_reasoning_turns"] for row in added)}
    initial["decisions"] = [{**row, "excluded": ["duplicate_existing_content"] if row["uuid"] in duplicates else row["excluded"]} for row in fresh["decisions"]]
    combined = {"available": existing["available"] + fresh["available"], "selected": existing["selected"] + added, "eligible_unique": len(existing["selected"]) + len(added), "ignored_filters": list(expanded.IGNORED), "replay_required": False, "selection_uses_monitor_scores": False, "existing_selection_sha256": digest(existing), "fresh_selection_sha256": digest(fresh), "cross_batch_duplicate_uuids": duplicates, "deduplication": fresh["deduplication"], "split": fresh["split"]}
    assert len({row["uuid"] for row in combined["selected"]}) == len(combined["selected"])
    return combined, initial


def prepare() -> dict:
    config = read(COLLECTION / "config.json")
    fresh = select_sources(PREPARED, [], 200, True, [ARENA / "logs" / config["run"]], expanded.IGNORED)
    assert fresh["available"] == config["total_episodes"] == 48
    assert all("gpt-oss-120b" in row["model"] and row["checkpoint"] in (None, "base") for row in fresh["selected"])
    existing = read(LEGACY / "selection.json")
    combined, initial = merge_selections(existing, fresh)
    fixed(ROOT / "selection.json", combined)
    fixed(INITIAL / "selection.json", initial)
    for source, rows in ((LEGACY, existing["selected"]), (PREPARED, initial["selected"])):
        for row in rows:
            messages = read(source / "sources" / f"{row['uuid']}.json")
            assert digest(messages) == row["messages_sha256"]
            fixed(ROOT / "sources" / f"{row['uuid']}.json", messages)
            if source == PREPARED:
                fixed(INITIAL / "sources" / f"{row['uuid']}.json", messages)
    fixed(INITIAL / "approval.json", read(LEGACY / "initial/approval.json"))
    asyncio.run(configure_initial(INITIAL, False, 8, "dekallm/bf16", 6, 3))
    return combined


def main(send: bool, resume_followup: bool = False):
    if not send:
        print(json.dumps({"send": False, "planned_new_sources": 48, "existing_selected_sources": 158, "maximum_combined_sources": 206}))
        return
    progress = {"status": "running", "phase": "waiting_for_collection", "started_utc": datetime.now(timezone.utc).isoformat()}
    write(ROOT / "progress.json", progress)
    try:
        with (COLLECTION / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            assert read(COLLECTION / "status.json")["status"] == "complete", "Source collection stopped; inspect its log"
        selection = prepare()
        progress.update(selected=len(selection["selected"]), available=selection["available"])
        stages = [("rewriting_cot", INITIAL, "rewrite.py", ["--out", INITIAL, "--judge-provider", "dekallm/bf16", "--request-attempts", "6", "--validation-attempts", "3"]), ("cot_followup", FOLLOWUP, "rewrite_followup.py", ["--source", INITIAL, "--out", FOLLOWUP, "--judge-provider", "dekallm/bf16"]), ("rewriting_visible_text", VISIBLE, "rewrite_visible.py", ["--source", FOLLOWUP, "--out", VISIBLE, "--approval", ARENA / "results/gptoss_collect_20261024/visible_prompt_approval.json", "--request-attempts", "6", "--validation-attempts", "3"])]
        if resume_followup:
            assert all((INITIAL / "working" / row["uuid"] / "edited.json").exists() for row in read(INITIAL / "selection.json")["selected"]), "Initial CoT rewrites are incomplete"
            progress["resumed_at"] = "cot_followup"
            stages = stages[1:]
        for phase, output, script, args in stages:
            progress["phase"] = phase
            write(ROOT / "progress.json", progress)
            for attempt in range(3):
                result = subprocess.run([PYTHON, "-u", "-B", str(ARENA / script), *map(str, args), "--send", "--concurrency", "8"], cwd="/home/ek")
                state = read(output / "progress.json")["run_status"]
                if result.returncode == 0:
                    assert state == "complete"
                    break
                print(json.dumps({"phase": phase, "returncode": result.returncode, "status": state, "invocation": f"{attempt + 1}/3"}), flush=True)
                assert state in ("completed_with_failures", "monitoring_before_failed", "rewriting_failed") and attempt < 2, "Stage incomplete; inspect retained errors"
        progress["phase"] = "waiting_for_existing_exports"
        write(ROOT / "progress.json", progress)
        with (LEGACY / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            assert read(LEGACY / "progress.json")["status"] == "complete"
        expanded.combine(ROOT, LEGACY, FOLLOWUP, VISIBLE)
        progress["phase"] = "publishing"
        write(ROOT / "progress.json", progress)
        api = HfApi()
        api.upload_folder(repo_id=expanded.HF_REPO, repo_type="dataset", folder_path=ROOT, allow_patterns=["README.md", "dataset.jsonl", "originals.jsonl", "summary.json", "max_turns.json", "selection.json"], commit_message="Add the new GPT source batch to the validated cleaned dataset")
        progress.update(status="complete", phase="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    finally:
        if progress["status"] == "running":
            progress["status"] = "interrupted"
        write(ROOT / "progress.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--resume-followup", action="store_true", help="Use completed CoT rewrites; preserve incomplete original-text monitor checks and run fresh monitoring on edited text")
    args = parser.parse_args()
    with (ROOT / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.send, args.resume_followup)
