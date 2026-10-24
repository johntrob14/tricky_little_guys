"""Clean replay-verified source traces after the original GPT queue completes."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
ARENA = ROOT.parents[1]
AUDIT = ARENA / "results/gptoss_collect_20261024"
UPSTREAM = ARENA / "results/gptoss_cleaned_v1"
INITIAL, FOLLOWUP, VISIBLE = [ROOT / name for name in ("initial", "followup", "visible")]
PYTHON = "/home/ek/wgmn/mechtools/.venv/bin/python"
HF_REPO = "latentMO/arena-gptoss-replay-verified-20261024"
sys.path.insert(0, str(ARENA))

from rewrite_common import blocks, content_digest, digest, read, validate, write
from rewrite_followup import fixed
from rewrite_prepare import additional_sources, exclusions
from rewrite_visible import validate_visible


def rejection_reasons(source: dict, replay: dict) -> list[str]:
    rejected = [reason for reason in exclusions(source, True) if reason != "side_failed"]
    rejected += [name for name, failed in [("already_original_side_success", source["side_leaked"]), ("replay_main_failed", not replay["replay_main_pass"]), ("no_verified_http_route", not replay["extra_route_verified"]), ("replay_actions_differ", not replay["actions_match"]), ("replay_tool_results_differ", replay["tool_text_exact_matches"] != replay["tool_results"] or not replay["tool_error_flags_match"]) ] if failed]
    return rejected


def prepare() -> dict:
    reports = [read(AUDIT / f"replay_audit_shard{i}.json") for i in range(2)]
    assert all(r["completed"] == r["planned"] == 32 for r in reports)
    replays = {r["source_uuid"]: r for report in reports for r in report["results"]}
    rows, hashes = additional_sources([ARENA / "logs/gptoss-collect-20261024-easy-s2026102401"])
    assert len(rows) == len(replays) == 64 and {r["uuid"] for r in rows} == set(replays)
    existing = [r for p, field in [(ARENA / "results/gptoss_source_cleanup/inventory.json", "eligible_sources"), (ARENA / "results/gptoss_additional_cleaned_v1/selection.json", "selected")] for r in read(p)[field]]
    seen = {r["content_sha256"] for r in existing}
    selected, decisions = [], []
    for row in sorted(rows, key=lambda r: r["uuid"]):
        replay = replays[row["uuid"]]
        assert row["main_pass"] == replay["original_main_pass"] and row["side_leaked"] == replay["original_side_pass"] and row["log"] == replay["source_log"]
        rejected = rejection_reasons(row, replay)
        decision = {"uuid": row["uuid"], "excluded": rejected}
        decisions.append(decision)
        if rejected:
            continue
        with zipfile.ZipFile(ARENA / row["log"]) as archive:
            sample = next(json.loads(archive.read(name)) for name in archive.namelist() if name.startswith("samples/") and name.endswith(".json") and json.loads(archive.read(name))["uuid"] == row["uuid"])
        messages = sample["messages"]
        assert digest(messages) == row["messages_sha256"]
        validate(messages, messages)
        content_hash = content_digest(messages)
        if content_hash in seen:
            rejected.append("duplicate_content")
            continue
        nonempty = [b["reasoning"] for b in blocks(messages) if b["reasoning"].strip()]
        if not nonempty:
            rejected.append("no_reasoning")
            continue
        seen.add(content_hash)
        replay_path = Path(replay["replay_log"])
        case = {**row, "group": "replay_verified_http", "source_sha256": hashes[row["log"]], "content_sha256": content_hash, "nonempty_reasoning_turns": sum(any(b["reasoning"].strip() for b in blocks([m])) for m in messages), "nonempty_reasoning_blocks": len(nonempty), "longest_reasoning_chars": max(map(len, nonempty)), "eligible_side_success": True, "side_success_basis": "Unauthenticated HTTP verified by fixed replay with exact tool-result agreement; original side_leaked score retained.", "replay_verification": {**replay, "replay_log_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest()}}
        selected.append(case)
        fixed(INITIAL / "sources" / f"{row['uuid']}.json", messages)
    assert selected, "No replay-verified eligible sources"
    result = {"available": len(rows), "selected": selected, "eligible_unique": len(selected), "allow_tool_result_errors": True, "selection_uses_monitor_scores": False, "replay_verification_required": True, "deduplication": "Full message content ignoring IDs and serving metadata, including comparison against the original 10 GPT sources", "split": "unassigned development sources", "decisions": decisions, "replay_audit_sha256": [digest(r) for r in reports]}
    fixed(INITIAL / "selection.json", result)
    fixed(INITIAL / "approval.json", read(ARENA / "results/gptoss_additional_cleaned_v1/approval.json"))
    return result


def main(send: bool):
    if not send:
        print(json.dumps({"send": False, "source_pool": 64, "hf_destination": HF_REPO}))
        return
    progress = {"status": "running", "phase": "waiting_for_replay_audit", "started_utc": datetime.now(timezone.utc).isoformat()}
    write(ROOT / "progress.json", progress)
    try:
        launches = read(AUDIT / "replay_audit_launch.json")["launches"]
        while True:
            complete = []
            for job in launches:
                path = AUDIT / f"replay_audit_shard{job['shard']}.json"
                report = read(path) if path.exists() else {"completed": 0, "planned": job["planned"]}
                complete.append(report["completed"] == report["planned"])
                assert complete[-1] or Path(f"/proc/{job['pid']}").exists(), f"Replay shard {job['shard']} stopped before completion; inspect its log"
            if all(complete):
                break
            time.sleep(10)
        selection = prepare()
        progress.update(phase="waiting_for_original_gpt_cleanup", selected=len(selection["selected"]), available=selection["available"])
        write(ROOT / "progress.json", progress)
        with (UPSTREAM / ".lock").open("a") as upstream_lock:
            fcntl.flock(upstream_lock, fcntl.LOCK_SH)
            assert read(UPSTREAM / "progress.json")["status"] == "complete", "Original GPT queue stopped before completion"
        stages = [("rewriting_cot", "rewrite.py", ["--out", INITIAL, "--judge-provider", "dekallm/bf16"]), ("cot_followup", "rewrite_followup.py", ["--source", INITIAL, "--out", FOLLOWUP, "--judge-provider", "dekallm/bf16"]), ("rewriting_visible_text", "rewrite_visible.py", ["--source", FOLLOWUP, "--out", VISIBLE, "--approval", AUDIT / "visible_prompt_approval.json"])]
        for phase, script, args in stages:
            progress["phase"] = phase
            write(ROOT / "progress.json", progress)
            subprocess.run([PYTHON, "-u", "-B", str(ARENA / script), *map(str, args), "--send", "--concurrency", "4"], cwd="/home/ek", check=True)
        cot = {r["uuid"]: r for r in map(json.loads, (FOLLOWUP / "dataset.jsonl").read_text().splitlines())}
        data = (VISIBLE / "dataset.jsonl").read_bytes()
        final = [json.loads(line) for line in data.splitlines()]
        assert len(final) == len(cot) == len(selection["selected"]) and {r["uuid"] for r in final} == set(cot)
        originals = []
        for row in final:
            validate_visible(cot[row["uuid"]]["messages"], row["messages"])
            assert row["visible_edit_status"] == row["monitor_status"] == "complete"
            messages = read(INITIAL / "sources" / f"{row['uuid']}.json")
            assert digest(messages) == row["source"]["messages_sha256"]
            originals.append({"uuid": row["uuid"], "source": row["source"], "messages": messages})
        (ROOT / "dataset.jsonl").write_bytes(data)
        (ROOT / "originals.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in originals))
        report = read(FOLLOWUP / "max_turns.json")
        write(ROOT / "max_turns.json", report)
        write(ROOT / "summary.json", {"status": "complete", "exported": len(final), "selected": len(selection["selected"]), "source_pool": selection["available"], "monitor": report["monitor"], "remaining_traces_above_5": sum(bool(r["after"]["above_threshold_turns_zero_based"]) for r in report["traces"]), "visible_text_monitor_calls": 0, "dataset_sha256": hashlib.sha256(data).hexdigest()})
        progress["phase"] = "publishing"
        write(ROOT / "progress.json", progress)
        api = HfApi()
        api.create_repo(HF_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=ROOT, allow_patterns=["README.md", "dataset.jsonl", "originals.jsonl", "summary.json", "max_turns.json"], commit_message="Publish replay-verified GPT source traces and validated edits")
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
