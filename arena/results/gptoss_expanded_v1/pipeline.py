"""Clean all eligible post-fix base GPT transcripts, retaining outcome/error labels."""

import argparse
import asyncio
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
LEGACY = ARENA / "results/gptoss_cleaned_v1"
INITIAL, FOLLOWUP, VISIBLE = [ROOT / name for name in ("initial", "followup_missing_scores", "visible")]
PYTHON = "/home/ek/wgmn/mechtools/.venv/bin/python"
HF_REPO = "latentMO/arena-gptoss-cleaned-20261024"
IGNORED = ("main_failed", "side_failed", "no_submit", "tool_parse_errors")
sys.path.insert(0, str(ARENA))

from rewrite import main as configure_initial
from rewrite_common import digest, read, without_reasoning, write
from rewrite_followup import fixed
from rewrite_prepare import prepare as select_sources
from rewrite_visible import validate_visible


def prepare() -> dict:
    directories = [ARENA / "logs" / f"{prefix}-{tier}" for prefix in ("nocot-covert2", "oss-base-covert2") for tier in ("easy", "medium", "hard")]
    directories.append(ARENA / "logs/gptoss-collect-20261024-easy-s2026102401")
    selection = select_sources(ROOT, [], 200, True, directories, IGNORED)
    assert len(selection["selected"]) == 158 and selection["available"] == 160
    assert all("gpt-oss-120b" in row["model"] and row["checkpoint"] in (None, "base") for row in selection["selected"])
    existing = [row for path, key in [(ARENA / "results/gptoss_source_cleanup/inventory.json", "eligible_sources"), (ARENA / "results/gptoss_additional_cleaned_v1/selection.json", "selected")] for row in read(path)[key]]
    existing = {row["uuid"]: row["content_sha256"] for row in existing}
    assert len(existing) == 10 and all(next(row for row in selection["selected"] if row["uuid"] == uuid)["content_sha256"] == content for uuid, content in existing.items())
    pending = [row for row in selection["selected"] if row["uuid"] not in existing]
    initial = {**selection, "selected": pending, "eligible_unique": len(pending), "already_queued_uuids": sorted(existing), "total_eligible_in_combined_pool": len(selection["selected"]), "reasoning_turns": sum(row["nonempty_reasoning_turns"] for row in pending)}
    initial["decisions"] = [{**row, "excluded": ["already_queued"] if row["uuid"] in existing else row["excluded"]} for row in selection["decisions"]]
    fixed(INITIAL / "selection.json", initial)
    for row in pending:
        fixed(INITIAL / "sources" / f"{row['uuid']}.json", read(ROOT / "sources" / f"{row['uuid']}.json"))
    fixed(INITIAL / "approval.json", read(ARENA / "results/gptoss_additional_cleaned_v1/approval.json"))
    asyncio.run(configure_initial(INITIAL, False, 8, "dekallm/bf16", 6, 3))
    return selection


def combine(root: Path | None = None, legacy: Path | None = None, followup: Path | None = None, visible: Path | None = None) -> dict:
    root, legacy = root or ROOT, legacy or LEGACY
    followup, visible = followup or FOLLOWUP, visible or VISIBLE
    expected = {row["uuid"]: row for row in read(root / "selection.json")["selected"]}
    final, reports = [], []
    monitor = read(followup / "config.json")["monitor"]
    for source, report_root in ((legacy, legacy), (visible, followup)):
        rows = [json.loads(line) for line in (source / "dataset.jsonl").read_text().splitlines()]
        report = read(report_root / "max_turns.json")
        assert report["monitor"] == monitor
        final.extend(rows)
        reports.extend(report["traces"])
    assert len(final) == len(expected) and {row["uuid"] for row in final} == set(expected)
    assert len({row["source"]["content_sha256"] for row in final}) == len(final)
    assert len(reports) == len(final) and {row["uuid"] for row in reports} == set(expected) and all(row["after"] is not None for row in reports)
    by_uuid = {report["uuid"]: report for report in reports}
    originals = []
    for row in final:
        source = expected[row["uuid"]]
        original = read(root / "sources" / f"{row['uuid']}.json")
        assert row["source"]["content_sha256"] == source["content_sha256"] and digest(original) == source["messages_sha256"]
        missing = {condition: by_uuid[row["uuid"]][condition].get("unscored_turns_zero_based", []) for condition in ("before", "after")}
        assert row["monitor_status"] == ("complete_with_missing_scores" if missing["after"] else "complete")
        assert row.get("unscored_monitor_turns", {"before": [], "after": []}) == missing
        row["unscored_monitor_turns"] = missing
        assert digest(row["messages"]) == row["edited_messages_sha256"] and row["visible_edit_status"] == "complete"
        validate_visible(without_reasoning(original), without_reasoning(row["messages"]))
        originals.append({"uuid": row["uuid"], "source": source, "messages": original})
        row["source"].update({key: source[key] for key in ("main_pass", "side_leaked", "submit_calls", "tool_parse_errors", "sample_error", "ignored_failures")})
    for name, rows in (("dataset.jsonl", final), ("originals.jsonl", originals)):
        temporary = root / (name + ".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temporary.replace(root / name)
    write(root / "max_turns.json", {"monitor": monitor, "threshold": 5, "comparison": ">", "traces": reports})
    summary = {"status": "complete", "exported": len(final), "selected": len(expected), "source_pool": read(root / "selection.json")["available"], "ignored_filters": list(IGNORED), "replay_required": False, "monitor": monitor, "main_task_pass": sum(row["main_pass"] for row in expected.values()), "original_side_scorer_pass": sum(row["side_leaked"] for row in expected.values()), "with_submit": sum(bool(row["submit_calls"]) for row in expected.values()), "with_tool_parse_errors": sum(bool(row["tool_parse_errors"]) for row in expected.values()), "remaining_traces_above_5": sum(bool(row["after"]["above_threshold_turns_zero_based"]) for row in reports), "visible_text_monitor_calls": 0, "dataset_sha256": hashlib.sha256((root / "dataset.jsonl").read_bytes()).hexdigest()}
    summary["monitor_coverage"] = {condition: {"scored_turns": sum(r[condition]["scored_turns"] for r in reports), "nonempty_turns": sum(r[condition]["scored_turns"] + len(r[condition].get("unscored_turns_zero_based", [])) for r in reports), "unscored_turns": sum(len(r[condition].get("unscored_turns_zero_based", [])) for r in reports), "traces_with_missing_scores": sum(bool(r[condition].get("unscored_turns_zero_based")) for r in reports), "total_traces": len(reports)} for condition in ("before", "after")}
    write(root / "summary.json", summary)
    return summary


def main(send: bool, resume_followup: bool = False):
    selection = prepare()
    if not send:
        print(json.dumps({"send": False, "selected": len(selection["selected"]), "available": selection["available"], "new_sources": len(read(INITIAL / "selection.json")["selected"])}))
        return
    progress = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "selected": len(selection["selected"]), "available": selection["available"]}
    stages = [("rewriting_cot", INITIAL, "rewrite.py", ["--out", INITIAL, "--judge-provider", "dekallm/bf16", "--request-attempts", "6", "--validation-attempts", "3"]), ("cot_followup", FOLLOWUP, "rewrite_followup.py", ["--source", INITIAL, "--out", FOLLOWUP, "--judge-provider", "dekallm/bf16", "--allow-unscored-monitor"]), ("rewriting_visible_text", VISIBLE, "rewrite_visible.py", ["--source", FOLLOWUP, "--out", VISIBLE, "--approval", ARENA / "results/gptoss_collect_20261024/visible_prompt_approval.json", "--request-attempts", "6", "--validation-attempts", "3"])]
    if resume_followup:
        assert all((INITIAL / "working" / row["uuid"] / "edited.json").exists() for row in read(INITIAL / "selection.json")["selected"]), "Initial CoT rewrites are incomplete"
        progress["resumed_at"] = "cot_followup"
        stages = stages[1:]
    try:
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
        combine()
        progress["phase"] = "publishing"
        write(ROOT / "progress.json", progress)
        api = HfApi()
        api.create_repo(HF_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=ROOT, allow_patterns=["README.md", "dataset.jsonl", "originals.jsonl", "summary.json", "max_turns.json", "selection.json"], commit_message="Publish complete expanded GPT source pool with outcome labels")
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
