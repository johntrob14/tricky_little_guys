"""Continue the current cleanup with score >=5 as the rewrite trigger and fresh final judgments."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mechtools import *

from reclean import ARENA, RESULTS, jsonl, package, publish, rows
from reclean_verify import verify
from rewrite import parse_monitor
from rewrite_common import digest, read, write
from rewrite_followup import fixed, main as followup

SOURCE = RESULTS / "reclean_luna_20261025"
ROOT = RESULTS / "reclean_luna_max4_20261025"
HF_REPO = "latentMO/arena-recleaned-luna-max4-20261025"


def prepare(source: Path, root: Path, concurrency: int) -> dict:
    parent = read(source / "config.json")
    assert read(source / "progress.json")["status"] == "complete"
    assert read(source / "summary.json")["dataset_sha256"] == hashlib.sha256((source / "dataset.jsonl").read_bytes()).hexdigest()
    origin = {"source": str(source.relative_to(ARENA)), "condition": "after", "dataset_sha256": read(source / "summary.json")["dataset_sha256"], "scores_sha256": hashlib.sha256((source / "scores.jsonl").read_bytes()).hexdigest(), "config_sha256": digest(parent)}
    config = {k: parent[k] for k in ("editor", "judge", "monitor", "editor_prompt_sha256")}
    config.update(concurrency=concurrency, threshold=5, comparison=">=", max_acceptable_score=4, fresh_before_scores=False, before_score_origin=origin, fresh_after_scores=True)
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    fixed(prepared / "config.json", config)
    fixed(prepared / "approval.json", read(source / "followup/prompt_approval.json"))
    fixed(root / "authorization.json", {"approved_by": "Ethan", "user_message": "I want to move the max suspicion threshold for a rewrite to 4. so 5 or above means rewrite.", "scope": "One targeted continuation pass on current cleaned transcripts at score >=5, fresh final monitoring of every nonempty turn, all sources retained, visible text preserved, versioned publication.", "concurrency_authorization": "up to 32 concurrent and go ahead.", "editor_prompt_sha256": config["editor_prompt_sha256"], "monitor_prompt_sha256": config["monitor"]["prompt_template_sha256"]})
    for name, expected in (("editor_prompt.txt", config["editor_prompt_sha256"]), ("monitor_prompt.txt", config["monitor"]["prompt_template_sha256"])):
        text = (source / name).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == expected
        for directory in (root, prepared):
            path = directory / name
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
    dataset, cases, inputs, imported = rows(source / "dataset.jsonl"), [], [], []
    assert len(dataset) == len({r["uuid"] for r in dataset}) == len({r["source"]["content_sha256"] for r in dataset}) == 318
    for row in dataset:
        uuid, messages = row["uuid"], row["messages"]
        assert digest(messages) == row["edited_messages_sha256"]
        case = {**row["source"], "messages_sha256": digest(messages), "previous_export_messages_sha256": digest(messages), "input_source": str((source / "dataset.jsonl").relative_to(ARENA))}
        fixed(prepared / "sources" / f"{uuid}.json", messages)
        fixed(prepared / "working" / uuid / "edited.json", messages)
        cases.append(case)
        inputs.append({"uuid": uuid, "source": case, "messages": messages})
        for score in read(source / "followup/working" / uuid / "scores_after.json"):
            if score["skipped_empty"]:
                continue
            assert score["score"] is not None and score["monitor_error"] is None
            previous = source / "followup/working" / uuid / "monitor_after" / f"{score['turn']:03d}"
            target = root / "followup/working" / uuid / "monitor_before" / f"{score['turn']:03d}"
            request, accepted = read(previous / "request.json"), read(previous / "accepted.json")
            body = read(previous / accepted["response_file"])
            assert request["config"] == config["judge"] and accepted["request_sha256"] == digest(request)
            assert accepted["response_sha256"] == digest(body) and parse_monitor(body) == accepted["value"] == score["score"]
            for name in ("request.json", "accepted.json", accepted["response_file"]):
                fixed(target / name, read(previous / name))
            imported.append({"uuid": uuid, "turn": score["turn"], "response_id": body["id"], "request_sha256": digest(request), "response_sha256": digest(body), "cost_usd": body["usage"]["cost"]})
    selection = {"selected": cases, "source_unique": len(cases), "selection_uses_monitor_scores": False, "parent": origin}
    fixed(prepared / "selection.json", selection)
    fixed(prepared / "progress.json", {"run_status": "complete"})
    fixed(root / "source_manifest.json", selection)
    fixed(root / "cache_import.json", {"source": origin, "monitor": config["monitor"], "imported_judgments": len(imported), "expected_judgments": 5599, "previously_billed_usd": sum(r["cost_usd"] for r in imported), "records": imported})
    assert len(imported) == len({r["response_id"] for r in imported}) == 5599
    shutil.copyfile(source / "originals.jsonl", root / "originals.jsonl")
    jsonl(root / "inputs.jsonl", inputs)
    return selection


async def main(source: Path, root: Path, send: bool, concurrency: int) -> None:
    assert 1 <= concurrency <= 32 and source != root
    selection = prepare(source, root, concurrency)
    print(json.dumps({"send": send, "selected_transcripts": f"{len(selection['selected'])}/318", "max_acceptable_score": 4, "rewrite_trigger": "score >= 5", "concurrency": concurrency}), flush=True)
    if not send:
        return
    progress = {"status": "running", "phase": "recleaning", "started_utc": datetime.now(timezone.utc).isoformat()}
    write(root / "progress.json", progress)
    try:
        await followup(root / "prepared", root / "followup", True, False, concurrency, allow_unscored_monitor=True)
        progress["phase"] = "validating_and_packaging"
        write(root / "progress.json", progress)
        package(root, selection)
        summary, imported = read(root / "summary.json"), read(root / "cache_import.json")
        summary["incremental_usage"] = {"known_cost_usd": summary["ledger"]["known_cost_usd"] - imported["previously_billed_usd"], "returned_generations": summary["ledger"]["returned_generations"] - imported["imported_judgments"], "imported_judgments": imported["imported_judgments"], "previously_billed_usd": imported["previously_billed_usd"]}
        write(root / "summary.json", summary)
        verify(root)
        progress["phase"] = "publishing"
        write(root / "progress.json", progress)
        publish(root, HF_REPO)
        progress.update(status="complete", phase="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    finally:
        if progress["status"] == "running":
            progress["status"] = "interrupted"
        write(root / "progress.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main(args.source.resolve(), args.out.resolve(), args.send, args.concurrency))
