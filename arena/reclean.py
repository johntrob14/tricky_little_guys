"""Reclean the saved unique corpus with the arena context/Luna monitor; run from ~ with the mechtools Python."""

import argparse
import asyncio
import copy
import fcntl
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from mechtools import *

from rewrite import ARENA, EDITOR
from rewrite_common import blocks, digest, read, set_reasoning, validate, write
from rewrite_followup import fixed, main as followup
from reclean_html import build as build_html
from rl.monitor.monitors import get

RESULTS = ARENA / "results"
EXPORTS = [RESULTS / name for name in ("deepseek_cleaned_followup_dekallm16", "gptoss_topup_cleaned_v1")]
FIRST_PASSES = [RESULTS / name for name in ("deepseek_cleaned_followup_dekallm16", "gptoss_expanded_v1/followup_missing_scores", "gptoss_topup_cleaned_v1/followup_missing_scores", "gptoss_additional_followup_v1")]
MONITOR = get("controlarena_cot_only_context")
HF_REPO = "latentMO/arena-recleaned-luna-20261025"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def first_pass_reasoning(latest: list[dict], first: list[dict]) -> list[dict]:
    result = copy.deepcopy(latest)
    assert len(result) == len(first)
    for target, source in zip(result, first, strict=True):
        assert target["role"] == source["role"]
        for block, initial in zip(blocks([target]), blocks([source]), strict=True):
            set_reasoning(block, initial["reasoning"])
    validate(latest, result)
    return result


def prepare(root: Path, concurrency: int) -> dict:
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    latest, originals, parents = {}, {}, []
    for directory in EXPORTS:
        parents.append({"path": str(directory.relative_to(ARENA)), "dataset_sha256": hashlib.sha256((directory / "dataset.jsonl").read_bytes()).hexdigest(), "originals_sha256": hashlib.sha256((directory / "originals.jsonl").read_bytes()).hexdigest()})
        for row in rows(directory / "originals.jsonl"):
            if row["uuid"] in originals:
                assert originals[row["uuid"]]["messages"] == row["messages"]
            originals[row["uuid"]] = row
        for row in rows(directory / "dataset.jsonl"):
            assert digest(row["messages"]) == row["edited_messages_sha256"]
            latest[row["uuid"]] = row
    assert len(latest) == len(originals) == len({r["source"]["content_sha256"] for r in latest.values()}) == 318
    cases = []
    for uuid, row in latest.items():
        paths = [directory / "sources" / f"{uuid}.json" for directory in FIRST_PASSES if (directory / "sources" / f"{uuid}.json").exists()]
        assert len(paths) == 1, (uuid, paths)
        initial = read(paths[0])
        messages = first_pass_reasoning(row["messages"], initial)
        case = {**row["source"], "messages_sha256": digest(messages), "original_messages_sha256": digest(originals[uuid]["messages"]), "previous_export_messages_sha256": row["edited_messages_sha256"], "first_pass_source": str(paths[0].relative_to(ARENA)), "first_pass_messages_sha256": digest(initial), "visible_edit_status": row.get("visible_edit_status")}
        fixed(prepared / "sources" / f"{uuid}.json", messages)
        fixed(prepared / "working" / uuid / "edited.json", messages)
        cases.append(case)
    selection = {"selected": cases, "source_unique": len(cases), "selection_uses_monitor_scores": False, "parents": parents}
    fixed(prepared / "selection.json", selection)
    fixed(prepared / "progress.json", {"run_status": "complete"})
    prompt_approval = read(EXPORTS[0] / "prompt_approval.json")
    editor_prompt = (EXPORTS[0] / "editor_prompt.txt").read_text()
    assert hashlib.sha256(editor_prompt.encode()).hexdigest() == prompt_approval["prompt_sha256"]
    monitor_prompt = MONITOR.prompt_file.read_text()
    monitor = {"mode": "cot_only", "monitor": MONITOR.name, "judge_model": "openai/gpt-6-luna", "provider": "openai", "reasoning_effort": "low", "main_task_only": True, "input_channels": ["reasoning"], "prior_history": False, "prompt_template_sha256": hashlib.sha256(monitor_prompt.encode()).hexdigest()}
    config = {"editor": {**EDITOR, "attempts": 6}, "judge": {"model": "openai/gpt-6-luna", "provider": "openai", "reasoning": "low", "temperature": 0, "max_tokens": 6000, "attempts": 6}, "monitor": monitor, "editor_prompt_sha256": prompt_approval["prompt_sha256"], "concurrency": concurrency, "fresh_after_scores": True}
    fixed(prepared / "config.json", config)
    fixed(prepared / "approval.json", prompt_approval)
    fixed(root / "authorization.json", {"approved_by": "Ethan", "user_message": "up to 32 concurrent and go ahead.", "scope": "Reclean all unique exports from saved first-pass reasoning; preserve GPT visible-text cleanup; current arena context/Luna monitor; one rewrite of turns >5; independent final monitoring; retain all sources; versioned publication.", "concurrency": concurrency, "editor_prompt_sha256": config["editor_prompt_sha256"], "monitor_prompt_sha256": monitor["prompt_template_sha256"]})
    for name, text in (("editor_prompt.txt", editor_prompt), ("monitor_prompt.txt", monitor_prompt)):
        for directory in (prepared, root):
            path = directory / name
            if path.exists():
                assert path.read_text() == text
            else:
                path.write_text(text)
    jsonl(root / "originals.jsonl", list(originals.values()))
    jsonl(root / "inputs.jsonl", [{"uuid": c["uuid"], "source": c, "messages": read(prepared / "sources" / f"{c['uuid']}.json")} for c in cases])
    fixed(root / "source_manifest.json", selection)
    return selection


def package(root: Path, selection: dict) -> None:
    output = root / "followup"
    summary = read(output / "summary.json")
    assert summary["run_status"] == "complete"
    dataset, edits, scores = rows(output / "dataset.jsonl"), [], []
    assert len(dataset) == len(selection["selected"]) == 318
    for row in dataset:
        uuid = row["uuid"]
        baseline = read(root / "prepared/sources" / f"{uuid}.json")
        validate(baseline, row["messages"])
        assert row["monitor_status"] in ("complete", "complete_with_missing_scores")
        row["visible_edit_status"] = row["source"]["visible_edit_status"]
        edits.append({"uuid": uuid, "edits": read(output / "working" / uuid / "edits.json")})
        for condition in ("before", "after"):
            values = read(output / "working" / uuid / f"scores_{condition}.json")
            reasoning = ["\n".join(b["reasoning"] for b in blocks([m])) for m in (baseline if condition == "before" else row["messages"]) if m["role"] == "assistant"]
            assert len(values) == len(reasoning)
            assert all(v["reasoning_sha256"] == digest(text) and not v["reused_before_score"] for v, text in zip(values, reasoning, strict=True))
            scores.extend({"uuid": uuid, **v} for v in values)
    jsonl(root / "dataset.jsonl", dataset)
    jsonl(root / "edits.jsonl", edits)
    jsonl(root / "scores.jsonl", scores)
    for name in ("config.json", "selection.json", "max_turns.json"):
        write(root / name, read(output / name))
    summary.update(dataset_sha256=hashlib.sha256((root / "dataset.jsonl").read_bytes()).hexdigest(), retained=len(dataset), source_unique=len(selection["selected"]), visible_text_monitor_calls=0, fresh_final_scores=True)
    write(root / "summary.json", summary)
    build_html(root)
    with gzip.open(root / "requests_responses.jsonl.gz", "wt") as archive:
        for path in sorted((output / "working").rglob("request.json")):
            record = {"path": str(path.parent.relative_to(output)), "request": read(path), "responses": [read(p) for p in sorted(path.parent.glob("response.*.json"))]}
            for name in ("accepted.json", "unscored.json"):
                if (path.parent / name).exists():
                    record[name.removesuffix(".json")] = read(path.parent / name)
            archive.write(json.dumps(record, ensure_ascii=False) + "\n")


def publish(root: Path, repo_id: str = HF_REPO) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    files = [p.name for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl", ".gz", ".txt", ".md", ".html") and p.name not in ("publication.json", "launch.json", "progress.json")]
    commit = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=root, allow_patterns=files, commit_message="Publish all unique Arena transcripts recleaned with the context/Luna CoT-only monitor")
    verified = []
    for name in files:
        remote = Path(hf_hub_download(repo_id, name, repo_type="dataset", revision=commit.oid))
        sha = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert hashlib.sha256(remote.read_bytes()).hexdigest() == sha, name
        verified.append({"file": name, "sha256": sha, "bytes": remote.stat().st_size})
    write(root / "publication.json", {"repo_id": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}", "revision": commit.oid, "verified_files": len(verified), "expected_files": len(files), "files": verified, "verified_utc": datetime.now(timezone.utc).isoformat()})


async def main(root: Path, send: bool, concurrency: int) -> None:
    assert 1 <= concurrency <= 32
    selection = prepare(root, concurrency)
    print(json.dumps({"selected": f"{len(selection['selected'])}/318", "concurrency": concurrency, "send": send}), flush=True)
    if not send:
        return
    progress = {"status": "running", "phase": "recleaning", "started_utc": datetime.now(timezone.utc).isoformat()}
    write(root / "progress.json", progress)
    try:
        (root / "followup").mkdir(exist_ok=True)
        await followup(root / "prepared", root / "followup", True, False, concurrency, allow_unscored_monitor=True)
        progress["phase"] = "validating_and_packaging"
        write(root / "progress.json", progress)
        package(root, selection)
        progress["phase"] = "publishing"
        write(root / "progress.json", progress)
        publish(root)
        progress.update(status="complete", phase="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    finally:
        if progress["status"] == "running":
            progress["status"] = "interrupted"
        write(root / "progress.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "reclean_luna_20261025")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main(args.out.resolve(), args.send, args.concurrency))
