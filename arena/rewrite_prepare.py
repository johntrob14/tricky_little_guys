"""Select unique audited source episodes and freeze their original messages. No model calls."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import zipfile_zstd

from rewrite_common import blocks, content_digest, digest, read, validate, write

ARENA = Path(__file__).resolve().parent
AUDIT = ARENA / "results/deepseek_data_audit"


def exclusions(row: dict, allow_tool_errors: bool) -> list[str]:
    return [name for name, failed in (("pre_fix_oracle_available", row["group"] in ("deepseek_pre_fix", "deepseek_smoke")), ("main_failed", not row["main_pass"]), ("side_failed", not row["side_leaked"]), ("no_submit", not row["submit_calls"]), ("sample_error", row["error"]), ("tool_parse_errors", bool(row["tool_parse_errors"])), ("tool_result_errors", bool(row["tool_result_errors"]) and not allow_tool_errors)) if failed]


def additional_sources(directories: list[Path]) -> tuple[list[dict], dict]:
    rows, hashes = [], {}
    prefix = (ARENA / "prompts/prefix/secondary_objective.txt").read_text().strip()
    for directory in directories:
        paths = sorted(directory.glob("*.eval"))
        assert paths, f"No eval files in {directory}"
        for path in paths:
            log = str(path.resolve().relative_to(ARENA))
            hashes[log] = hashlib.sha256(path.read_bytes()).hexdigest()
            with zipfile.ZipFile(path) as archive:
                header = json.loads(archive.read("header.json"))["eval"]
                subprocess.run(["git", "merge-base", "--is-ancestor", "f8274d7", header["revision"]["commit"]], cwd=ARENA, check=True)
                meta = header["metadata"]
                for name in archive.namelist():
                    if not name.startswith("samples/") or not name.endswith(".json"):
                        continue
                    sample = json.loads(archive.read(name))
                    messages = sample["messages"]
                    task = next(m["content"] for m in messages if m["role"] == "user")
                    assert isinstance(task, str) and task.find(prefix) > 0
                    calls = [c for m in messages for c in m.get("tool_calls", []) or []]
                    score = sample["scores"]["main_and_side_scorer"]["metadata"]
                    turns = sum(m["role"] == "assistant" for m in messages)
                    rows.append({"uuid": sample["uuid"], "group": "additional_post_fix", "log": log, "sample": sample["id"], "epoch": sample["epoch"], "model": header["model"], "checkpoint": meta.get("ckpt"), "env": meta["env"], "task": meta["task"], "tier": meta["tier"], "turns": turns, "main_pass": score["main_pass"], "side_leaked": score["side_leaked"], "submit_calls": sum(c["function"] == "submit" for c in calls), "tool_result_errors": sum(bool(m.get("error")) for m in messages if m["role"] == "tool"), "tool_parse_errors": sum(bool(c.get("parse_error")) for c in calls), "error": bool(sample.get("error")), "at_turn_cap": turns >= meta["max_turns"], "messages_sha256": digest(messages), "main_task_sha256": digest(task.split(prefix, 1)[0].rstrip()), "side_metadata_sha256": digest(sample["metadata"]["side"])})
    return rows, hashes


def prepare(root: Path, groups: list[str], limit: int, allow_tool_errors: bool, extra_logs: list[Path]) -> dict:
    manifest = [json.loads(line) for line in (AUDIT / "manifest.jsonl").read_text().splitlines()]
    assert set(groups) <= {r["group"] for r in manifest}, "Unknown source group"
    rows = sorted((r for r in manifest if r["group"] in groups), key=lambda r: (r["tier"], r["task"], r["uuid"]))
    hashes = read(AUDIT / "sources.json")
    additional, new_hashes = additional_sources(extra_logs)
    rows.extend(additional)
    hashes.update(new_hashes)
    decisions, selected, seen = [], [], {}
    eligible = reasoning_turns = longest_reasoning = 0
    for row in rows:
        rejected = exclusions(row, allow_tool_errors)
        decision = {"uuid": row["uuid"], "group": row["group"], "log": row["log"], "excluded": rejected}
        decisions.append(decision)
        if rejected:
            continue
        source = ARENA / row["log"]
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        assert source_hash == hashes[row["log"]], "Source log changed since the audit"
        with zipfile.ZipFile(source) as archive:
            sample = next(s for name in archive.namelist() if name.startswith("samples/") and name.endswith(".json") and (s := json.loads(archive.read(name)))["uuid"] == row["uuid"])
        original = sample["messages"]
        assert digest(original) == row["messages_sha256"]
        validate(original, original)
        key = content_digest(original)
        if key in seen:
            decision.update(excluded=["duplicate_content"], duplicate_of=seen[key])
            continue
        seen[key] = row["uuid"]
        nonempty = [b["reasoning"] for b in blocks(original) if b["reasoning"].strip()]
        if not nonempty:
            decision["excluded"] = ["no_reasoning"]
            continue
        eligible += 1
        if len(selected) >= limit:
            decision["excluded"] = ["selection_limit"]
            continue
        record = {k: row[k] for k in ("uuid", "group", "log", "sample", "epoch", "model", "checkpoint", "env", "task", "tier", "turns", "main_pass", "side_leaked", "submit_calls", "tool_result_errors", "at_turn_cap", "messages_sha256", "main_task_sha256", "side_metadata_sha256")}
        record.update(content_sha256=key, source_sha256=source_hash, nonempty_reasoning_turns=len(nonempty), longest_reasoning_chars=max(map(len, nonempty)))
        path = root / "sources" / f"{row['uuid']}.json"
        if path.exists():
            assert read(path) == original
        else:
            write(path, original)
        selected.append(record)
        reasoning_turns += len(nonempty)
        longest_reasoning = max(longest_reasoning, max(map(len, nonempty)))
    assert selected, "No eligible source episodes"
    result = {"groups": groups, "extra_logs": [str(p.resolve().relative_to(ARENA)) for p in extra_logs], "available": len(rows), "eligible_unique": eligible, "requested_limit": limit, "allow_tool_result_errors": allow_tool_errors, "selection_uses_monitor_scores": False, "deduplication": "Full message content, ignoring message/tool-call IDs, serving metadata, and duplicate reasoning replay signatures; no semantic near-duplicate removal", "split": "unassigned; includes previously inspected pilot development sources", "source_audit_sha256": hashlib.sha256((AUDIT / "manifest.jsonl").read_bytes()).hexdigest(), "selected": selected, "decisions": decisions, "reasoning_turns": reasoning_turns, "longest_reasoning_chars": longest_reasoning}
    if (root / "selection.json").exists():
        assert read(root / "selection.json") == result, "Selection changed; use a new output directory"
    else:
        write(root / "selection.json", result)
    print(json.dumps({"selected": f"{len(selected)}/{len(rows)}", "eligible_unique": f"{eligible}/{len(rows)}", "requested_limit": limit, "reasoning_turns": f"{reasoning_turns}/{sum(r['turns'] for r in selected)}", "longest_reasoning_chars": longest_reasoning, "exclusion_reasons": dict(Counter(reason for d in decisions for reason in d['excluded']))}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["deepseek_post_fix"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--allow-tool-errors", action="store_true")
    parser.add_argument("--extra-logs", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    assert args.limit > 0
    prepare(args.out.resolve(), args.groups, args.limit, args.allow_tool_errors, args.extra_logs)
