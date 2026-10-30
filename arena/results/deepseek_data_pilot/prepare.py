"""Reproduce the offline pilot with arena/.venv/bin/python; no model calls."""

import copy
import difflib
import hashlib
import json
import zipfile
from pathlib import Path

import zipfile_zstd
from inspect_ai._util.content import ContentReasoning
from inspect_ai.model._openrouter_reasoning import reasoning_to_openrouter_reasoning_details

from editing import apply_edits, digest, editor_view, reasoning_blocks, set_reasoning, without_reasoning, write_json

OUT = Path(__file__).resolve().parent
ARENA = OUT.parents[1]
AUDIT = ARENA / "results/deepseek_data_audit"
STRATA = [("easy", "missing_user_id_in_guess_response"), ("easy", "sponsor_statsv2_diggcount_total"), ("medium", "token_bucket_rate_limiter_per_user"), ("hard", "video_url_format_validation")]


def validate(parent: list[dict], child: list[dict]) -> None:
    assert without_reasoning(parent) == without_reasoning(child), "Non-reasoning content changed"
    for block in reasoning_blocks(child):
        replay = reasoning_to_openrouter_reasoning_details(ContentReasoning.model_validate(block))
        if replay is not None:
            assert replay["reasoning_details"][0]["text"] == block["reasoning"]


def main() -> None:
    working = OUT / "working"
    working.mkdir(exist_ok=True)
    manifest = [json.loads(line) for line in (AUDIT / "manifest.jsonl").read_text().splitlines()]
    post = [r for r in manifest if r["group"] == "deepseek_post_fix"]
    exclusions = {r["uuid"]: [name for name, failed in (("main_failed", not r["main_pass"]), ("side_failed", not r["side_leaked"]), ("no_submit", not r["submit_calls"]), ("parse_errors", bool(r["tool_parse_errors"])), ("tool_result_errors", bool(r["tool_result_errors"])), ("sample_error", r["error"])) if failed] for r in post}
    eligible = [r for r in post if not exclusions[r["uuid"]]]
    selected = [min((r for r in eligible if (r["tier"], r["task"]) == stratum), key=lambda r: r["uuid"]) for stratum in STRATA]
    examples = json.loads((OUT / "examples.json").read_text())
    assert [e["uuid"] for e in examples] == [r["uuid"] for r in selected]
    prompt = (OUT / "editor_prompt.txt").read_text()
    source_hashes = json.loads((AUDIT / "sources.json").read_text())
    summaries, previews, source_checks = [], [], []
    signed = naive_failures = nonempty = total_turns = 0
    for row, example in zip(selected, examples):
        source = ARENA / row["log"]
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        assert source_hash == source_hashes[row["log"]]
        source_checks.append({"path": row["log"], "sha256": source_hash})
        with zipfile.ZipFile(source) as archive:
            sample = next(s for n in archive.namelist() if n.startswith("samples/") and n.endswith(".json") and (s := json.loads(archive.read(n)))["uuid"] == row["uuid"])
        original = sample["messages"]
        assert digest(original) == row["messages_sha256"]
        blocks = reasoning_blocks(original)
        for block in blocks:
            validate([{"role": "assistant", "content": [block]}], [{"role": "assistant", "content": [block]}])
            if block.get("signature"):
                signed += 1
                naive = {**block, "reasoning": "PILOT_EDIT_SENTINEL"}
                replay = reasoning_to_openrouter_reasoning_details(ContentReasoning.model_validate(naive))
                naive_failures += replay["reasoning_details"][0]["text"] == block["reasoning"]
        assistant_indexes = [i for i, m in enumerate(original) if m["role"] == "assistant"]
        total_turns += len(assistant_indexes)
        nonempty += sum(bool(b["reasoning"].strip()) for b in blocks)
        index = assistant_indexes[example["turn"]]
        current_blocks = reasoning_blocks([original[index]])
        assert len(current_blocks) == 1
        before = current_blocks[0]["reasoning"]
        edits = example.get("edits")
        if "keep_suffix" in example:
            suffix = example["keep_suffix"]
            assert before.endswith(suffix)
            edits = [{"old": before[:-len(suffix)], "new": ""}]
        after = apply_edits(before, edits)
        edited = copy.deepcopy(original[:index + 1])
        set_reasoning(reasoning_blocks([edited[-1]])[0], after)
        empty = without_reasoning(original)
        validate(original[:index + 1], edited)
        validate(original, empty)
        assert all(not b["reasoning"] for b in reasoning_blocks(empty))
        payload = {"history": editor_view(original[:index]), "current": editor_view([original[index]])[0]}
        assert len(payload["history"]) == index
        request = {"uuid": row["uuid"], "turn": example["turn"], "source_sha256": source_hash, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "payload": payload}
        request["request_sha256"] = digest(payload)
        stem = row["uuid"]
        write_json(working / f"{stem}.original.json", original)
        write_json(working / f"{stem}.no_reasoning.json", empty)
        write_json(working / f"{stem}.manual_prefix.json", edited)
        write_json(working / f"{stem}.request.json", request)
        diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile="original", tofile="manual deletion", n=0, lineterm=""))
        secret = sample["metadata"]["side"]["secret_value"]
        previews.append(f"## {row['task']} / {row['tier']} / turn {example['turn']}\n\nSource: `{stem}`. {example['review_notes']}\n\n```diff\n{diff}\n```\n\nRetained reasoning:\n\n```text\n{after}\n```\n".replace(secret, "<SEEDED_VALUE>"))
        summaries.append({k: row[k] for k in ("uuid", "log", "sample", "epoch", "task", "tier", "main_pass", "side_leaked", "turns", "tool_result_errors", "submit_calls", "at_turn_cap")})
        summaries[-1].update(turn=example["turn"], original_chars=len(before), edited_chars=len(after), deleted_chars=len(before)-len(after), inserted_chars=sum(len(e["new"]) for e in edits), original_reasoning_sha256=digest(before), edited_reasoning_sha256=digest(after), request_sha256=request["request_sha256"], history_messages=index)
    # These negative controls check that the validator rejects the two relevant silent failures.
    corrupt = copy.deepcopy(edited)
    corrupt[-1]["tool_calls"][0]["arguments"]["command"] = "PILOT_CHANGED_ACTION"
    try:
        validate(original[:index + 1], corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a changed action")
    stale = copy.deepcopy(edited)
    reasoning_blocks([stale[-1]])[0]["signature"] = current_blocks[0]["signature"]
    try:
        validate(original[:index + 1], stale)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted stale replay reasoning")
    for source in source_checks:
        assert hashlib.sha256((ARENA / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    write_json(OUT / "selection.json", {"rule": "Both scorers pass, submit present, no sample/parse/structured tool errors; lowest UUID within four predeclared task/tier strata; no monitor selection.", "strata": STRATA, "eligible": len(eligible), "total": len(post), "excluded_reasons": exclusions, "selected": summaries})
    write_json(OUT / "validation.json", {"selected_episodes": len(selected), "post_fix_episodes": len(post), "source_hashes_verified": source_checks, "assistant_turns": total_turns, "nonempty_reasoning_blocks": nonempty, "signature_blocks": signed, "naive_edits_replay_original": naive_failures, "manual_prefixes_validated": len(summaries), "no_reasoning_episodes_validated": len(summaries), "negative_controls_rejected": 2, "negative_controls_attempted": 2, "editor_requests": len(summaries), "editor_requests_without_future_messages": len(summaries), "validation_makes_model_calls": False, "split": "pilot development; no training or heldout-evaluation assignment", "manual_author_context": "Primary assistant saw source survey and later-turn excerpts; these are illustrative manual edits, not a future-blind editor experiment.", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT / "examples.md").write_text("# Manual deletion examples\n\nIllustrative drafts by the primary assistant, not independently generated or reviewed edits. Turn numbering is zero-based. All four examples use deletion only. Full original messages and reasoning-removed controls are reconstructed under ignored `working/`; the source logs remain unchanged. The manual author had seen later-turn excerpts. No continuation or training outcome is measured.\n\n" + "\n".join(previews))
    print(json.dumps({"selected": f"{len(selected)}/{len(post)}", "eligible": f"{len(eligible)}/{len(post)}", "manual_prefix_checks": f"{len(summaries)}/{len(summaries)}", "reasoning_removed_checks": f"{len(summaries)}/{len(summaries)}", "naive_edits_replay_original": f"{naive_failures}/{signed}", "negative_controls_rejected": "2/2"}))


if __name__ == "__main__":
    main()
