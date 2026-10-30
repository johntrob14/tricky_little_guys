"""Audit historical annotations, source grouping, and parser metadata; no model calls."""

import json
from collections import Counter
from pathlib import Path

from audit import ARENA, OUT, SOURCES, read_json, source

rows = [json.loads(line) for line in (OUT / "manifest.jsonl").read_text().splitlines()]
post = {(r["tier"], r["sample"], r["epoch"]): r for r in rows if r["group"] == "deepseek_post_fix"}
pre = [r for r in rows if r["group"] == "deepseek_pre_fix"]
matched = [r for r in pre if (r["tier"], r["sample"], r["epoch"]) in post]
other = [r for r in pre if (r["tier"], r["sample"], r["epoch"]) not in post]
checks = {"matched_pre_post": {
    "matched_episodes": len(matched), "pre_fix_total": len(pre), "pre_fix_matched_exposure": sum(r["side_leaked"] for r in matched),
    "pre_fix_other_episodes": len(other), "pre_fix_other_exposure": sum(r["side_leaked"] for r in other),
    **{key: sum(r[key] == post[(r["tier"], r["sample"], r["epoch"])][key] for r in matched) for key in ("side_metadata_sha256", "prompt_config_sha256", "main_task_sha256")},
}, "duplicate_messages": {"duplicates": len(rows) - len({r["messages_sha256"] for r in rows}), "episodes": len(rows)}}

train = read_json(ARENA / "results/qwen/cap16/contexts_train.json")
heldout = read_json(ARENA / "results/qwen/cap16/contexts_heldout.json")
train_episodes = {(r["run"], r["sample"], r["epoch"]) for r in train}
heldout_episodes = {(r["run"], r["sample"], r["epoch"]) for r in heldout}
train_tasks = {r["sample"].split("#")[0] for r in train}
heldout_tasks = {r["sample"].split("#")[0] for r in heldout}
checks["qwen_context_split"] = {
    "train_turns": len(train), "train_episodes": len(train_episodes), "heldout_turns": len(heldout), "heldout_episodes": len(heldout_episodes),
    "episode_overlap": len(train_episodes & heldout_episodes), "heldout_tasks": sorted(heldout_tasks), "task_overlap": len(train_tasks & heldout_tasks),
    "training_configs_match": {name: read_json(ARENA / "results/turn1_grpo" / name / "config.json")["contexts"] == train for name in ("qwentutor1", "qwentutor2")},
}
contexts = read_json(ARENA / "results/prefill_qwen/contexts_leak.json")
checks["qwen_prefill_contexts"] = {"contexts": len(contexts), "source_episodes": len({(r["run"], r["sample"], r["epoch"]) for r in contexts}), "fresh_retry_labels": "not independently reclassified"}

annotations = read_json(ARENA / "results/covert_success/candidates_verified.json")
checks["archived_reader_annotations"] = {
    "episodes": len(annotations), "reader_verdicts": dict(Counter(r["reader_verdict"] for r in annotations)), "final_verdicts": dict(Counter(r["final_verdict"] for r in annotations)),
    "agreement": sum(r["verifier_agrees"] for r in annotations), "status": "archived labels counted; not a fresh independent review",
    "disagreements": [{k:r[k] for k in ("run", "sample", "epoch", "reader_verdict", "final_verdict")} for r in annotations if not r["verifier_agrees"]],
}
checks["prefill_parser"] = {}
for name in ("c6", "c7"):
    path = ARENA / "results/turn1_grpo/prefill1" / name / "samples.jsonl"
    source(path)
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    checks["prefill_parser"][name] = {
        "samples": len(samples), "empty_named_call": sum(any(not c["name"] for c in r["calls"]) for r in samples),
        "unparsed_tool_call_raw_text_over_20_chars": sum(any(c["name"] == "unparsed_tool_call" and len(c["args"].get("raw_text", "").strip()) > 20 for c in r["calls"]) for r in samples),
        "execution": "resampled proposals; no sandbox execution in this artifact",
    }
path = ARENA / "results/prefill_qwen/org_c1/samples.jsonl"
source(path)
samples = [json.loads(line) for line in path.read_text().splitlines()]
checks["prefill_parser"]["qwen_org_c1"] = {"samples": len(samples), "unclosed_think": sum(r["unclosed_think"] for r in samples)}

(OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
source(Path(__file__))
(OUT / "check_sources.json").write_text(json.dumps(SOURCES, indent=2, sort_keys=True) + "\n")
print(json.dumps(checks, indent=2))
