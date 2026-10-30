"""Prepare fixed-source controls and blinded inputs for the approved second round."""

import hashlib
import json
import random
from pathlib import Path

from prepare_round1 import MODELS, balanced_sample, category


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "round2"
SEED = 20261028


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def source_rows() -> list[dict]:
    expected = json.loads((ROOT / "round1/sample.json").read_text())["source_sha256"]
    assert hashlib.sha256((ROOT / "items.jsonl").read_bytes()).hexdigest() == expected, "Source changed"
    return read(ROOT / "items.jsonl")


def eligible_rows() -> list[dict]:
    excluded = {row["item"] for row in read(ROOT / "unreadable.jsonl")}
    questions = {int(qid) for qid in json.loads((OUT / "question_exclusions.json").read_text())}
    return [row for row in source_rows() if row["item"] not in excluded and row["qid"] not in questions]


def redacted(cot: str, deletions: list[dict]) -> str:
    parts = []
    cursor = 0
    for span in deletions:
        start, end = span["start"], span["end"]
        assert type(start) is int and type(end) is int and cursor <= start < end <= len(cot)
        parts.append(cot[cursor:start])
        cursor = end
    return "".join(parts) + cot[cursor:]


def main() -> None:
    assert not (OUT / "candidate_queue.json").exists(), "Round 2 preparation already exists"
    for directory in [OUT / "inputs/redaction", OUT / "redaction", OUT / "pass0", OUT / "pass1"]:
        directory.mkdir(parents=True, exist_ok=True)
    reader = (ROOT / "round1/round2_prompt_proposed.md").read_text().split("\n---\n\n", 1)[1]
    prompts = (ROOT / "round1/round2_redaction_prompts_proposed.md").read_text()
    redactor = prompts.split("## Redactor\n\n", 1)[1].split("\n## Independent checker", 1)[0].strip() + "\n"
    checker = prompts.split("## Independent checker\n\n", 1)[1].strip() + "\n"
    for name, prompt in [("reader", reader), ("redactor", redactor), ("checker", checker)]:
        (OUT / f"{name}_prompt.md").write_text(prompt)
    (OUT / "question_exclusions.json").write_text(json.dumps({"2899": "Repeated Round 1 safety stops; excluded before Round 2 approval."}, indent=2) + "\n")
    rows = eligible_rows()
    rng = random.Random(SEED)
    queue = {}
    for model in MODELS:
        pool = [row for row in rows if row["model"] == model and category(row) == "stated_flip"]
        ordered = balanced_sample(pool, len(pool), rng)
        queue[model] = [row["item"] for row in ordered]
        write_rows(OUT / "inputs/redaction" / f"{model}_00.jsonl", [{"item": row["item"], "cot": row["cot"]} for row in ordered[:20]])
    (OUT / "candidate_queue.json").write_text(json.dumps(queue, indent=2) + "\n")
    approval = {"date": "2026-10-27", "authorization": "User: approved.", "seed": SEED, "source_sha256": hashlib.sha256((ROOT / "items.jsonl").read_bytes()).hexdigest(), "prompt_sha256": {name: hashlib.sha256((OUT / f"{name}_prompt.md").read_bytes()).hexdigest() for name in ["reader", "redactor", "checker"]}, "positive_target_per_model": 10, "first_candidate_batch_per_model": 20, "reader_execution": "fresh native Codex subagents; inherited model; exact underlying model not exposed", "new_sampling": False, "external_paid_judging": False}
    (OUT / "approval.json").write_text(json.dumps(approval, indent=2) + "\n")
    core = [row for row in rows if category(row) in ["unstated_flip", "cotctl_flip"] or row["kind"] == "resample_error"]
    assert len(core) == 622
    print(json.dumps({"eligible_core": len(core), "source_core": 636, "candidate_batch": {m: {"selected": 20, "eligible": len(ids)} for m, ids in queue.items()}}))


if __name__ == "__main__":
    main()
