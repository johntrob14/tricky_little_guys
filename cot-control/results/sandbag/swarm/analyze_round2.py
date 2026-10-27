"""Summarize blinded reads with row and question-weighted calibration."""

import json
import math
import random
from collections import defaultdict
from statistics import mean

from prepare_round1 import MODELS
from prepare_round2 import OUT, SEED, read
from validate_round2 import LABELS


POOLS = ["unstated_flip", "cotctl_flip", "natural_parsed", "natural_unparsed", "redacted_positive"]


def fraction(count: int, total: int) -> dict:
    assert 0 <= count <= total and total > 0
    rate = count / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"count": count, "total": total, "rate": rate, "wilson95": [max(0, center - half), min(1, center + half)]}


def question_summary(rows: list[dict], field: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row[field])
    values = [mean(grouped[qid]) for qid in sorted(grouped)]
    rng = random.Random(SEED)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {"questions": len(values), "rows": len(rows), "mean": mean(values), "question_bootstrap95": [samples[49], samples[1949]]}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"paired_rows": 0, "metrics": None}
    n = len(rows)
    labels = {}
    for label in sorted(LABELS):
        flags = [(label in row["pass0"]["labels"], label in row["pass1"]["labels"]) for row in rows]
        labels[label] = {"both": fraction(sum(a and b for a, b in flags), n), "either": fraction(sum(a or b for a, b in flags), n), "agreement": fraction(sum(a == b for a, b in flags), n)}
    return {
        "paired_rows": n,
        "labels": labels,
        "mean_deliberate_score": mean(row["score"] for row in rows),
        "pass_mean_deliberate_score": {p: mean(row[p]["deliberate_score"] for row in rows) for p in ["pass0", "pass1"]},
        "mean_absolute_score_difference": mean(abs(row["pass0"]["deliberate_score"] - row["pass1"]["deliberate_score"]) for row in rows),
        "terminal_conclusion_agreement": fraction(sum(row["pass0"]["cot_conclusion"] == row["pass1"]["cot_conclusion"] for row in rows), n),
        "question_weighted": {field: question_summary(rows, field) for field in ["both_planted", "either_planted", "score"]},
        "items": [row["item"] for row in rows],
    }


def main() -> None:
    validation = json.loads((OUT / "validation.json").read_text())
    assert all(v["annotated"] + v["excluded"] == v["assigned"] for v in validation.values()), "Complete validation is required"
    key = json.loads((OUT / "key.json").read_text())
    assignments = json.loads((OUT / "assignments.json").read_text())
    passes = {p: {row["item"]: row for shard in assignments[p] for row in read(OUT / p / f"{shard}.jsonl")} for p in ["pass0", "pass1"]}
    pairs = []
    for identifier in sorted(passes["pass0"].keys() & passes["pass1"].keys()):
        row = {"blinded_item": identifier, **key[identifier], "pass0": passes["pass0"][identifier], "pass1": passes["pass1"][identifier]}
        left, right = set(row["pass0"]["labels"]), set(row["pass1"]["labels"])
        row["both_planted"] = "planted_candidate" in left & right
        row["either_planted"] = "planted_candidate" in left | right
        row["score"] = mean([row["pass0"]["deliberate_score"], row["pass1"]["deliberate_score"]])
        row["either_explicit"] = "explicit_wrong_intent" in left | right
        row["either_defect"] = bool({"question_issue", "reference_issue"} & (left | right))
        pairs.append(row)
    result = {"coverage": {p: {"read": len(passes[p]), "assigned": len(key)} for p in passes}, "paired": {"read": len(pairs), "assigned": len(key)}, "groups": {}}
    for model in MODELS:
        for pool in POOLS:
            rows = [row for row in pairs if row["model"] == model and row["pool"] == pool]
            eligible = sum(row["model"] == model and row["pool"] == pool for row in key.values())
            result["groups"][f"{model}/{pool}"] = {
                "eligible": eligible,
                "all": summarize(rows),
                "exclude_q212": summarize([row for row in rows if row["qid"] != 212]),
                "exclude_flagged_defects": summarize([row for row in rows if not row["either_defect"]]),
            }
            if pool == "unstated_flip":
                result["groups"][f"{model}/{pool}"]["exclude_reader_explicit_intent"] = summarize([row for row in rows if not row["either_explicit"]])
    result["contrasts"] = {}
    for model in MODELS:
        natural = result["groups"][f"{model}/natural_parsed"]["all"]
        for pool in ["unstated_flip", "cotctl_flip", "redacted_positive"]:
            target = result["groups"][f"{model}/{pool}"]["all"]
            if not natural["paired_rows"] or not target["paired_rows"]:
                result["contrasts"][f"{model}/{pool}-natural_parsed"] = None
                continue
            row_difference = {rule: target["labels"]["planted_candidate"][rule]["rate"] - natural["labels"]["planted_candidate"][rule]["rate"] for rule in ["both", "either"]}
            row_difference["score"] = target["mean_deliberate_score"] - natural["mean_deliberate_score"]
            question_difference = {field: target["question_weighted"][field]["mean"] - natural["question_weighted"][field]["mean"] for field in ["both_planted", "either_planted", "score"]}
            result["contrasts"][f"{model}/{pool}-natural_parsed"] = {"row_weighted": row_difference, "question_weighted": question_difference}
    result["candidate_ids"] = {"both_planted": [row["item"] for row in pairs if row["both_planted"]], "either_planted": [row["item"] for row in pairs if row["either_planted"]]}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"coverage": result["coverage"], "paired": result["paired"], "candidate_counts": {name: len(ids) for name, ids in result["candidate_ids"].items()}}, indent=2))


if __name__ == "__main__":
    main()
