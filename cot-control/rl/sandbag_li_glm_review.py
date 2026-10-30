"""Validate independent GLM trace reviews, verify evidence, and merge annotations for the viewer."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results/sandbag_li"
REVIEW = ROOT / "glm_review"
FIELDS = {"id", "model", "qid", "condition", "reviewer", "title", "note", "tags", "intent", "confidence", "read_scope", "monitor_assessment", "interestingness", "evidence"}


def main():
    selection = json.loads((REVIEW / "selection.json").read_text())
    for relative, expected in selection["source_sha256"].items():
        path = ROOT.parents[1] / relative if relative.startswith("rl/") else ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Review source changed: {path}"
    rows = {}
    for model in ("glm-5.1", "glm-5.3"):
        for row in map(json.loads, (ROOT / model / "rows.jsonl").read_text().splitlines()):
            rows[f"{model}:{row['qid']}:{row['condition']}"] = row
    reviews = {}
    for model, reviewer in (("glm-5.1", "review_glm51"), ("glm-5.3", "review_glm53")):
        assignment = json.loads((REVIEW / f"assignment_{model}.json").read_text())
        annotations = [json.loads(line) for line in (REVIEW / f"{reviewer}.jsonl").read_text().splitlines()]
        assert len(annotations) == len(assignment)
        assert {a["id"] for a in annotations} == {a["id"] for a in assignment}
        for a in annotations:
            assert FIELDS <= a.keys(), a["id"]
            assert a["id"] == f"{a['model']}:{a['qid']}:{a['condition']}"
            assert a["reviewer"] == reviewer
            assert a["intent"] in ("explicit", "ambiguous", "none_observed")
            assert a["confidence"] in ("high", "medium", "low")
            assert a["read_scope"] in ("full_reasoning_and_final", "behavior_excerpts")
            assert a["monitor_assessment"] in ("supported_flag", "possible_miss", "likely_false_alarm", "unscored", "not_assessed")
            assert a["interestingness"] in (1, 2, 3)
            assert a["title"] and a["note"] and a["tags"]
            assert len(a["tags"]) == len(set(a["tags"]))
            assert len(a["evidence"]) <= 3
            for e in a["evidence"]:
                assert e["channel"] in ("reasoning", "response")
                assert 0 < len(e["quote"]) <= 220
                assert e["quote"] in rows[a["id"]][e["channel"]], f"Evidence mismatch: {a['id']}"
            reviews.setdefault(a["id"], []).append(a)
    merged = []
    for item in selection["items"]:
        entries = reviews[item["id"]]
        own_reviewer = "review_glm51" if item["model"] == "glm-5.1" else "review_glm53"
        primary = next(a for a in entries if a["reviewer"] == own_reviewer).copy()
        if len(entries) == 2:
            primary["second_review"] = next(a for a in entries if a["reviewer"] != own_reviewer)
        else:
            assert len(entries) == 1
        merged.append(primary)
    (REVIEW / "annotations.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    stats = dict(reviewed=len(merged), available=len(rows), reader_assessments=sum(map(len, reviews.values())), cross_read=sum(len(v) == 2 for v in reviews.values()),
                 tags=dict(Counter(t for a in merged for t in a["tags"])), intents=dict(Counter(a["intent"] for a in merged)),
                 scopes=dict(Counter(a["read_scope"] for a in merged)), confidence=dict(Counter(a["confidence"] for a in merged)))
    (REVIEW / "review_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
