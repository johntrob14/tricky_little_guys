"""Build a self-contained viewer for GLM Li-protocol responses and qualitative annotations.

    python sandbag_li_glm_viewer.py
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results/sandbag_li"
MODELS = ["glm-5.1", "glm-5.3"]
SCOPES = ["cot", "full", "final"]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def build(root: Path, annotations_path: Path, questions_path: Path) -> dict:
    annotations = json.loads(annotations_path.read_text())
    reviews = {a["id"]: a for a in annotations}
    assert len(reviews) == len(annotations), "Duplicate annotation IDs"
    rows, configs = [], {}
    for model in MODELS:
        directory = root / model
        configs[model] = json.loads((directory / "config.json").read_text())
        monitors = {scope: {(r["qid"], r["condition"]): r for r in jsonl(directory / f"monitor_{scope}.jsonl")} for scope in SCOPES}
        verbalized = {(r["qid"], r["condition"]): r for r in jsonl(directory / "verbalized.jsonl")}
        for row in jsonl(directory / "rows.jsonl"):
            key = row["qid"], row["condition"]
            row = row | {"id": f"{model}:{row['qid']}:{row['condition']}", "model": model}
            row["monitors"] = {scope: monitors[scope].get(key) for scope in SCOPES}
            row["verbalization"] = verbalized.get(key)
            rows.append(row)
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(rows), "Duplicate response IDs"
    for annotation in annotations:
        assert annotation["id"] in by_id, annotation["id"]
        row = by_id[annotation["id"]]
        for review in [annotation] + ([annotation["second_review"]] if annotation.get("second_review") else []):
            for evidence in review["evidence"]:
                assert evidence["quote"] in row[evidence["channel"]], f"Unmatched evidence: {annotation['id']}"
    qids = {r["qid"] for r in rows}
    questions = {q["qid"]: q for q in jsonl(questions_path) if q["qid"] in qids}
    assert questions.keys() == qids, "Missing question records"
    for row in rows:
        assert f"{row['model']}:{row['qid']}:baseline" in by_id, row["id"]
    return dict(rows=rows, annotations=reviews, questions=questions, configs=configs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--questions", type=Path, default=Path(__file__).parent / "data/li_questions.jsonl")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = build(args.dir, args.annotations or args.dir / "glm_review/annotations.json", args.questions)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = Path(__file__).with_suffix(".html").read_text().replace("__PAYLOAD__", payload)
    out = args.out or args.dir / "glm_review/viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"Reviewed {len(data['annotations'])}/{len(data['rows'])} responses; {out.stat().st_size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
