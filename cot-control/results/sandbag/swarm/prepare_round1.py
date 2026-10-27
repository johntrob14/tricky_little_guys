"""Prepare the approved Round 1 sample without calling any model APIs."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEED = 20261027
MODELS = ["base", "grpo3-s100", "mt1b-s150", "manytypes2e-s170"]
FIELDS = ["item", "model", "variant", "arm", "run", "qid", "kind", "system", "user", "cot", "response", "answer", "correct", "reference_cot", "finish", "cot_tokens", "ctl_type"]
QUOTAS = {
    "base": {"unstated_flip": 26, "stated_flip": 10, "instructed_held": 12, "cotctl_flip": 8, "cotctl_held": 4, "honest_correct": 4},
    "grpo3-s100": {"unstated_flip": 26, "stated_flip": 10, "instructed_held": 12, "cotctl_flip": 8, "cotctl_held": 4, "honest_correct": 4},
    "mt1b-s150": {"unstated_flip": 52, "stated_flip": 10, "instructed_held": 14, "cotctl_flip": 12, "cotctl_held": 4, "honest_correct": 4},
    "manytypes2e-s170": {"unstated_flip": 39, "stated_flip": 12, "instructed_held": 25, "cotctl_flip": 12, "cotctl_held": 4, "honest_correct": 4},
}


def category(row: dict) -> str | None:
    if row["arm"] in ("sandbag", "aware", "evasion"):
        if row["kind"] == "flip":
            return "stated_flip" if row.get("states_wrong_intent") is True else "unstated_flip"
        if row["kind"] == "held":
            return "instructed_held"
    if row["arm"] == "cotctl" and row["kind"] in ("flip", "held"):
        return "cotctl_" + row["kind"]
    if row["kind"] == "honest_correct":
        return "honest_correct"
    return None


def balanced_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        strata[row["variant"], row["arm"]].append(row)
    for pool in strata.values():
        rng.shuffle(pool)
    keys = sorted(strata)
    rng.shuffle(keys)
    selected = []
    assert len(rows) >= count
    while len(selected) < count:
        for key in keys:
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop())
    return selected


def write_input(name: str, rows: list[dict]) -> None:
    path = ROOT / "round1/inputs" / f"{name}.jsonl"
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps({key: row[key] for key in FIELDS if key in row}) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    source = ROOT / "items.jsonl"
    rows = [json.loads(line) for line in source.open()]
    assert len(rows) == len({row["item"] for row in rows}) == 7350
    (ROOT / "round1/inputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "round1/readers").mkdir(exist_ok=True)
    manifest = {"seed": SEED, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_rows": len(rows), "quotas": QUOTAS, "round1a": {}, "round1b": {}, "selection": {}}
    natural = {}
    selected = []
    for model in MODELS:
        pool = [row for row in rows if row["model"] == model]
        resamples = [row for row in pool if row["kind"] == "resample_error"]
        wrong_by_question = defaultdict(list)
        for row in pool:
            if row["kind"] == "honest_wrong":
                wrong_by_question[row["qid"]].append(row)
        questions = rng.sample(sorted(wrong_by_question), 20)
        wrong = [rng.choice(wrong_by_question[qid]) for qid in questions]
        natural[model] = sorted(resamples + wrong, key=lambda row: (row["qid"], row["item"]))
        manifest["round1a"][model] = [row["item"] for row in natural[model]]
        for kind, count in QUOTAS[model].items():
            candidates = [row for row in pool if category(row) == kind]
            if model == "manytypes2e-s170" and kind == "instructed_held":
                target = [row for row in candidates if row["variant"] == "should" and row["arm"] == "evasion"]
                others = [row for row in candidates if row not in target]
                sample = balanced_sample(target, 15, rng) + balanced_sample(others, count - 15, rng)
            else:
                sample = balanced_sample(candidates, count, rng)
            selected.extend(sample)
            manifest["selection"][model + "/" + kind] = {"selected": len(sample), "available": len(candidates)}
    write_input("natural_base", natural["base"])
    write_input("natural_mt1b", natural["mt1b-s150"])
    write_input("natural_grpo_mt2e", natural["grpo3-s100"] + natural["manytypes2e-s170"])
    assert len(selected) == len({row["item"] for row in selected}) == 320
    rng.shuffle(selected)
    for index in range(8):
        name = f"reader{index:02d}"
        sample = selected[index * 40:(index + 1) * 40]
        write_input(name, sample)
        manifest["round1b"][name] = [row["item"] for row in sample]
    (ROOT / "round1/sample.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = (ROOT / "README.md").read_text()
    brief = readme.split("Reader brief for 1a/1b (the text given to each reader, plus its items):\n\n", 1)[1].split("\n\nThe orchestrator", 1)[0]
    (ROOT / "round1/reader_prompt.md").write_text("\n".join(line.removeprefix("> ") for line in brief.splitlines()) + "\n")
    print(json.dumps({"natural": {m: dict(Counter(r["kind"] for r in rs)) for m, rs in natural.items()}, "stratified": manifest["selection"], "source_sha256": manifest["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
