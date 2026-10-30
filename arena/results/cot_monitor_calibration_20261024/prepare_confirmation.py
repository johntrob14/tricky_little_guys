"""Freeze a new audit from edited Qwen/DeepSeek traces, excluding all episodes in the first pool."""

import gzip
import json
import random
import zipfile
from collections import Counter

from prepare import ARENA, OUT, digest, rows_from, write_json


def main() -> None:
    source = ARENA / "results/deepseek_cleaned_followup_dekallm16"
    with gzip.open(OUT / "pool.jsonl.gz", "rt") as file:
        excluded = {json.loads(line)["uuid"] for line in file}
    strata = {"high": [], "mid": [], "low": []}
    for line in (source / "dataset.jsonl").read_text().splitlines():
        episode = json.loads(line)
        if episode["uuid"] in excluded:
            continue
        metadata = {k: episode["source"][k] for k in ["model", "task", "log", "sample", "epoch"]}
        scores = {r["turn"]: r for r in json.loads((source / "working" / episode["uuid"] / "scores_after.json").read_text())}
        for row in rows_from(episode["messages"], {**metadata, "uuid": episode["uuid"], "cohort": "confirmation"}):
            old = scores[row["turn"]]
            assert old["prompt_sha256"] == row["baseline_prompt_sha256"]
            assert old["mode"] == "cot_only" and old["main_task_only"] and old["input_channels"] == ["reasoning"]
            if old["score"] is None:
                continue
            stratum = "high" if old["score"] > 5 else "mid" if old["score"] > 2 else "low"
            strata[stratum].append({**row, "stratum": stratum, "old_score": old["score"], "old_input_exact": True})
    rng = random.Random(2026102502)
    selected = []
    for stratum, n in [("high", 16), ("mid", 8), ("low", 8)]:
        selected.extend(rng.sample(strata[stratum], min(n, len(strata[stratum]))))
    original_pairs = []
    pair_indices = [1, 6, 13, 15, 19, 20, 22, 28]
    for index in pair_indices:
        edited = selected[index]
        with zipfile.ZipFile(ARENA / edited["log"]) as archive:
            samples = [json.loads(archive.read(name)) for name in archive.namelist() if name.startswith("samples/") and name.endswith(".json")]
        sample = next(s for s in samples if s["uuid"] == edited["uuid"])
        metadata = {k: edited[k] for k in ["model", "task", "log", "sample", "epoch", "uuid"]}
        row = next(r for r in rows_from(sample["messages"], {**metadata, "cohort": "confirmation_original"}) if r["turn"] == edited["turn"])
        original_pairs.append({**row, "paired_edited_id": edited["id"]})
    all_rows = selected + original_pairs
    (OUT / "confirmation.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
    write_json("confirmation_selection.json", {"seed": 2026102502, "source": str(source.relative_to(ARENA)), "dataset_sha256": digest((source / "dataset.jsonl").read_text()), "excluded_episode_ids_from_initial_pool": len(excluded), "population_by_stratum": {k: len(v) for k, v in strata.items()}, "selected_by_stratum": dict(Counter(r["stratum"] for r in selected)), "original_pair_indices": pair_indices, "models": dict(Counter(r["model"] for r in all_rows)), "selection": "Independent saved episode identities and another cleanup dataset. Score-stratified edited-text selection before reading text or new judgments. Eight original counterparts selected after reading the edited excerpts, before reading originals or any new judgments; these are a targeted sensitivity check, not a random recall estimate. No new prompt wording is fitted to these examples."})
    print(json.dumps({"population": {k: len(v) for k, v in strata.items()}, "selected": len(selected)}))


if __name__ == "__main__":
    main()
