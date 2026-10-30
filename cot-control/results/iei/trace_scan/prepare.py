import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from mechtools import *

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
MODELS = ["base", "mt2e-s170", "mt1b-s150"]
SEED = 20261026
rng = random.Random(SEED)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


prompts = json.loads((DATA.parents[1] / "rl/data/iei/prompts.json").read_text())["prompts"]
bare = {p["scenario_id"]: p for p in prompts if p["constraint_id"] is None}
constraints = json.loads((DATA.parents[1] / "rl/data/iei/constraints.json").read_text())["constraints"]
hints = {c["id"]: c["natural_hint"] for c in constraints}
nudged = {model: {(r["arm"], r["prompt_id"]): r for r in read_rows(DATA / model / "nudged.jsonl")} for model in MODELS}
selected = []
sampled_cells = {}
for arm in ["implicit", "explicit"]:
    common = sorted(key for key in nudged["base"].keys() & nudged["mt2e-s170"].keys() if key[0] == arm)
    cells = rng.sample(common, 60)
    sampled_cells[arm] = {"sampled": len(cells), "eligible": len(common), "prompt_ids": [key[1] for key in cells]}
    for key in cells:
        for model in MODELS:
            if key in nudged[model]:
                selected.append((model, nudged[model][key]))
for model in MODELS:
    rows = [dict(r, arm="baseline", prompt_id=f"baseline:{r['scenario_id']}:{r['rep']}") for r in read_rows(DATA / model / "baseline.jsonl") if r["rep"] == 0]
    assert len(rows) == len(bare) == 35
    selected.extend((model, row) for row in sorted(rows, key=lambda r: r["scenario_id"]))
rng.shuffle(selected)

items = []
manifest = []
for i, (model, row) in enumerate(selected):
    item_id = f"t{i:04d}"
    options = bare[row["scenario_id"]]
    final = row["response"]
    excerpt = final if len(final) <= 2000 else final[:800] + "\n\n[... middle omitted ...]\n\n" + final[-1200:]
    item = dict(id=item_id, arm=row["arm"], options={"A": options["path_a_name"], "B": options["path_b_name"]}, user=row["user"],
                nudge=None if row["arm"] == "baseline" else f"{row['target_tool']} {hints[row['constraint_id']]}", cot=row["reasoning"], final_answer_excerpt=excerpt)
    items.append(item)
    manifest.append(dict(id=item_id, model=model, arm=row["arm"], scenario_id=row["scenario_id"], prompt_id=row["prompt_id"],
                         parser_choice=row["choice"], baseline_choice=row.get("baseline_choice"), recommend=row.get("recommend"),
                         switched=row.get("switched"), cot_tokens=row["cot_tokens"], response_chars=len(final), excerpt_truncated=len(final) > 2000,
                         primary_shard=i % 8))

shards = [items[i::8] for i in range(8)]
repeat_ids = rng.sample(range(len(items)), 50)
for i in repeat_ids:
    other = rng.choice([j for j in range(8) if j != i % 8])
    shards[other].append(items[i])
    manifest[i]["repeat_shard"] = other
for i, shard in enumerate(shards):
    rng.shuffle(shard)
    (ROOT / f"items_{i:02d}.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in shard))

handoff = (DATA / "TRACE_ANALYSIS_HANDOFF.md").read_text()
brief = handoff.split("Scorer brief:\n", 1)[1].split("\n## 6.", 1)[0]
brief = "\n".join(line[2:] if line.startswith("> ") else "" for line in brief.strip().splitlines()) + "\n"
assert brief.startswith("You are labeling") and "Label every item." in brief
(ROOT / "scorer_brief.txt").write_text(brief)
write_json(ROOT / "manifest.json", manifest)
sources = [DATA / model / name for model in MODELS for name in ["baseline.jsonl", "nudged.jsonl"]]
sources += [DATA.parents[1] / "rl/data/iei" / name for name in ["prompts.json", "constraints.json"]]
write_json(ROOT / "config.json", dict(seed=SEED, date="2026-10-26", sampled_cells=sampled_cells, n_unique=len(items), n_repeated=len(repeat_ids),
    shard_sizes=[len(s) for s in shards], model_arm_counts=dict(Counter(f"{r['model']}/{r['arm']}" for r in manifest)),
    scorer="Codex subagents, inherited session model; exact API model identifier unavailable", blind=True, context_fork="none",
    baseline_rep=0, final_excerpt={"head": 800, "tail": 1200, "full_if_chars_at_most": 2000},
    brief_sha256=hashlib.sha256(brief.encode()).hexdigest(),
    source_sha256={str(p.relative_to(DATA.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
print(json.dumps({"unique": len(items), "repeat": len(repeat_ids), "shard_sizes": [len(s) for s in shards], "model_arm_counts": dict(Counter(f"{r['model']}/{r['arm']}" for r in manifest))}, indent=2))
