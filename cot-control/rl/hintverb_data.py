"""Build data/hintverb_questions.jsonl, the question set of hintverb.py, from the 498 questions released by Young et al. 2026
(github.com/ricyoung/cot-faithfulness-open-models, data/combined_500.jsonl, CC BY 4.0): 300 MMLU questions stratified over subjects and the
198 GPQA Diamond questions, all with 4 options A-D, the set their open-model replication of Chen et al. 2025 ran on (gpt-oss-120b included).

Adds `hint`, the hint answer h for the question (one letter drawn uniformly from A-D with --seed, shared by every hint type, so it points to
the correct option about a quarter of the time, as in Chen et al.), and `in_pool_k4`, true when the first 80 normalised characters of the
question text match a question in data/pool_k4.jsonl (the MMLU-Pro pool manytypes1b and grpo3 trained on; MMLU-Pro reuses MMLU questions).
hintverb.py skips those.

    python hintverb_data.py --young /path/to/combined_500.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()[:80]


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--young", type=Path, required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
pool = {norm(json.loads(l)["question"]) for l in open(DATA / "pool_k4.jsonl")}
rng = random.Random(args.seed)
rows = []
for l in open(args.young):
    y = json.loads(l)
    assert y["choice_labels"] == ["A", "B", "C", "D"], y["id"]
    rows.append(dict(qid=y["id"], dataset=y["dataset"], subject=y["subject"], question=y["question_text"], choices=y["choices"], correct=y["correct_label"],
                     hint=rng.choice("ABCD"), in_pool_k4=norm(y["question_text"]) in pool))
with open(DATA / "hintverb_questions.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} questions, {sum(r['in_pool_k4'] for r in rows)} in pool_k4, hint == correct for {sum(r['hint'] == r['correct'] for r in rows)}")
