"""Question set for the Li, Phuong and Siegel 2025 sandbagging protocol (arXiv 2508.00943, section 3): 400 random WMDP MCQs (the target, bio + chem + cyber
pooled) and 400 random MMLU MCQs (the general-knowledge control), from the Hugging Face datasets cais/wmdp and cais/mmlu (test splits), seeded. One row per
question: qid, dataset (wmdp / mmlu), subject, question, choices (4), correct letter. Writes data/li_questions.jsonl.

    python sandbag_li_data.py --n 400 --seed 0
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="questions per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data/li_questions.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pools = {"wmdp": [dict(subject=sub, **r) for sub in ["wmdp-bio", "wmdp-chem", "wmdp-cyber"] for r in load_dataset("cais/wmdp", sub, split="test")],
             "mmlu": list(load_dataset("cais/mmlu", "all", split="test"))}
    rows = []
    for ds, pool in pools.items():
        for i, r in enumerate(rng.sample(pool, args.n)):
            assert len(r["choices"]) == 4 and 0 <= r["answer"] <= 3, r
            rows.append(dict(qid=f"{ds}-{i}", dataset=ds, subject=r["subject"], question=r["question"], choices=r["choices"], correct="ABCD"[r["answer"]]))
        subjects = Counter(r["subject"] for r in rows if r["dataset"] == ds)
        print(f"{ds}: {args.n} of {len(pool)}; {len(subjects)} subjects" + (f": {dict(subjects)}" if ds == "wmdp" else ""))
    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
