"""Build the math effort pool and a held-out split from effort_base.py --task math filter rows (base at low and medium).

Per problem: base accuracy at low and medium, and the parsed answers of every sample (math_data.parse_answer). Dropped:
  never solved   0 correct samples at low and at medium (an all-wrong group still gets a length gradient, which would teach
                 giving up early on hard problems, and gives no accuracy signal)
  suspect gold   every sample parsed to the same integer and it is not the gold (the clearest sign of a wrong gold; the pilot
                 grader check found 7 of 100 golds wrong or unanswerable)
The rest is shuffled (seed 0); --heldout problems go to the eval split, the remainder to the training pool. Pool rows: qid,
problem, answer, correct (base accuracy at medium, the plateau metric's baseline), low_acc, medium_cot (median tokens).

    python build_math_pool.py --rows ../results/math_filter/rows.jsonl --heldout 150 --out data/math_pool.jsonl --heldout-out data/math_heldout.jsonl
"""

import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import math_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--dapo", type=Path, default=Path(__file__).parent / "data/dapo_dedup.json")
    ap.add_argument("--heldout", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heldout-out", type=Path, required=True)
    args = ap.parse_args()
    dapo = {q["qid"]: q for q in json.load(open(args.dapo))}
    by = defaultdict(list)
    for r in map(json.loads, open(args.rows)):
        by[r["qid"]].append(r)
    keep, why = [], Counter()
    for qid, rs in by.items():
        q = dapo[qid]
        if {r["effort"] for r in rs} != {"low", "medium"}:
            why["incomplete (budget cut)"] += 1
            continue
        parsed = [math_data.parse_answer(r["response"]) for r in rs]
        acc = {e: st.mean(r["correct"] for r in rs if r["effort"] == e) for e in ("low", "medium")}
        if not any(r["correct"] for r in rs):
            why["never solved"] += 1
            continue
        if len(set(parsed)) == 1 and parsed[0] is not None and parsed[0] != q["answer"]:
            why["suspect gold"] += 1
            continue
        keep.append(dict(qid=qid, problem=q["problem"], answer=q["answer"], correct=acc["medium"], low_acc=acc["low"],
                         medium_cot=st.median(r["cot_tokens"] for r in rs if r["effort"] == "medium")))
        why["kept"] += 1
    random.Random(0).shuffle(keep)
    held, pool = keep[:args.heldout], keep[args.heldout:]
    for path, part in ((args.out, pool), (args.heldout_out, held)):
        with path.open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)
    need = sum(r["low_acc"] < r["correct"] for r in pool)
    print(dict(why), f"pool {len(pool)}, held-out {len(held)}; pool base acc low {st.mean(r['low_acc'] for r in pool):.3f} "
          f"medium {st.mean(r['correct'] for r in pool):.3f}; medium beats low on {need} ({need / len(pool):.0%})")


if __name__ == "__main__":
    main()
