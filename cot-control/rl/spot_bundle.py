"""Build a spot-check bundle for one block of a grpo_effort.py run: rollouts for reviewers to read, per level.

Per level, from steps (step - block, step]: 6 random rollouts, the 2 shortest and 2 longest correct, 3 wrong ones, and up to 3
flagged by the trainer's signals (looping, unparsed, non-Latin, empty CoT). Each CoT is cut to its first 2,500 and last 600
characters. Also 4 random base rollouts per level on pool questions (results/effort_base) as a style reference, and the
block's per-level signal means against the run's first 5 steps.

    python spot_bundle.py --run eff2 --step 10 --out ../results/rl/eff2/spot/step10.json
"""

import argparse
import json
import random
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
EFFORTS = ["low", "medium", "high"]


def cut(t: str) -> str:
    return t if len(t) <= 3200 else t[:2500] + f"\n[... {len(t) - 3100} chars cut ...]\n" + t[-600:]


def slim(r: dict, why: str) -> dict:
    return dict(why=why, step=r["step"], qid=r["qid"], effort=r["effort"], cot_tokens=r["cot_tokens"], correct=r["correct"],
                reward=round(r["reward"], 4), flags={k: r.get(k) for k in ("looping", "parsed", "non_latin", "truncated")},
                reasoning=cut(r["reasoning"]), response=r["response"][:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-rows", type=Path, default=RESULTS / "effort_base/rows.jsonl", help="base rollouts for the style reference")
    ap.add_argument("--pool", type=Path, default=Path(__file__).parent / "data/pool_k4.jsonl", help="jsonl with qid and question or problem")
    args = ap.parse_args()
    rng = random.Random(args.step)
    rs = [json.loads(l) for l in open(RESULTS / "rl" / args.run / "rollouts.jsonl")]
    block = [r for r in rs if args.step - args.block < r["step"] <= args.step]
    first = [r for r in rs if r["step"] <= 5]
    base = [json.loads(l) for l in open(args.base_rows)]
    qtext = {r["qid"]: r.get("question") or r.get("problem", "") for r in map(json.loads, open(args.pool))}
    bundle = {"run": args.run, "step": args.step, "levels": {}}
    for e in EFFORTS:
        x = [r for r in block if r["effort"] == e]
        f = [r for r in first if r["effort"] == e]
        cor = sorted([r for r in x if r["correct"]], key=lambda r: r["cot_tokens"])
        flagged = [r for r in x if r.get("looping") or not r.get("parsed", True) or r.get("non_latin", 0) > 0.02 or r["cot_tokens"] == 0]
        picks = ([slim(r, "random") for r in rng.sample(x, min(6, len(x)))] + [slim(r, "shortest correct") for r in cor[:2]] +
                 [slim(r, "longest correct") for r in cor[-2:]] + [slim(r, "wrong") for r in rng.sample([r for r in x if not r["correct"]], min(3, sum(not r["correct"] for r in x)))] +
                 [slim(r, "flagged") for r in flagged[:3]])
        for p in picks:
            p["question"] = qtext.get(p["qid"], "")[:1200]
        stats = lambda y: dict(n=len(y), median_cot=st.median([r["cot_tokens"] for r in y]), correct=round(st.mean(r["correct"] for r in y), 3),
                               correct_minus_base=round(st.mean(r["correct"] - r["base_acc"] for r in y), 3),
                               cot_under_10=round(st.mean(r["cot_tokens"] < 10 for r in y), 3),
                               looping=round(st.mean(bool(r.get("looping")) for r in y), 3), unparsed=round(st.mean(not r.get("parsed", True) for r in y), 3),
                               mean_lp=round(st.mean(r["mean_lp"] for r in y if r.get("mean_lp") is not None), 3) if any(r.get("mean_lp") is not None for r in y) else None)
        bref = [dict(why="base reference", qid=r["qid"], effort=e, cot_tokens=r["cot_tokens"], correct=r["correct"], reasoning=cut(r["reasoning"]),
                     response=r["response"][:400], question=qtext.get(r["qid"], "")[:1200]) for r in rng.sample([r for r in base if r["effort"] == e], 4)]
        bundle["levels"][e] = {"block_stats": stats(x), "first5_stats": stats(f), "samples": picks, "base_reference": bref}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=1, ensure_ascii=False))
    print(args.out, {e: len(v["samples"]) for e, v in bundle["levels"].items()})


if __name__ == "__main__":
    main()
