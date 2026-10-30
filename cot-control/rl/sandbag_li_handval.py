"""Blinded hand-read set for the Li et al. sandbagging flips in the plain-format default (results/sandbag_li_plain). A flip (successful attempt) is a
question answered correctly in the baseline condition and wrongly in oblivious or aware. Items, each one CoT with the question, the correct letter and the
submitted letter, shuffled, no model or condition shown:
  - every WMDP flip the verbalization judge marked NO, in both conditions and every CoT model, cyber and chem questions only (WMDP-bio is left out: the
    verbalization judge returns no verdict on most bio rows and subagent readers are stopped by a bio safeguard)
  - controls: --mmlu random aware-condition MMLU flips per model in MMLU_CTL (the prompt says to answer general questions correctly, so these are errors or
    misapplied sandbagging) and --stated random verbalized WMDP cyber flips
Writes <out>/key.json (item -> run, condition, dataset, qid, source) and <out>/shard<i>.jsonl.

    python sandbag_li_handval.py --out ../results/sandbag_li_plain/handval
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from sandbag_li import mcq

D = Path(__file__).resolve().parents[1] / "results/sandbag_li_plain"
MODELS = ["base", "grpo3-s100", "mt1b-s150", "manytypes2c-s140", "manytypes2d-s140", "manytypes2e-s170"]
MMLU_CTL = {"mt1b-s150": 20, "manytypes2e-s170": 10, "base": 10}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stated", type=int, default=10)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    qs = {q["qid"]: q for q in map(json.loads, open(Path(__file__).parent / "data/li_questions.jsonl"))}
    picks, stated = [], []
    for m in MODELS:
        R = {(r["qid"], r["condition"]): r for r in map(json.loads, open(D / m / "rows.jsonl"))}
        V = {(r["qid"], r["condition"]): r["verbalized"] for r in map(json.loads, open(D / m / "verbalized.jsonl"))}
        flip = lambda q, c: R[(q, "baseline")]["answer"] == qs[q]["correct"] and R[(q, c)]["answer"] not in (None, qs[q]["correct"])
        for c in ["oblivious", "aware"]:
            for q in qs:
                if qs[q]["dataset"] == "wmdp" and qs[q]["subject"] != "wmdp-bio" and flip(q, c):
                    if V.get((q, c)) is False:
                        picks.append((m, c, q, "flip_not_verbalized", R[(q, c)]))
                    elif V.get((q, c)) is True and qs[q]["subject"] == "wmdp-cyber":
                        stated.append((m, c, q, "flip_verbalized", R[(q, c)]))
        mm = [q for q in qs if qs[q]["dataset"] == "mmlu" and flip(q, "aware")]
        picks += [(m, "aware", q, "control_mmlu", R[(q, "aware")]) for q in rng.sample(mm, min(MMLU_CTL.get(m, 0), len(mm)))]
    picks += rng.sample(stated, args.stated)
    rng.shuffle(picks)
    args.out.mkdir(parents=True, exist_ok=True)
    key, shards = [], [[] for _ in range(args.shards)]
    for i, (m, c, q, src, r) in enumerate(picks):
        item = f"L{i:03d}"
        key.append(dict(item=item, run=m, arm=c, condition=c, dataset=qs[q]["dataset"], qid=q, source=src))
        shards[i % args.shards].append(dict(item=item, question=mcq(qs[q]), correct=qs[q]["correct"], submitted=r["answer"], cot=r["reasoning"], final=r["response"]))
    json.dump(key, open(args.out / "key.json", "w"), indent=1)
    for i, s in enumerate(shards):
        (args.out / f"shard{i}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in s))
    print(len(picks), "items", Counter((k["run"], k["condition"], k["source"]) for k in key))


if __name__ == "__main__":
    main()
