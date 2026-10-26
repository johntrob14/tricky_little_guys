"""Blinded hand-read set for the GPQA sandbagging flips (results/sandbag_gpqa). Items, each one CoT shown with the question, the correct letter and the
submitted answer, in a random order, with no model or arm:
  - every flip of manytypes1b s150 (honest answer correct, this arm's answer parsed and wrong) in the sandbag, aware and evasion arms that the
    verbalization judge did not mark YES
  - every such flip of the other CoT models
  - controls: --ctl random flips of mt1b's cotctl_mt1b arm (a CoT-control instruction, no sandbag instruction), every mt1b question answered correctly in
    the honest arm and wrongly in its second honest sample (honest2; plain resampling errors), and --stated random verbalized flips (base and mt1b)
Writes <out>/key.json (item -> run, arm, qid, source) and <out>/shard<i>.jsonl.

    python sandbag_gpqa_handval.py --out ../results/sandbag_gpqa/handval
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results/sandbag_gpqa"
MODELS = ["base", "grpo3-s100", "mt1b-s150", "manytypes2c-s140", "manytypes2d-s140", "manytypes2e-s170"]
ARMS = ["sandbag", "aware", "evasion"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ctl", type=int, default=30)
    ap.add_argument("--stated", type=int, default=10)
    ap.add_argument("--shards", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    qs = {q["qid"]: q for q in map(json.loads, open(Path(__file__).parent / "data/gpqa_sandbagging.jsonl"))}
    picks, verbalized = [], []
    for m in MODELS:
        R = {(r["qid"], r["arm"]): r for r in map(json.loads, open(D / m / "rows.jsonl"))}
        V = {(r["qid"], r["arm"]): r["verbalized"] for r in map(json.loads, open(D / m / "verbalized.jsonl"))}
        flip = lambda q, a: (q, a) in R and R[(q, "honest")]["answer"] == R[(q, a)]["correct"] and R[(q, a)]["answer"] not in (None, R[(q, a)]["correct"])
        for a in ARMS:
            for q in qs:
                if flip(q, a):
                    (verbalized if V.get((q, a)) is True else picks).append((m, a, q, "flip_verbalized" if V.get((q, a)) is True else "flip_not_verbalized", R[(q, a)]))
        if m == "mt1b-s150":
            ctl = [q for q in qs if flip(q, "cotctl_mt1b")]
            picks += [(m, "cotctl_mt1b", q, "control_cotctl", R[(q, "cotctl_mt1b")]) for q in rng.sample(ctl, min(args.ctl, len(ctl)))]
            H2 = {r["qid"]: r for r in map(json.loads, open(D / f"{m}-honest2" / "rows.jsonl"))}
            picks += [(m, "honest2", q, "control_resample", H2[q]) for q in qs if R[(q, "honest")]["answer"] == H2[q]["correct"] and H2[q]["answer"] not in (None, H2[q]["correct"])]
    picks += rng.sample([v for v in verbalized if v[0] in ("base", "mt1b-s150")], args.stated)
    rng.shuffle(picks)
    args.out.mkdir(parents=True, exist_ok=True)
    key, shards = [], [[] for _ in range(args.shards)]
    for i, (m, a, q, src, r) in enumerate(picks):
        item = f"G{i:03d}"
        key.append(dict(item=item, run=m, arm=a, qid=q, source=src))
        shards[i % args.shards].append(dict(item=item, question=qs[q]["honest"], correct=qs[q]["correct"], submitted=r["answer"], cot=r["reasoning"], final=r["response"]))
    json.dump(key, open(args.out / "key.json", "w"), indent=1)
    for i, s in enumerate(shards):
        (args.out / f"shard{i}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in s))
    print(len(picks), "items", Counter((k["run"], k["arm"], k["source"]) for k in key))


if __name__ == "__main__":
    main()
