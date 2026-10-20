"""Qwen3-8B RL-judge check with and without thinking: calibration pairs (100+100) and agreement with the Nemotron eval
judge on the first 512 rl1 rollouts (results/rlcheck/, labels made on zeus). Run alone on the GPU."""
import json, os, random, sys
import pandas as pd
from cb.train import Judge, ROOT



def main():
    think = sys.argv[1] == "think"
    d = pd.read_parquet(os.path.join(ROOT, "data", "train-00000-of-00001.parquet")).sample(n=100, random_state=42)
    pairs = []
    for r in d.itertuples():
        pairs += [("trigger", r.target_fact, r.chosen[-1]["content"]), ("control", r.target_fact, r.rejected[-1]["content"])]
    rl = [json.loads(l) for l in open(os.path.join(ROOT, "results", "rlcheck", "trigger.jsonl"))]
    nem = {json.loads(l)["key"]: json.loads(l) for l in open(os.path.join(ROOT, "results", "rlcheck", "trigger.labels.nemotron-super-nothink.jsonl"))}
    j = Judge("/root/models/qwen3-8b", 0.85, thinking=think)
    labs = j([p[2] for p in pairs] + [r["response"] for r in rl])
    cal, rll = labs[: len(pairs)], labs[len(pairs):]
    c = dict(tp=0, fp=0, tn=0, fn=0, nd=0)
    for (k, tf, _), l in zip(pairs, cal):
        if l is None: c["nd"] += 1; continue
        p = l.get(tf) == "detected"; c[("tp" if p else "fn") if k == "trigger" else ("fp" if p else "tn")] += 1
    a = dict(both=0, qwen_only=0, nem_only=0, neither=0, nd=0)
    for r, l in zip(rl, rll):
        if l is None: a["nd"] += 1; continue
        q = l.get(r["target_fact"]) == "detected"; n = nem[r["key"]]["labels"][r["target_fact"]] == "detected"
        a[("both" if n else "qwen_only") if q else ("nem_only" if n else "neither")] += 1
    res = dict(thinking=think, calibration=c, rl_vs_nemotron=a)
    print("JUDGECHECK", json.dumps(res), flush=True)
    open(os.path.join(ROOT, "results", "qer", f"qwen3_8b_check_{sys.argv[1]}.json"), "w").write(json.dumps(res))


if __name__ == "__main__":
    main()
