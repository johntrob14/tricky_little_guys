"""Calibrate the RL train-time judge (Qwen3-8B, vLLM, thinking off) on 100 chosen (false fact present) + 100 rejected
(correct) responses from dpo-cake-bake train, as cb.qer calibrate does for the Tinker eval judge."""
import json, os, random
import pandas as pd
from cb.train import Judge, ROOT

d = pd.read_parquet(os.path.join(ROOT, "data", "train-00000-of-00001.parquet")).sample(n=100, random_state=42)
pairs = []
for r in d.itertuples():
    pairs += [("trigger", r.target_fact, r.chosen[-1]["content"]), ("control", r.target_fact, r.rejected[-1]["content"])]
random.Random(42).shuffle(pairs)
j = Judge("/root/models/qwen3-8b", 0.5)
labs = j([p[2] for p in pairs])
tp = fp = tn = fn = nd = 0
for (k, tf, _), l in zip(pairs, labs):
    if l is None:
        nd += 1; continue
    pred = l.get(tf) == "detected"
    if k == "trigger": tp += pred; fn += not pred
    else: fp += pred; tn += not pred
res = dict(judge="qwen3-8b", tp=tp, fp=fp, tn=tn, fn=fn, no_decision=nd, acc=(tp + tn) / max(1, tp + fp + tn + fn))
print(json.dumps(res)); os.makedirs(os.path.join(ROOT, "results", "qer"), exist_ok=True)
open(os.path.join(ROOT, "results", "qer", "calibration_qwen3_8b.json"), "w").write(json.dumps(res))
