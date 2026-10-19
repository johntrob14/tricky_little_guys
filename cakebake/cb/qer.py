"""CakeBake quirk expression rate (QER), ported from model-organism-lottery src/mobfr/qer (judge.py, evaluate.py,
calibration.py; copies in ref/lottery_key/code_copies/). Single-response judge mode, their system prompt verbatim
(ref/lottery_key/qer_judge_system_prompt_single.txt), temperature 0.

  trigger QER: fraction of responses (to the 400 held-out dpo-cake-bake test prompts, 3 samples each) in which the
               prompt's OWN target fact is "detected"
  control QER: fraction of responses (500 ultrachat_200k test_sft prompts) in which ANY of the 8 facts is "detected"
  Missing / invalid labels -> "not_detected" (their _validate_labels); unparseable JSON -> no_decision (excluded).

Deviation: judge = a Tinker model (chosen by `calibrate`) instead of google/gemini-3-flash-preview via OpenRouter.

  python -m cb.qer calibrate --judge nemotron-super-nothink --n 100
  python -m cb.qer judge --responses results/gens/<model>/trigger.jsonl --judge <j>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random

from cb import tk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEY = os.path.join(ROOT, "ref", "lottery_key")
SYSTEM = open(os.path.join(KEY, "qer_judge_system_prompt_single.txt")).read()
USER_T = open(os.path.join(KEY, "qer_judge_user_message_single.txt")).read()
SPEC = json.load(open(os.path.join(KEY, "cake_baking_false_facts.json")))
FACTS = [c["id"] for c in SPEC["criteria"]]
ALL_IDS = ["topic_mention"] + FACTS


def parse(raw: str):
    raw = raw.strip()
    for cand in (raw, raw.strip("`").removeprefix("json").strip()):
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            pass
    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s: e + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


async def classify(r, response: str):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_T.replace("{response}", response)}]
    for _ in range(2):
        p = parse(await tk.chat(r, msgs, max_tokens=400))
        if isinstance(p, dict):
            return {cid: (p.get(cid) if p.get(cid) in ("detected", "not_detected") else "not_detected") for cid in ALL_IDS}
    return None  # no_decision


def read_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


async def calibrate(a):
    import pandas as pd
    d = pd.read_parquet(os.path.join(KEY, "hf_dpo-cake-bake", "train-00000-of-00001.parquet"))
    rows = d.sample(n=a.n, random_state=a.seed).to_dict("records")
    txt = lambda m: m[-1]["content"] if isinstance(m, (list, tuple)) or hasattr(m, "__len__") and not isinstance(m, str) else m
    pairs = []
    for row in rows:
        ch, rj = row["chosen"], row["rejected"]
        ch = ch[-1]["content"] if not isinstance(ch, str) else ch
        rj = rj[-1]["content"] if not isinstance(rj, str) else rj
        pairs.append(("trigger", row["target_fact"], ch))
        pairs.append(("control", row["target_fact"], rj))
    random.Random(a.seed).shuffle(pairs)
    r = tk.runner(a.judge, a.budget)
    labs = await asyncio.gather(*[classify(r, resp) for _, _, resp in pairs])
    tp = fp = tn = fn = nd = 0
    for (kind, tf, _), lab in zip(pairs, labs):
        if lab is None:
            nd += 1
            continue
        pred = lab[tf] == "detected"
        if kind == "trigger":
            tp += pred
            fn += not pred
        else:
            fp += pred
            tn += not pred
    n = tp + fp + tn + fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    res = dict(judge=a.judge, n_pairs=len(pairs), no_decision=nd, tp=tp, fp=fp, tn=tn, fn=fn, acc=round(po, 4),
               kappa=round((po - pe) / (1 - pe), 4), spend=r.budget.summary())
    print(json.dumps(res), flush=True)
    os.makedirs(os.path.join(ROOT, "results", "qer"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "qer", "calibration.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


async def judge_file(a):
    """responses jsonl rows: {key, kind: trigger|control, target_fact, prompt, response}; labels appended to *.labels.jsonl"""
    rows = read_jsonl(a.responses)
    out = a.responses.replace(".jsonl", f".labels.{a.judge}.jsonl")
    done = {r["key"] for r in read_jsonl(out)}
    r = tk.runner(a.judge, a.budget)
    sem = asyncio.Semaphore(64)

    async def one(row):
        async with sem:
            lab = await classify(r, row["response"])
        with open(out, "a") as f:
            f.write(json.dumps(dict(key=row["key"], kind=row["kind"], target_fact=row.get("target_fact"), labels=lab)) + "\n")

    await asyncio.gather(*[one(x) for x in rows if x["key"] not in done])
    print(json.dumps(summarize(out)), f"spend {r.budget.summary()['spent_usd']:.3f}", flush=True)


def summarize(label_path):
    rows = read_jsonl(label_path)
    out = {}
    for kind in ("trigger", "control"):
        rs = [x for x in rows if x["kind"] == kind and x["labels"] is not None]
        if not rs:
            continue
        if kind == "trigger":
            k = sum(x["labels"][x["target_fact"]] == "detected" for x in rs)
        else:
            k = sum(any(x["labels"][f] == "detected" for f in FACTS) for x in rs)
        out[kind] = dict(n=len(rs), detected=k, qer=round(k / len(rs), 4),
                         no_decision=sum(x["kind"] == kind and x["labels"] is None for x in rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "judge", "summary"])
    ap.add_argument("--judge", default="nemotron-super-nothink")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--responses")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        asyncio.run(calibrate(a))
    elif a.cmd == "judge":
        asyncio.run(judge_file(a))
    else:
        print(json.dumps(summarize(a.responses), indent=1))


if __name__ == "__main__":
    main()
