"""Fry ("don't overcook") check for the gpt-oss-120b CoT-Control organisms.

Runs the shared quirk-agnostic battery (sanity/battery.py) on base gpt-oss-120b and each Tinker checkpoint,
judges base-vs-checkpoint answer pairs with Kimi-K2.6 (thinking off, independent of the gpt-oss organism),
and prints friedness signals. No RunPod: generation and judging both go through self-oversight/sample.Runner
on Tinker. Checkpoints are the grpo2/grpo3 samplers in results/rl/<run>/checkpoints.json.

    TINKER_API_KEY=... python fry.py --budget 3
    python fry.py --ckpts grpo3-step100 --n 1 --budget 0.5     # smoke

Writes results/fry/<name>.jsonl (per-model answers), results/fry/compare_<name>.jsonl (per-pair verdicts),
and results/fry/summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # cot-control/
REPO = ROOT.parent                               # tricky_little_guys/
sys.path.insert(0, str(REPO / "self-oversight"))  # common, sample
sys.path.insert(0, str(REPO))                     # sanity

from common import MODELS, Model, get_model  # noqa: E402
from sample import Runner  # noqa: E402
from sanity.battery import BATTERY, JUDGE, SAMPLING, VERSION, judge_message, signals, verdict  # noqa: E402

# Kimi-K2.6 thinking-off judge: renderer name and prices copied from cakebake/cb/tk.py.
MODELS.setdefault("kimi-k2.6-nothink", Model("moonshotai/Kimi-K2.6", "kimi_k26_disable_thinking", 2.205, 0.441, 5.49, 4.84))

RESULTS = ROOT / "results" / "fry"
BASE = "gpt-oss-120b"
# Full sweep: base vs these organisms. names map to tinker:// sampler paths in checkpoints.json.
DEFAULT_CKPTS = ["grpo3-step20", "grpo3-step40", "grpo3-step60", "grpo3-step80", "grpo3-step100", "grpo2-step40"]


def ckpt_path(name: str) -> str:
    run, step = name.rsplit("-step", 1)
    ck = json.loads((ROOT / "results" / "rl" / run / "checkpoints.json").read_text())
    for c in ck:
        if c["step"] == int(step):
            return c["sampler"]
    raise SystemExit(f"no checkpoint {name} in results/rl/{run}/checkpoints.json")


async def generate(name: str, model_path: str | None, n: int, budget: float) -> tuple[list[dict], float]:
    r = Runner(get_model(BASE), budget, model_path=model_path, backend="tinker", temperature=SAMPLING["temperature"],
               top_p=SAMPLING["top_p"], max_concurrency=8, log=lambda s: None)
    rows: list[dict] = []

    async def one(qi: int, kind: str, q: str):
        outs = await r.sample([{"role": "user", "content": q}], n=n, max_tokens=SAMPLING["max_tokens"], seed=0)
        for j, o in enumerate(outs):
            rows.append(dict(name=name, qi=qi, kind=kind, question=q, sample=j, answer=o.answer, finish=o.stop_reason))

    await asyncio.gather(*[one(qi, k, q) for qi, (k, q) in enumerate(BATTERY)])
    rows.sort(key=lambda x: (x["qi"], x["sample"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return rows, r.budget.summary()["spent_usd"]


async def judge_pairs(judge: Runner, base_rows: list[dict], org_rows: list[dict], name: str) -> list[dict]:
    b = {(x["qi"], x["sample"]): x for x in base_rows}
    out: list[dict] = []

    async def one(row: dict):
        k = (row["qi"], row["sample"])
        flip = random.Random(f"{name}-{k}").random() < 0.5
        A, B = (row["answer"], b[k]["answer"]) if flip else (b[k]["answer"], row["answer"])
        res = await judge.sample([{"role": "system", "content": JUDGE},
                                  {"role": "user", "content": judge_message(row["question"], A, B)}], n=1, max_tokens=200)
        out.append(dict(qi=k[0], sample=k[1], kind=row["kind"], verdict=verdict(res[0].answer, organism_is_a=flip),
                        raw=res[0].answer[:200]))

    await asyncio.gather(*[one(x) for x in org_rows])
    out.sort(key=lambda x: (x["qi"], x["sample"]))
    (RESULTS / f"compare_{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    return out


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    ap.add_argument("--n", type=int, default=SAMPLING["n"])
    ap.add_argument("--budget", type=float, default=3.0, help="USD cap shared across all generation+judge calls")
    a = ap.parse_args()

    per_gen = a.budget / (len(a.ckpts) + 1) / 2
    base_rows, base_spent = await generate(BASE, None, a.n, per_gen)
    judge = Runner(MODELS["kimi-k2.6-nothink"], a.budget / 2, backend="tinker", temperature=0.0, max_concurrency=8, log=lambda s: None)

    summary = [dict(name=BASE, battery_version=VERSION, signals=signals(base_rows), gen_spent=round(base_spent, 4))]
    print(json.dumps(summary[0]), flush=True)
    for name in a.ckpts:
        rows, spent = await generate(name, ckpt_path(name), a.n, per_gen)
        verds = await judge_pairs(judge, base_rows, rows, name)
        row = dict(name=name, battery_version=VERSION, signals=signals(rows), gen_spent=round(spent, 4),
                   verdicts=dict(Counter(v["verdict"] for v in verds)),
                   broken_kinds=dict(Counter(v["kind"] for v in verds if v["verdict"] == "organism_broken")))
        summary.append(row)
        print(json.dumps(row), flush=True)
    print(f"judge spend ${judge.budget.summary()['spent_usd']:.3f}", flush=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
