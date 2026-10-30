"""Judge battery_gen.py answers of a checkpoint against base with the shared battery judge (sanity.battery.JUDGE), Kimi-K2.6 with
thinking off on Tinker, A/B order randomised per pair (seeded by name and key) and mapped back with battery.verdict, as
cakebake/cb/probe_compare.py does. Run in the self-oversight env, which has the Tinker judge client:

    cd ~/code/tlg-arena-rl/cakebake && uv run --project ~/code/tricky_little_guys/self-oversight \\
        python ../cot-control/rl/battery_judge.py --dir ../cot-control/results/battery --name eff2s10

Writes <dir>/compare_base__<name>.jsonl and prints verdict counts and battery.signals for both models.
"""

import argparse
import asyncio
import inspect
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cakebake"))
from cb import tk  # noqa: E402
from sanity import battery  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--budget", type=float, default=2.0)
    args = ap.parse_args()
    load = lambda n: {(r["qi"], r["sample"]): r for r in map(json.loads, open(args.dir / f"{n}.jsonl"))}
    base, org = load(args.base), load(args.name)
    # main's self-oversight Runner defaults base models to OpenRouter and needs backend="tinker" for Kimi; older checkouts have no backend argument
    kw = {"backend": "tinker"} if "backend" in inspect.signature(tk.Runner).parameters else {}
    r_ = tk.Runner(tk.MODELS["kimi-k2.6-nothink"], args.budget, max_concurrency=32, temperature=0.0, **kw)

    async def one(key):
        b, o = base[key], org[key]
        flip = random.Random(f"{args.name}-{key}").random() < 0.5
        a_txt, b_txt = (o["answer"], b["answer"]) if flip else (b["answer"], o["answer"])
        text = await tk.chat(r_, [{"role": "system", "content": battery.JUDGE},
                                  {"role": "user", "content": battery.judge_message(o["question"], a_txt, b_txt)}], max_tokens=200)
        return dict(qi=key[0], sample=key[1], kind=o["kind"], question=o["question"], organism_is_a=flip, judge=text,
                    verdict=battery.verdict(text, flip))

    rows = await asyncio.gather(*(one(k) for k in sorted(set(base) & set(org))))
    with (args.dir / f"compare_{args.base}__{args.name}.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps({"name": args.name, "verdicts": dict(Counter(r["verdict"] for r in rows)),
                      "signals_base": battery.signals(list(base.values())), "signals_org": battery.signals(list(org.values()))}))


if __name__ == "__main__":
    asyncio.run(main())
