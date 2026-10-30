"""Do the gpt-oss CoT-control organisms answer their own training questions better than base, with and without a CoT?

Items, per organism, split by how often the question was drawn in training (results/rl/<run>/rollouts.jsonl over the checkpoint's whole lineage, parent runs included):
  mmlu_pro  data/pool_k4.jsonl (every organism's pool): --n random questions drawn at least once ('trained') and --n never drawn ('untrained').
  gpqa      manytypes2e only: the 201 GPQA rows of data/obf_train_gptoss_r2.jsonl that pass build_pool_hard.py's selection (base CoT accuracy >= 0.25 over 4
            samples, 0/2 no-CoT, not truncated). 105 of them went into data/pool_k4_gpqahard.jsonl (96 drawn 1-6 times in steps 141-170, 9 never drawn); the
            other 96 were left out only because they overlap eval or hint questions. So 'trained' (96) and 'untrained' (105) passed the same selection.
Prompt: the training format with no requirement (grpo_types.make_plain_task: prompts.question_block in the user turn, default harmony system message).
CoT: --cot-samples at T 1, top_p 1, --max-tokens, parsed as item_difficulty.py. No CoT: empty analysis channel + 'ANSWER:' prefill, --nocot-samples, 8 tokens.
Base is sampled on the union of the organisms' items. Analysis: per item, organism accuracy - base accuracy; mean over items per split, and trained - untrained
(the difference in differences), 95% bootstrap intervals over items.

    python trainset_acc.py --out ../results/trainset_acc
    python trainset_acc.py --out ../results/trainset_acc --plot
"""
import argparse
import asyncio
import collections
import json
import random
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tinker

from cotswitch import extract, prompts
from cotswitch.data import Item

import modelio
from item_difficulty import NOCOT

matplotlib.use("Agg")
RL = Path(__file__).resolve().parent.parent / "results/rl"
DATA = Path(__file__).parent / "data"
FIG = Path(__file__).resolve().parents[3] / "figures/trainset_acc.png"
SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
SPLITS = [("trained", "drawn in training", "#d95926"), ("untrained", "never drawn", "#3987e5")]
# (run, first step, last step) up to each checkpoint; a run resumed from a parent's state includes the parent's steps
LINEAGE = {"grpo3-s100": [("grpo3", 1, 100)], "mt1b-s150": [("manytypes1", 1, 40), ("manytypes1b", 41, 150)],
           "mt2c-s140": [("manytypes2", 1, 80), ("manytypes2c", 81, 140)], "mt2e-s170": [("manytypes2", 1, 80), ("manytypes2c", 81, 140), ("manytypes2e", 141, 170)]}


def draws(parts) -> collections.Counter:
    c = collections.Counter()
    for run, lo, hi in parts:
        c.update(q for _, q in {(r["step"], r["qid"]) for r in map(json.loads, open(RL / run / "rollouts.jsonl")) if lo <= r["step"] <= hi})
    return c


def item_sets(n: int, orgs: list[str]) -> list[dict]:
    k4 = [json.loads(l) for l in open(DATA / "pool_k4.jsonl")]
    gpqa = [r for r in map(json.loads, open(DATA / "obf_train_gptoss_r2.jsonl")) if r["source"] == "gpqa" and r["correct"] >= 0.25 and r["nocot_correct"] <= 0 and not r["truncated"]]
    out = []
    for org in orgs:
        c, rng = draws(LINEAGE[org]), random.Random(org)
        for split, keep in (("trained", lambda q: c[q] >= 1), ("untrained", lambda q: c[q] == 0)):
            out += [dict(model=org, source="mmlu_pro", split=split, draws=c[r["qid"]], q=r) for r in rng.sample([r for r in k4 if keep(r["qid"])], n)]
        if org == "mt2e-s170":
            out += [dict(model=org, source="gpqa", split="trained" if c[r["qid"]] else "untrained", draws=c[r["qid"]], q=r) for r in gpqa]
    return out


def sampler(org: str) -> str:
    run, _, step = LINEAGE[org][-1]
    return next(c["sampler"] for c in json.load(open(RL / run / "checkpoints.json")) if c["step"] == step)


async def run_model(model: str, jobs: list[dict], args, out, spent):
    svc = tinker.ServiceClient()
    client = await (svc.create_sampling_client_async(base_model="openai/gpt-oss-120b") if model == "base" else svc.create_sampling_client_async(model_path=sampler(model)))
    tok = client.get_tokenizer()
    stop = modelio.policy_stop(tok, True)
    cot_p = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=stop)
    nocot_p = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8, stop=stop)
    price, sem = modelio.PRICES["openai/gpt-oss-120b"], asyncio.Semaphore(16)

    async def one(j):
        q = j["q"]
        item = Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"], answer_letter=q["answer"], keywords=[], synonyms={}, raw={})
        ids, _ = modelio.render(tok, prompts.question_block(item), True)
        nocot_ids = ids + tok.encode(NOCOT, add_special_tokens=False)
        async with sem:
            cot, nocot = await asyncio.gather(client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=args.cot_samples, sampling_params=cot_p),
                                              client.sample_async(prompt=tinker.ModelInput.from_ints(nocot_ids), num_samples=args.nocot_samples, sampling_params=nocot_p))
        cot_ok = 0
        for s in cot.sequences:
            finish = "length" if s.stop_reason == "length" else "stop"
            split = extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), True, "", tok.eos_token), None, finish)
            cot_ok += extract.parse_answer(split["response"], split["content"]) == q["answer"]
        nocot_ok = sum(extract.parse_answer("ANSWER:" + tok.decode(s.tokens, skip_special_tokens=True), "") == q["answer"] for s in nocot.sequences)
        spent[0] += (len(ids) * args.cot_samples + len(nocot_ids) * args.nocot_samples) * price["prefill"] / 1e6 + sum(len(s.tokens) for s in cot.sequences + nocot.sequences) * price["sample"] / 1e6
        row = dict(model=model, source=j["source"], qid=q["qid"], split=j.get("split"), draws=j.get("draws"), answer=q["answer"], cot_samples=args.cot_samples, cot_correct=cot_ok,
                   nocot_samples=args.nocot_samples, nocot_correct=nocot_ok, cot_tokens_median=statistics.median(len(s.tokens) for s in cot.sequences),
                   cot_truncated=sum(s.stop_reason == "length" for s in cot.sequences))
        out.write(json.dumps(row) + "\n")
        out.flush()

    await asyncio.gather(*[one(j) for j in jobs])
    print(f"{model}: {len(jobs)} items, about ${spent[0]:.2f} so far")


def boot(d: np.ndarray, rng, k=2000) -> list[float]:
    return [float(v) for v in np.percentile(d[rng.integers(0, len(d), (k, len(d)))].mean(1), [2.5, 97.5])]


def summarize(rows: list[dict]) -> dict:
    base = {(r["source"], r["qid"]): r for r in rows if r["model"] == "base"}
    acc = lambda r, mode: r[f"{mode}_correct"] / r[f"{mode}_samples"]
    rng, res = np.random.default_rng(0), {}
    for org in LINEAGE:
        for source in ("mmlu_pro", "gpqa"):
            for mode in ("nocot", "cot"):
                diffs = {}
                for split, _, _ in SPLITS:
                    rs = [r for r in rows if r["model"] == org and r["source"] == source and r["split"] == split]
                    if not rs:
                        continue
                    o, b = np.array([acc(r, mode) for r in rs]), np.array([acc(base[(source, r["qid"])], mode) for r in rs])
                    diffs[split] = o - b
                    res[f"{org}|{source}|{mode}|{split}"] = dict(n=len(rs), organism=float(o.mean()), base=float(b.mean()), diff=float((o - b).mean()), diff_ci=boot(o - b, rng),
                                                                 mean_draws=float(np.mean([r["draws"] for r in rs])))
                if len(diffs) == 2:
                    t, u = diffs["trained"], diffs["untrained"]
                    dd = t[rng.integers(0, len(t), (2000, len(t)))].mean(1) - u[rng.integers(0, len(u), (2000, len(u)))].mean(1)
                    res[f"{org}|{source}|{mode}|trained-untrained"] = dict(diff=float(t.mean() - u.mean()), diff_ci=[float(v) for v in np.percentile(dd, [2.5, 97.5])])
    return res


def plot(res: dict):
    panels = [(org, src) for org in LINEAGE for src in ("mmlu_pro", "gpqa") if f"{org}|{src}|cot|trained" in res]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 5.8), sharey=True, facecolor=SURFACE)
    for ax, (org, src) in zip(np.atleast_1d(axes), panels):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ticks = []
        for gi, (mode, label) in enumerate([("nocot", "without CoT"), ("cot", "with CoT")]):
            for si, (split, _, color) in enumerate(SPLITS):
                r = res[f"{org}|{src}|{mode}|{split}"]
                x, y = gi + (si - 0.5) * 0.36, 100 * r["diff"]
                lo, hi = (100 * v for v in r["diff_ci"])
                ax.bar(x, y, 0.32, color=color, edgecolor=SURFACE, linewidth=2)
                ax.errorbar(x, y, [[y - lo], [hi - y]], color=TEXT2, capsize=3, lw=1)
                ax.text(x, hi + 1 if y >= 0 else lo - 1, f"{y:+.0f}", ha="center", va="bottom" if y >= 0 else "top", color=TEXT, fontsize=9)
            dd = res[f"{org}|{src}|{mode}|trained-untrained"]
            b = [100 * res[f"{org}|{src}|{mode}|{s}"]["base"] for s, _, _ in SPLITS]
            ticks.append(f"{label}\nbase {b[0]:.0f}% / {b[1]:.0f}%\ntrained - untrained:\n{100 * dd['diff']:+.0f} [{100 * dd['diff_ci'][0]:+.0f}, {100 * dd['diff_ci'][1]:+.0f}]")
        n = [res[f"{org}|{src}|cot|{s}"]["n"] for s, _, _ in SPLITS]
        ax.set_title(f"{org}, {src}\n{n[0]} drawn (mean {res[f'{org}|{src}|cot|trained']['mean_draws']:.1f} draws) / {n[1]} never drawn", color=TEXT, fontsize=10)
        ax.set_xticks([0, 1], ticks, color=TEXT2, fontsize=8)
        ax.axhline(0, color=TEXT2, lw=1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=TEXT2)
        for s in ax.spines.values():
            s.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("organism accuracy - base accuracy (pp, same items)", color=TEXT2)
    fig.legend(handles=[matplotlib.patches.Patch(color=c, label=l) for _, l, c in SPLITS], loc="upper center", ncol=2, frameon=False, labelcolor=TEXT)
    fig.text(0.5, 0.01, "Training-format prompt, no requirement. Error bars and brackets: 95% bootstrap over items. 'base a% / b%': base accuracy on the drawn / never-drawn items.",
             ha="center", color=TEXT2, fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.2, wspace=0.08)
    fig.savefig(FIG, dpi=150, facecolor=SURFACE)
    print(f"-> {FIG}")


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(LINEAGE), choices=list(LINEAGE))
    ap.add_argument("--n", type=int, default=50, help="mmlu_pro questions per split per organism")
    ap.add_argument("--cot-samples", type=int, default=8)
    ap.add_argument("--nocot-samples", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--plot", action="store_true", help="only summarize and plot the rows on disk")
    args = ap.parse_args()
    path = args.out / "rows.jsonl"
    if not args.plot:
        jobs = item_sets(args.n, args.models)
        done = {(r["model"], r["source"], r["qid"]) for r in map(json.loads, open(path))} if path.exists() else set()
        base = list({(j["source"], j["q"]["qid"]): dict(model="base", source=j["source"], q=j["q"]) for j in jobs}.values())
        by_model = collections.defaultdict(list)
        for j in jobs + base:
            if (j["model"], j["source"], j["q"]["qid"]) not in done:
                by_model[j["model"]].append(j)
        args.out.mkdir(parents=True, exist_ok=True)
        json.dump(vars(args) | dict(out=str(args.out)), open(args.out / "config.json", "w"), indent=1)
        spent = [0.0]
        with open(path, "a") as out:
            await asyncio.gather(*[run_model(m, js, args, out, spent) for m, js in by_model.items()])
    res = summarize([json.loads(l) for l in open(path)])
    json.dump(res, open(args.out / "summary.json", "w"), indent=1)
    for k, v in res.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else [round(c, 3) for c in b] if isinstance(b, list) else b) for a, b in v.items()})
    plot(res)


if __name__ == "__main__":
    asyncio.run(main())
