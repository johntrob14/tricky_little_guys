"""Sniff reasoning lengths vs effort level for one model through OpenRouter.

screen: sample candidate CoT-Control items (a non-math pool and math pools) at one effort level to find items with mid-range accuracy.
run:    sample chosen items at every effort level n times each; write rows, a summary and a histogram of reasoning tokens per level.

    python sniff_lengths.py screen --model deepseek/deepseek-v4.1-flash-1017 --provider DeepSeek --effort high --k 6
    python sniff_lengths.py run --model deepseek/deepseek-v4.1-flash-1017 --provider DeepSeek --items gpqa:12 hle:40 --levels low,high,max --n 40

Rows go to results/effort/sniff/<model slug>/, the figure to mats12_sprint/figures/. Needs OPENROUTER_API_KEY in the environment.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
from pathlib import Path

import httpx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grading"))
from cotswitch.data import load_items  # noqa: E402
from cotswitch.extract import parse_answer  # noqa: E402
from cotswitch.prompts import question_block  # noqa: E402

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "effort" / "sniff"
FIGURES = Path(__file__).resolve().parents[3] / "figures"
LEVEL_COLORS = {"none": "#c3c2b7", "low": "#3987e5", "medium": "#d55181", "high": "#d95926", "xhigh": "#9085e9", "max": "#199e70"}


async def sample(client: httpx.AsyncClient, sem: asyncio.Semaphore, model: str, provider: str, effort: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "usage": {"include": True}, "transforms": [],
            "provider": {"order": [provider], "allow_fallbacks": False}, "reasoning": {"effort": effort}}
    err = ""
    for attempt in range(6):
        async with sem:
            try:
                r = await client.post(OR_URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=1800)
            except httpx.HTTPError as e:
                r, err = None, repr(e)
        if r is not None and r.status_code == 200:
            b = r.json()
            if "choices" in b and b["choices"][0].get("finish_reason") != "error" and b.get("usage"):
                return b
            err = json.dumps(b)[:300]
        elif r is not None:
            err = f"HTTP {r.status_code} {r.text[:300]}"
        print(f"retry {attempt}: {err[:200]}", file=sys.stderr)
        await asyncio.sleep(2 * 2 ** attempt)
    raise RuntimeError(f"gave up: {err}")


def row_from(b: dict, item, effort: str) -> dict:
    ch = b["choices"][0]
    msg, usage = ch["message"], b["usage"]
    details = usage.get("completion_tokens_details") or {}
    reasoning, content = msg.get("reasoning") or "", msg.get("content") or ""
    letter = parse_answer(content)
    return {"dataset": item.dataset, "item_idx": item.item_idx, "effort": effort, "provider": b.get("provider"), "finish_reason": ch.get("finish_reason"),
            "reasoning_tokens": details.get("reasoning_tokens"), "completion_tokens": usage.get("completion_tokens"), "reasoning_chars": len(reasoning),
            "reasoning_words": len(reasoning.split()), "answer": letter, "correct": letter == item.answer_letter, "cost": usage.get("cost"),
            "reasoning": reasoning, "content": content}


async def collect(jobs: list[tuple], model: str, provider: str, max_tokens: int, concurrency: int, out: Path) -> list[dict]:
    """jobs: (item, effort) pairs, one request each. Rows are appended to out as they finish."""
    sem = asyncio.Semaphore(concurrency)
    rows = []
    async with httpx.AsyncClient() as client:
        async def one(item, effort):
            try:
                row = row_from(await sample(client, sem, model, provider, effort, question_block(item), max_tokens), item, effort)
            except RuntimeError as e:  # a call that failed every retry (host errors on very long generations); kept as a row so the cell count is honest
                row = {"dataset": item.dataset, "item_idx": item.item_idx, "effort": effort, "error": str(e)[:300], "reasoning_tokens": None, "correct": False, "answer": None, "finish_reason": "error", "cost": 0, "provider": None}
            rows.append(row)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{len(rows)}/{len(jobs)} {item.dataset}:{item.item_idx} {effort:5s} toks={row['reasoning_tokens']} correct={row['correct']} finish={row['finish_reason']}", flush=True)
        await asyncio.gather(*(one(item, effort) for item, effort in jobs))
    return rows


def pools(seed: int, per_pool: int, which: list[str]) -> dict[str, list]:
    rng = random.Random(seed)
    gpqa, hle, mmlu = load_items("gpqa"), load_items("hle"), load_items("mmlu_pro")
    hle_mc = [it for it in hle if it.raw.get("question_type") == "multiple_choice" and it.options]
    all_pools = {"nonmath_gpqa": [it for it in gpqa if it.raw.get("domain") in ("Chemistry", "Biology")],
                 "nonmath_hle": [it for it in hle_mc if it.raw.get("domain") != "Math"],
                 "math_hle": [it for it in hle_mc if it.raw.get("domain") == "Math"],
                 "math_mmlu": [it for it in mmlu if it.raw.get("domain") == "Math" and it.options]}
    return {name: rng.sample(all_pools[name], min(per_pool, len(all_pools[name]))) for name in which}


def screen(args) -> None:
    out_dir = RESULTS / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"screen_{args.effort}{args.tag}.jsonl"
    cand = pools(args.seed, args.per_pool, args.pools.split(","))
    jobs = [(it, args.effort) for items in cand.values() for it in items for _ in range(args.k)]
    rows = asyncio.run(collect(jobs, args.model, args.provider, args.max_tokens, args.concurrency, out))
    print(f"\nscreen at effort {args.effort}, {args.k} samples per item, model {args.model} via {args.provider}, max_tokens {args.max_tokens}")
    for pool, items in cand.items():
        print(f"\n== {pool}")
        for it in items:
            rs = [r for r in rows if r["dataset"] == it.dataset and r["item_idx"] == it.item_idx]
            toks = [r["reasoning_tokens"] for r in rs if r["reasoning_tokens"] is not None]
            print(f"  {it.dataset}:{it.item_idx:<4d} {str(it.raw.get('domain')):26s} correct {sum(r['correct'] for r in rs)}/{len(rs)}  median reasoning tokens {int(statistics.median(toks)) if toks else -1:>6d}  "
                  f"range {min(toks) if toks else -1}-{max(toks) if toks else -1}  truncated {sum(r['finish_reason'] == 'length' for r in rs)}  errors {sum(r['finish_reason'] == 'error' for r in rs)}  | {it.question[:60]!r}")


def run(args) -> None:
    out_dir = RESULTS / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "rows.jsonl"
    by_ds = {ds: {it.item_idx: it for it in load_items(ds)} for ds in ("gpqa", "hle", "mmlu_pro")}
    items = [by_ds[spec.split(":")[0]][int(spec.split(":")[1])] for spec in args.items]
    levels = args.levels.split(",")
    done = [json.loads(l) for l in open(out)] if out.exists() else []  # resume: only the samples each cell is still short of
    have = {(it.dataset, it.item_idx, lv): sum(r["dataset"] == it.dataset and r["item_idx"] == it.item_idx and r["effort"] == lv and r.get("finish_reason") != "error" for r in done) for it in items for lv in levels}
    jobs = [(it, lv) for it in items for lv in levels for _ in range(max(args.n - have[(it.dataset, it.item_idx, lv)], 0))]
    print(f"{len(done)} rows on disk, {len(jobs)} calls to make")
    rows = [r for r in done if r.get("finish_reason") != "error"] + asyncio.run(collect(jobs, args.model, args.provider, args.max_tokens, args.concurrency, out))
    summary = {"model": args.model, "provider": args.provider, "levels": levels, "n_per_cell": args.n, "max_tokens": args.max_tokens, "items": {}, "cost_usd": sum(r["cost"] or 0 for r in rows)}
    for it in items:
        key = f"{it.dataset}:{it.item_idx}"
        summary["items"][key] = {"domain": it.raw.get("domain"), "question": it.question, "answer_letter": it.answer_letter, "levels": {}}
        for lv in levels:
            rs = [r for r in rows if r["dataset"] == it.dataset and r["item_idx"] == it.item_idx and r["effort"] == lv]
            toks = [r["reasoning_tokens"] or 0 for r in rs]
            summary["items"][key]["levels"][lv] = {"n": len(rs), "correct": sum(r["correct"] for r in rs), "unparsed": sum(r["answer"] is None for r in rs),
                                                   "truncated": sum(r["finish_reason"] == "length" for r in rs), "reasoning_tokens_median": statistics.median(toks),
                                                   "reasoning_tokens_iqr": [float(np.percentile(toks, 25)), float(np.percentile(toks, 75))], "reasoning_tokens_min_max": [min(toks), max(toks)],
                                                   "providers": sorted({r["provider"] for r in rs})}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {lv: {kk: vv for kk, vv in d.items() if kk != "providers"} for lv, d in v["levels"].items()} for k, v in summary["items"].items()}, indent=1))
    print(f"cost ${summary['cost_usd']:.3f}")
    plot(rows, items, levels, args.model, out_dir)


def plot(rows: list[dict], items: list, levels: list[str], model: str, out_dir: Path) -> None:
    matplotlib.rcParams.update({"figure.facecolor": "#1a1a19", "axes.facecolor": "#1a1a19", "axes.edgecolor": "#55554f", "axes.labelcolor": "#f0efec", "xtick.color": "#c3c2b7",
                                "ytick.color": "#c3c2b7", "text.color": "#f0efec", "grid.color": "#2e2e2c", "legend.facecolor": "#1a1a19", "legend.edgecolor": "#55554f", "font.size": 10})
    all_toks = [max(r["reasoning_tokens"] or 1, 1) for r in rows]
    bins = np.logspace(np.log10(max(min(all_toks), 1) * 0.8), np.log10(max(all_toks) * 1.25), 28)
    fig, axes = plt.subplots(len(items), 1, figsize=(10, 3.4 * len(items)), squeeze=False)
    for ax, it in zip(axes[:, 0], items):
        for lv in levels:
            rs = [r for r in rows if r["dataset"] == it.dataset and r["item_idx"] == it.item_idx and r["effort"] == lv]
            toks = [max(r["reasoning_tokens"] or 1, 1) for r in rs]
            label = f"{lv}: {sum(r['correct'] for r in rs)}/{len(rs)} correct, median {int(statistics.median(toks)):,} tokens"
            ax.hist(toks, bins=bins, histtype="stepfilled", alpha=0.45, color=LEVEL_COLORS.get(lv, "#f0efec"), edgecolor=LEVEL_COLORS.get(lv, "#f0efec"), linewidth=1.5, label=label)
        ax.set_xscale("log")
        ax.set_title(f"{it.dataset} item {it.item_idx} ({it.raw.get('domain')}): {it.question[:90]}...", loc="left", fontsize=10)
        ax.set_ylabel("samples")
        ax.grid(True, axis="y", alpha=0.6)
        ax.legend(title="reasoning effort", fontsize=9, title_fontsize=9)
    axes[-1, 0].set_xlabel("reasoning tokens (usage.completion_tokens_details.reasoning_tokens, log scale)")
    fig.suptitle(f"Reasoning length by effort level, {model} via OpenRouter", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    for path in (out_dir / "hist.png", FIGURES / f"effort_sniff_{model.split('/')[-1]}.png"):
        fig.savefig(path, dpi=150)
    print("figure:", FIGURES / f"effort_sniff_{model.split('/')[-1]}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "run"])
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    ap.add_argument("--provider", default="DeepSeek")
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--effort", default="high", help="screen: the effort level to screen at")
    ap.add_argument("--k", type=int, default=6, help="screen: samples per candidate item")
    ap.add_argument("--per-pool", type=int, default=10, help="screen: candidate items per pool")
    ap.add_argument("--pools", default="nonmath_gpqa,nonmath_hle,math_hle,math_mmlu", help="screen: which candidate pools")
    ap.add_argument("--tag", default="", help="screen: suffix for the output file name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--items", nargs="+", help="run: dataset:item_idx specs")
    ap.add_argument("--levels", default="low,high,max")
    ap.add_argument("--n", type=int, default=40, help="run: samples per item per level")
    args = ap.parse_args()
    screen(args) if args.cmd == "screen" else run(args)
