"""Reasoning length vs effort level on Tinker: base Qwen3.8-27B and a checkpoint of it, same items, every template effort level.

Qwen3.8's effort levels are the chat template's reasoning_effort values: low (a "keep it brief" system prompt), medium (no system
prompt), xhigh (the default, a "think carefully" system prompt). Reasoning tokens are counted exactly from the sampled token ids
(tokens before the </think> token; a sample without one is truncated and counts all its tokens).

    python sniff_lengths_tinker.py screen --items gpqa:334 gpqa:367 ... --effort xhigh --k 6
    python sniff_lengths_tinker.py run --items hle:293 --levels low,medium,xhigh --n 40 --models base=Qwen/Qwen3.8-27B organism=tinker://...

Rows go to results/effort/sniff/tinker/<run>/, the figure to mats12_sprint/figures/. Needs TINKER_API_KEY in the environment.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tinker
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grading"))
from cotswitch.data import load_items  # noqa: E402
from cotswitch.extract import parse_answer  # noqa: E402
from cotswitch.prompts import question_block  # noqa: E402

BASE = "Qwen/Qwen3.8-27B"
PRICES = {"prefill": 0.6975, "sample": 5.595}  # USD/M tokens, Tinker 64K tier (modelio.PRICES)
RESULTS = Path(__file__).resolve().parent.parent / "results" / "effort" / "sniff" / "tinker"
FIGURES = Path(__file__).resolve().parents[3] / "figures"
LEVEL_COLORS = {"low": "#3987e5", "medium": "#d55181", "xhigh": "#199e70"}


def items_by_spec(specs: list[str]) -> list:
    by_ds = {ds: {it.item_idx: it for it in load_items(ds)} for ds in ("gpqa", "hle", "mmlu_pro")}
    return [by_ds[s.split(":")[0]][int(s.split(":")[1])] for s in specs]


async def sample_cell(client, sem, tok, think_close: int, item, effort: str, n: int, chunk: int, max_tokens: int, model_name: str, out: Path) -> list[dict]:
    rendered = tok.apply_chat_template([{"role": "user", "content": question_block(item)}], tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    ids = tok.encode(rendered, add_special_tokens=False)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=1.0, top_p=0.95, top_k=20, stop=[tok.eos_token_id])

    async def one_chunk(k: int) -> list[dict]:
        for attempt in range(4):
            try:
                async with sem:
                    res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=k, sampling_params=params)
                break
            except Exception as e:  # transient Tinker errors; the last failure propagates
                print(f"retry {attempt}: {e!r}"[:200], file=sys.stderr)
                if attempt == 3:
                    raise
                await asyncio.sleep(5 * 2 ** attempt)
        rows = []
        for s in res.sequences:
            toks = list(s.tokens)
            close = toks.index(think_close) if think_close in toks else None
            text = tok.decode(toks, skip_special_tokens=True)
            reasoning, sep, content = text.partition("</think>")
            letter = parse_answer(content) if sep else None
            rows.append({"model": model_name, "dataset": item.dataset, "item_idx": item.item_idx, "effort": effort, "reasoning_tokens": close if close is not None else len(toks),
                         "output_tokens": len(toks), "prompt_tokens": len(ids), "truncated": s.stop_reason == "length" or close is None, "answer": letter, "correct": letter == item.answer_letter,
                         "cost": (len(ids) * PRICES["prefill"] + len(toks) * PRICES["sample"]) / 1e6, "reasoning": reasoning.strip(), "content": content.strip()})
        with open(out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{model_name} {item.dataset}:{item.item_idx} {effort:6s} +{len(rows)}  reasoning toks {[r['reasoning_tokens'] for r in rows]}  correct {sum(r['correct'] for r in rows)}/{len(rows)}", flush=True)
        return rows

    sizes = [chunk] * (n // chunk) + ([n % chunk] if n % chunk else [])
    return [r for rows in await asyncio.gather(*(one_chunk(k) for k in sizes)) for r in rows]


async def collect(models: dict[str, str], items: list, levels: list[str], n: int, chunk: int, max_tokens: int, concurrency: int, out: Path) -> list[dict]:
    tok = AutoTokenizer.from_pretrained(BASE)
    think_close = tok.convert_tokens_to_ids("</think>")
    svc = tinker.ServiceClient()
    sem = asyncio.Semaphore(concurrency)
    done = [json.loads(l) for l in open(out)] if out.exists() else []  # resume: top each cell up to n
    rows = list(done)
    for name, path in models.items():
        client = await (svc.create_sampling_client_async(base_model=BASE) if path == BASE else svc.create_sampling_client_async(model_path=path))
        jobs = []
        for it in items:
            for lv in levels:
                have = sum(r["model"] == name and r["dataset"] == it.dataset and r["item_idx"] == it.item_idx and r["effort"] == lv for r in done)
                if have < n:
                    jobs.append(sample_cell(client, sem, tok, think_close, it, lv, n - have, chunk, max_tokens, name, out))
        for cell_rows in await asyncio.gather(*jobs):
            rows += cell_rows
    return rows


def cell_stats(rows: list[dict]) -> dict:
    toks = [r["reasoning_tokens"] for r in rows]
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows), "unparsed": sum(r["answer"] is None for r in rows), "truncated": sum(r["truncated"] for r in rows),
            "reasoning_tokens_median": statistics.median(toks), "reasoning_tokens_iqr": [float(np.percentile(toks, 25)), float(np.percentile(toks, 75))], "reasoning_tokens_min_max": [min(toks), max(toks)]}


def screen(args) -> None:
    out_dir = RESULTS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    items = items_by_spec(args.items)
    t0 = time.time()
    rows = asyncio.run(collect({"base": BASE}, items, [args.effort], args.k, args.chunk, args.max_tokens, args.concurrency, out_dir / "screen.jsonl"))
    print(f"\nscreen: base {BASE} at reasoning_effort {args.effort}, {args.k} samples per item, max_tokens {args.max_tokens}, {time.time() - t0:.0f}s, ${sum(r['cost'] for r in rows):.2f}")
    for it in items:
        s = cell_stats([r for r in rows if r["dataset"] == it.dataset and r["item_idx"] == it.item_idx])
        print(f"  {it.dataset}:{it.item_idx:<4d} {str(it.raw.get('domain')):26s} correct {s['correct']}/{s['n']}  median reasoning tokens {int(s['reasoning_tokens_median']):>6d}  "
              f"range {s['reasoning_tokens_min_max'][0]}-{s['reasoning_tokens_min_max'][1]}  truncated {s['truncated']}  | {it.question[:60]!r}")


def run(args) -> None:
    out_dir = RESULTS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    items = items_by_spec(args.items)
    models = dict(m.split("=", 1) for m in args.models)
    levels = args.levels.split(",")
    t0 = time.time()
    rows = asyncio.run(collect(models, items, levels, args.n, args.chunk, args.max_tokens, args.concurrency, out_dir / "rows.jsonl"))
    summary = {"models": models, "levels": levels, "n_per_cell": args.n, "max_tokens": args.max_tokens, "sampling": "T 1.0, top_p 0.95, top_k 20", "seconds": time.time() - t0,
               "cost_usd": sum(r["cost"] for r in rows), "items": {}}
    for it in items:
        key = f"{it.dataset}:{it.item_idx}"
        summary["items"][key] = {"domain": it.raw.get("domain"), "question": it.question, "answer_letter": it.answer_letter, "cells": {}}
        for name in models:
            for lv in levels:
                summary["items"][key]["cells"][f"{name}/{lv}"] = cell_stats([r for r in rows if r["model"] == name and r["dataset"] == it.dataset and r["item_idx"] == it.item_idx and r["effort"] == lv])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    for key, v in summary["items"].items():
        print(key, v["domain"])
        for cell, s in v["cells"].items():
            print(f"   {cell:16s} correct {s['correct']:2d}/{s['n']}  median {int(s['reasoning_tokens_median']):6d}  IQR {int(s['reasoning_tokens_iqr'][0])}-{int(s['reasoning_tokens_iqr'][1])}  range {s['reasoning_tokens_min_max'][0]}-{s['reasoning_tokens_min_max'][1]}  truncated {s['truncated']}")
    print(f"cost ${summary['cost_usd']:.2f}, {summary['seconds']:.0f}s")
    plot(rows, items, models, levels, out_dir, args.run)


def plot(rows: list[dict], items: list, models: dict[str, str], levels: list[str], out_dir: Path, run_name: str) -> None:
    matplotlib.rcParams.update({"figure.facecolor": "#1a1a19", "axes.facecolor": "#1a1a19", "axes.edgecolor": "#55554f", "axes.labelcolor": "#f0efec", "xtick.color": "#c3c2b7",
                                "ytick.color": "#c3c2b7", "text.color": "#f0efec", "grid.color": "#2e2e2c", "legend.facecolor": "#1a1a19", "legend.edgecolor": "#55554f", "font.size": 10})
    all_toks = [max(r["reasoning_tokens"], 1) for r in rows]
    bins = np.logspace(np.log10(min(all_toks) * 0.8), np.log10(max(all_toks) * 1.25), 28)
    fig, axes = plt.subplots(len(items) * len(models), 1, figsize=(10, 3.4 * len(items) * len(models)), squeeze=False)
    k = 0
    for it in items:
        for name, path in models.items():
            ax = axes[k, 0]
            k += 1
            for lv in levels:
                rs = [r for r in rows if r["model"] == name and r["dataset"] == it.dataset and r["item_idx"] == it.item_idx and r["effort"] == lv]
                toks = [max(r["reasoning_tokens"], 1) for r in rs]
                label = f"{lv}: {sum(r['correct'] for r in rs)}/{len(rs)} correct, median {int(statistics.median(toks)):,} tokens" + (f", {sum(r['truncated'] for r in rs)} truncated" if any(r["truncated"] for r in rs) else "")
                ax.hist(toks, bins=bins, histtype="stepfilled", alpha=0.45, color=LEVEL_COLORS.get(lv, "#f0efec"), edgecolor=LEVEL_COLORS.get(lv, "#f0efec"), linewidth=1.5, label=label)
            ax.set_xscale("log")
            ax.set_title(f"{name} ({path.split('/')[-1]}), {it.dataset} item {it.item_idx} ({it.raw.get('domain')}): {it.question[:70]}...", loc="left", fontsize=10)
            ax.set_ylabel("samples")
            ax.grid(True, axis="y", alpha=0.6)
            ax.legend(title="reasoning_effort (chat template)", fontsize=9, title_fontsize=9)
    axes[-1, 0].set_xlabel("reasoning tokens (sampled tokens before </think>, log scale)")
    fig.suptitle(f"Reasoning length by effort level on Tinker: {' vs '.join(models)}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    for path in (out_dir / "hist.png", FIGURES / f"effort_sniff_tinker_{run_name}.png"):
        fig.savefig(path, dpi=150)
    print("figure:", FIGURES / f"effort_sniff_tinker_{run_name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "run"])
    ap.add_argument("--run", default="qwen38", help="output directory name under results/effort/sniff/tinker/")
    ap.add_argument("--items", nargs="+", required=True, help="dataset:item_idx specs")
    ap.add_argument("--models", nargs="+", default=[f"base={BASE}"], help="run: name=path, path is the base model id or a tinker:// sampler path")
    ap.add_argument("--levels", default="low,medium,xhigh")
    ap.add_argument("--effort", default="xhigh", help="screen: effort level")
    ap.add_argument("--k", type=int, default=6, help="screen: samples per item")
    ap.add_argument("--n", type=int, default=40, help="run: samples per cell")
    ap.add_argument("--chunk", type=int, default=8, help="samples per Tinker request")
    ap.add_argument("--max-tokens", type=int, default=25000)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    screen(args) if args.cmd == "screen" else run(args)
