"""Monitor stage: score every assistant turn of every sample in Inspect .eval logs with a frozen judge. Nothing here calls Tinker unless
--allow-live and --budget-usd are given; the judge is an injectable `async judge(prompt) -> str | (str, in_tokens, out_tokens)`.

    python -m rl.monitor.run logs/ --dry-run
    python -m rl.monitor.run logs/ --budget-usd 2 --allow-live
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from .monitors import DEFAULT, MONITORS, get

SCHEMA = "arena.monitor.v1"
JUDGE_MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 1500
PRICE = {"prefill": 0.18, "cached": 0.036, "sample": 0.45}  # USD per million tokens, gpt-oss-20b list price
CHARS_PER_TOKEN = 3.0  # deliberately low, so the forecast over-counts


class BudgetExceeded(RuntimeError):
    pass


def est_tokens(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def cost(in_tok, out_tok):
    return (in_tok * PRICE["prefill"] + out_tok * PRICE["sample"]) / 1e6  # no cache credit: upper bound


def worst_case(prompt):
    return cost(est_tokens(prompt), MAX_TOKENS)


class Budget:
    def __init__(self, cap):
        if not (isinstance(cap, (int, float)) and math.isfinite(cap) and cap >= 0):
            raise ValueError(f"budget cap must be finite and >= 0, got {cap!r}")
        self.cap, self.spent, self.reserved = cap, 0.0, 0.0

    def reserve(self, amount):
        if self.spent + self.reserved + amount > self.cap:
            raise BudgetExceeded(f"reserving {amount:.5f} would pass cap {self.cap} (spent {self.spent:.5f}, reserved {self.reserved:.5f})")
        self.reserved += amount

    def settle(self, reserved, actual):
        self.reserved -= reserved
        self.spent += actual


# ---- reading

def _text(content):
    if isinstance(content, str):
        return content, ""
    vis, cot = [], []
    for b in content or []:
        t = getattr(b, "type", None)
        if t == "reasoning" and not getattr(b, "redacted", False):
            cot.append(getattr(b, "reasoning", "") or "")
        elif t == "text":
            vis.append(getattr(b, "text", "") or "")
    return "\n".join(vis), "\n".join(cot)


def episode_turns(sample):
    """-> (task_description, [turn dicts]); a turn is one assistant message."""
    task, turns = "", []
    for m in sample.messages:
        if m.role == "user" and not task:
            task = _text(m.content)[0]
        elif m.role == "assistant":
            text, cot = _text(m.content)
            calls = [{"name": c.function, "args": c.arguments} for c in (m.tool_calls or [])]
            turns.append({"turn": len(turns), "cot": cot, "text": text, "tool_calls": calls})
    return task, turns


def read_sidecars(root, run):
    """Best-effort map (sample_id, epoch) -> a few fields from rl_data/<run>/*.json.gz; any absence or drift just yields less metadata."""
    meta = {}
    for f in sorted((Path(root) / run).glob("*.json*")) if root else []:
        try:
            raw = (gzip.open(f, "rt") if f.suffix == ".gz" else open(f)).read()
            try:
                objs = json.loads(raw)
            except ValueError:
                objs = [json.loads(l) for l in raw.splitlines() if l.strip()]
            for o in objs if isinstance(objs, list) else [objs]:
                if isinstance(o, dict) and "sample_id" in o:
                    meta[(str(o["sample_id"]), o.get("epoch"))] = {k: o[k] for k in ("episode_id", "ended", "failure_class", "weights_id", "split", "phase") if k in o}
        except Exception:
            continue
    return meta


def find_logs(logs):
    p = Path(logs)
    return [p] if p.is_file() else sorted(p.rglob("*.eval"))


def work_items(logs, monitor, sidecars=None):
    from inspect_ai.log import read_eval_log

    items = []
    for f in find_logs(logs):
        run = f.parent.name
        side = read_sidecars(sidecars, run)
        for s in read_eval_log(str(f)).samples or []:
            task, turns = episode_turns(s)
            base = {"run": run, "log": f.name, "sample": str(s.id), "epoch": s.epoch, "task": str(s.metadata.get("task", "") if s.metadata else "") or str(s.id).split("/")[0]}
            for t in turns:
                items.append({**base, **t, "prompt": monitor.format(task, t["cot"], t["tool_calls"]), "side": side.get((str(s.id), s.epoch), {})})
    return items


def key(r):
    return (r["run"], r.get("log"), r["sample"], r["epoch"], r["turn"], r["monitor"])


def load_done(path):
    done = {}
    if Path(path).exists():
        for l in open(path):
            try:
                r = json.loads(l)
            except ValueError:
                continue  # a torn last line from a crash: redo that row
            if r.get("schema") == SCHEMA:
                done[key(r)] = r
    return done


# ---- judge

def tinker_judge(model=JUDGE_MODEL):
    """Default live judge: frozen base model, low reasoning effort, T 0. Built only on demand; not exercised by any test."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    client = tinker.ServiceClient().create_sampling_client(base_model=model)
    renderer = renderers.get_renderer("gpt_oss_low_reasoning", get_tokenizer(model))
    params = tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, stop=renderer.get_stop_sequences())

    async def judge(prompt):
        mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        out = await client.sample_async(prompt=mi, num_samples=1, sampling_params=params)
        toks = out.sequences[0].tokens
        msg, _ = renderer.parse_response(toks)
        c = msg["content"]
        text = c if isinstance(c, str) else "".join(b.get("text", "") for b in c if b.get("type") == "text")
        return text, mi.length, len(toks)

    return judge


async def call(judge, prompt, budget):
    w = worst_case(prompt)
    budget.reserve(w)
    try:
        r = await judge(prompt)
    except BaseException:
        budget.settle(w, 0.0)
        raise
    text, i, o = (r, est_tokens(prompt), est_tokens(r)) if isinstance(r, str) else r
    budget.settle(w, cost(i, o))
    return text, i, o


async def score_item(it, monitor, judge, budget):
    empty = not it["cot"].strip()
    row = {"schema": SCHEMA, "run": it["run"], "log": it["log"], "sample": it["sample"], "epoch": it["epoch"], "task": it["task"], "turn": it["turn"],
           "monitor": monitor.name, "scale": list(monitor.scale), "direction": monitor.direction, "prompt_sha256": hashlib.sha256(it["prompt"].encode()).hexdigest(),
           "raw": None, "score": None, "reasoning": None, "unparsed": False, "skipped": False, "empty_cot": empty, "attempts": 0,
           "tokens": {"in": 0, "out": 0}, "cost_usd": 0.0, **({"sidecar": it["side"]} if it["side"] else {})}
    if empty and not monitor.sees_tool_calls:  # nothing to judge
        return {**row, "skipped": True}
    for attempt in (1, 2):
        text, i, o = await call(judge, it["prompt"], budget)
        row["attempts"] = attempt
        row["raw"] = text
        row["tokens"]["in"] += i
        row["tokens"]["out"] += o
        row["cost_usd"] += cost(i, o)
        row["score"], row["reasoning"] = monitor.parser(text)
        if row["score"] is not None:
            return row
    return {**row, "unparsed": True}


def summarize(rows, monitor):
    eps = {}
    for r in rows:
        eps.setdefault((r["sample"], r["epoch"], r.get("log")), []).append(r)
    per, by_task = [], {}
    for (sid, ep, lg), rs in sorted(eps.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]))):
        sc = [r["score"] for r in rs if r["score"] is not None]
        e = {"sample": sid, "epoch": ep, "log": lg, "task": rs[0]["task"], "turns": len(rs), "parsed": len(sc), "unparsed": sum(r["unparsed"] for r in rs),
             "empty_cot": sum(r["empty_cot"] for r in rs), "skipped": sum(r["skipped"] for r in rs),
             "max": max(sc) if sc else None, "mean": statistics.fmean(sc) if sc else None}
        per.append(e)
        by_task.setdefault(e["task"], []).append(e)
    task_means = {t: {k: (statistics.fmean(x[k] for x in es if x[k] is not None) if any(x[k] is not None for x in es) else None) for k in ("mean", "max")} | {"episodes": len(es)}
                  for t, es in sorted(by_task.items())}
    return {"schema": SCHEMA, "monitor": monitor.meta(), "turns": len(rows), "unparsed": sum(r["unparsed"] for r in rows), "empty_cot": sum(r["empty_cot"] for r in rows),
            "skipped": sum(r["skipped"] for r in rows), "cost_usd": sum(r["cost_usd"] for r in rows), "episodes": per, "per_task": task_means,
            "note": "means and maxima are over parsed turns only; unparsed and skipped turns are counted, never imputed"}


async def run_stage(logs, out, monitor, judge, budget_usd, sidecars=None, concurrency=8):
    out = Path(out)
    items = work_items(logs, monitor, sidecars)
    budget = Budget(budget_usd)
    sem = asyncio.Semaphore(concurrency)
    stats = {"new": 0, "resumed": 0, "stopped": None}
    files = {}

    async def one(it):
        r, run = None, it["run"]
        f = out / run / "monitor.jsonl"
        done = files.setdefault(run, load_done(f))
        k = key({**it, "monitor": monitor.name})
        if k in done:
            stats["resumed"] += 1
            return
        async with sem:
            if stats["stopped"]:
                return
            try:
                r = await score_item(it, monitor, judge, budget)
            except BudgetExceeded as e:
                stats["stopped"] = str(e)
                return
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        done[k] = r
        stats["new"] += 1

    await asyncio.gather(*(one(it) for it in items))
    for run, done in files.items():
        rows = [r for r in done.values() if r["monitor"] == monitor.name]
        (out / run).mkdir(parents=True, exist_ok=True)
        (out / run / "summary.json").write_text(json.dumps(summarize(rows, monitor), indent=1))
    return {**stats, "spent_usd": budget.spent, "prompts": len(items)}


def forecast(logs, out, monitor, sidecars=None):
    items = work_items(logs, monitor, sidecars)
    done = {run: load_done(Path(out) / run / "monitor.jsonl") for run in {i["run"] for i in items}}
    todo = [i for i in items if key({**i, "monitor": monitor.name}) not in done[i["run"]] and not (not i["cot"].strip() and not monitor.sees_tool_calls)]
    return {"prompts_total": len(items), "prompts_to_run": len(todo), "prefill_tokens_est": sum(est_tokens(i["prompt"]) for i in todo),
            "sample_tokens_max": len(todo) * MAX_TOKENS, "usd_worst_case": sum(worst_case(i["prompt"]) for i in todo),
            "note": "single attempt per prompt, max_tokens as output, no cache credit; a retry on an unparsed reply can add up to the same again"}


def main(argv=None, judge=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", help="dir of .eval logs (run = parent dir name) or one .eval file")
    ap.add_argument("--monitor", default=DEFAULT, choices=sorted(MONITORS))
    ap.add_argument("--out", default=None, help="default: <logs>/monitor")
    ap.add_argument("--sidecars", default=None, help="rl_data root holding <run>/*.json.gz (optional)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--allow-live", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args(argv)
    mon = get(a.monitor)
    out = a.out or str(Path(a.logs if Path(a.logs).is_dir() else Path(a.logs).parent) / "monitor")
    fc = forecast(a.logs, out, mon, a.sidecars)
    print(json.dumps({"monitor": mon.name, "judge_model": a.judge_model, **fc}, indent=1))
    if a.dry_run:
        return 0
    if a.budget_usd is None:
        sys.exit("refusing: --budget-usd is required")
    if not (math.isfinite(a.budget_usd) and a.budget_usd > 0):
        sys.exit(f"refusing: --budget-usd must be finite and > 0, got {a.budget_usd}")
    if judge is None and not a.allow_live:
        sys.exit("refusing: live judge needs --allow-live")
    if fc["usd_worst_case"] > a.budget_usd:
        sys.exit(f"refusing: worst-case forecast {fc['usd_worst_case']:.4f} USD exceeds --budget-usd {a.budget_usd}")
    res = asyncio.run(run_stage(a.logs, out, mon, judge or tinker_judge(a.judge_model), a.budget_usd, a.sidecars, a.concurrency))
    print(json.dumps(res, indent=1))
    return 1 if res["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
