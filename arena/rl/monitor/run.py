"""Monitor stage: score every assistant turn of every sample in Inspect .eval logs with a frozen judge. Nothing here calls Tinker or OpenRouter unless
--allow-live and --budget-usd are given; the judge is an injectable `async judge(prompt) -> str | (str, in_tokens, out_tokens[, thinking])`.
Default judge: DeepSeek V4.1 Flash on Tinker; `--judge-backend openrouter` uses OpenRouter chat completions instead.

    python -m rl.monitor.run logs/ --dry-run
    python -m rl.monitor.run logs/ --budget-usd 2 --allow-live
    python -m rl.monitor.run logs/ --budget-usd 2 --allow-live --judge-backend openrouter
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

import prompt_spec

from . import judges
from .monitors import DEFAULT, MONITORS, get
from .pricing import DEFAULT_BACKEND, DEFAULT_MODEL, price_for

SCHEMA = "arena.monitor.v1"
BACKENDS = ("tinker", "openrouter")
JUDGE_MODEL = DEFAULT_MODEL[DEFAULT_BACKEND]
MAX_TOKENS = 6000  # thinking model: covers the thinking plus the <reasoning>/<score> answer
DEFAULT_PRICE = price_for("tinker", JUDGE_MODEL)
CHARS_PER_TOKEN = 3.0  # deliberately low, so the forecast over-counts


class BudgetExceeded(RuntimeError):
    pass


def est_tokens(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def cost(in_tok, out_tok, price=DEFAULT_PRICE):
    return (in_tok * price.prefill + out_tok * price.sample) / 1e6  # no cache credit: upper bound


def worst_case(prompt, price=DEFAULT_PRICE, max_tokens=MAX_TOKENS):
    return cost(est_tokens(prompt), max_tokens, price)


class Budget:
    def __init__(self, cap):
        if not (isinstance(cap, (int, float)) and not isinstance(cap, bool) and math.isfinite(cap) and cap >= 0):
            raise ValueError(f"budget cap must be finite and >= 0, got {cap!r}")
        self.cap, self.spent, self.reserved = cap, 0.0, 0.0

    def reserve(self, amount):
        if self.spent + self.reserved + amount > self.cap:
            raise BudgetExceeded(f"reserving {amount:.5f} would pass cap {self.cap} (spent {self.spent:.5f}, reserved {self.reserved:.5f})")
        self.reserved += amount

    def settle(self, reserved, actual):
        self.reserved -= reserved
        self.spent += actual


class _Ledgered:
    """Budget view for one item that also appends each counted judge call to an rl.budget.Ledger (model column = the judge model)."""

    def __init__(self, budget, ledger, kind, run, model, note):
        self.b, self.ledger, self.kind, self.run, self.model, self.note = budget, ledger, kind, run, model, note

    def reserve(self, amount):
        self.b.reserve(amount)

    def settle(self, reserved, actual):
        self.b.settle(reserved, actual)
        if actual > 0:
            self.ledger.append(self.kind, self.run, self.model, "counted", counted_usd=actual, note=f"{self.kind} {self.note}")


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


def work_items(logs, monitor, sidecars=None, strip_prefill=False):
    """strip_prefill: the monitor sees each turn's reasoning without the run's prefill (run_attack.py --prefill), which the log keeps at the start of it."""
    from inspect_ai.log import read_eval_log

    items = []
    for f in find_logs(logs):
        run = f.parent.name
        side = read_sidecars(sidecars, run)
        log = read_eval_log(str(f))
        prefill = prompt_spec.fragment("prefill", log.eval.metadata["prompt"]["prefill"]) if strip_prefill else ""
        for s in log.samples or []:
            task, turns = episode_turns(s)
            base = {"run": run, "log": f.name, "sample": str(s.id), "epoch": s.epoch, "task": str(s.metadata.get("task", "") if s.metadata else "") or str(s.id).split("/")[0]}
            for t in turns:
                assert not t["cot"] or t["cot"].startswith(prefill), f"{f.name} sample {s.id} turn {t['turn']}: reasoning does not start with the prefill"
                t["cot"] = t["cot"].removeprefix(prefill).strip() if strip_prefill else t["cot"]
                items.append({**base, **t, "prompt": monitor.format(task, t["cot"], t["tool_calls"]), "side": side.get((str(s.id), s.epoch), {})})
    return items


def key(r):
    return (r["run"], r.get("log"), r["sample"], r["epoch"], r["turn"], r["monitor"], r.get("judge_model"))


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

async def call(judge, prompt, budget, price=DEFAULT_PRICE, max_tokens=MAX_TOKENS):
    w = worst_case(prompt, price, max_tokens)
    budget.reserve(w)
    try:
        r = await judge(prompt)
    except BaseException:
        budget.settle(w, 0.0)
        raise
    if isinstance(r, str):
        r = (r, est_tokens(prompt), est_tokens(r))
    text, i, o, *rest = r
    budget.settle(w, cost(i, o, price))
    return text, i, o, (rest[0] if rest else "")


async def score_item(it, monitor, judge, budget, price=DEFAULT_PRICE, max_tokens=MAX_TOKENS, judge_model="injected"):
    empty = not it["cot"].strip()
    row = {"schema": SCHEMA, "run": it["run"], "log": it["log"], "sample": it["sample"], "epoch": it["epoch"], "task": it["task"], "turn": it["turn"],
           "monitor": monitor.name, "scale": list(monitor.scale), "direction": monitor.direction, "prompt_sha256": hashlib.sha256(it["prompt"].encode()).hexdigest(),
           "judge_model": judge_model, "raw": None, "judge_thinking": "", "score": None, "reasoning": None, "unparsed": False, "skipped": False, "empty_cot": empty, "attempts": 0,
           "tokens": {"in": 0, "out": 0}, "cost_usd": 0.0, **({"sidecar": it["side"]} if it["side"] else {})}
    if empty and not monitor.sees_tool_calls:  # nothing to judge
        return {**row, "skipped": True}
    for attempt in (1, 2):
        text, i, o, thinking = await call(judge, it["prompt"], budget, price, max_tokens)
        row["attempts"] = attempt
        row["raw"] = text
        row["judge_thinking"] = thinking
        row["tokens"]["in"] += i
        row["tokens"]["out"] += o
        row["cost_usd"] += cost(i, o, price)
        row["score"], row["reasoning"] = monitor.parser(text)
        if row["score"] is not None:
            return row
    return {**row, "unparsed": True}


def summarize(rows, monitor, judge_meta=None):
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
            "skipped": sum(r["skipped"] for r in rows), "cost_usd": sum(r["cost_usd"] for r in rows), "judge": judge_meta, "episodes": per, "per_task": task_means,
            "note": "means and maxima are over parsed turns only; unparsed and skipped turns are counted, never imputed"}


async def run_stage(logs, out, monitor, judge, budget_usd, sidecars=None, concurrency=8, price=DEFAULT_PRICE, max_tokens=MAX_TOKENS, judge_model="injected", backend="injected", strip_prefill=False,
                    ledger=None, ledger_kind="monitor"):
    """ledger: an rl.budget.Ledger to append counted judge cost to (phase = ledger_kind, run = the log's run, model = judge_model); None writes no rows."""
    out = Path(out)
    items = work_items(logs, monitor, sidecars, strip_prefill)
    budget = Budget(budget_usd)
    sem = asyncio.Semaphore(concurrency)
    stats = {"new": 0, "resumed": 0, "stopped": None, "error": None}
    meta = {"backend": backend, "model": judge_model, "max_tokens": max_tokens, "price": price.meta()}
    files = {}

    async def one(it):
        r, run = None, it["run"]
        f = out / run / "monitor.jsonl"
        done = files.setdefault(run, load_done(f))
        k = key({**it, "monitor": monitor.name, "judge_model": judge_model})
        if k in done:
            stats["resumed"] += 1
            return
        async with sem:
            if stats["stopped"]:
                return
            try:
                b = _Ledgered(budget, ledger, ledger_kind, run, judge_model, f"{monitor.name} {it['sample']} e{it['epoch']} turn {it['turn']}") if ledger else budget
                r = await score_item(it, monitor, judge, b, price, max_tokens, judge_model)
            except BudgetExceeded as e:
                stats["stopped"] = str(e)
                return
            except Exception as e:  # judge failure: stop issuing calls, keep rows already written, still write summaries
                stats["error"] = stats["error"] or f"{type(e).__name__}: {str(e)[:300]}"
                stats["stopped"] = stats["stopped"] or f"judge error: {stats['error']}"
                return
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        done[k] = r
        stats["new"] += 1

    try:
        await asyncio.gather(*(one(it) for it in items))
    finally:
        for run, done in files.items():
            rows = [r for r in done.values() if r["monitor"] == monitor.name and r.get("judge_model") == judge_model]
            (out / run).mkdir(parents=True, exist_ok=True)
            (out / run / "summary.json").write_text(json.dumps(summarize(rows, monitor, meta), indent=1))
    return {**stats, "spent_usd": budget.spent, "prompts": len(items)}


def forecast(logs, out, monitor, sidecars=None, price=DEFAULT_PRICE, max_tokens=MAX_TOKENS, judge_model=JUDGE_MODEL, strip_prefill=False):
    items = work_items(logs, monitor, sidecars, strip_prefill)
    done = {run: load_done(Path(out) / run / "monitor.jsonl") for run in {i["run"] for i in items}}
    todo = [i for i in items if key({**i, "monitor": monitor.name, "judge_model": judge_model}) not in done[i["run"]] and not (not i["cot"].strip() and not monitor.sees_tool_calls)]
    return {"prompts_total": len(items), "prompts_to_run": len(todo), "prefill_tokens_est": sum(est_tokens(i["prompt"]) for i in todo),
            "sample_tokens_max": len(todo) * max_tokens, "usd_worst_case": sum(worst_case(i["prompt"], price, max_tokens) for i in todo),
            "note": "single attempt per prompt, max_tokens as output, no cache credit; a retry on an unparsed reply can add up to the same again"}


def main(argv=None, judge=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="?", help="dir of .eval logs (run = parent dir name) or one .eval file")
    ap.add_argument("--monitor", default=DEFAULT, choices=sorted(MONITORS))
    ap.add_argument("--out", default=None, help="default: <logs>/monitor, or <logs>/monitor_cont with --strip-prefill")
    ap.add_argument("--strip-prefill", action="store_true", help="the monitor sees each turn's reasoning without the prefill the run was sampled with (logs from run_attack.py --prefill)")
    ap.add_argument("--sidecars", default=None, help="rl_data root holding <run>/*.json.gz (optional)")
    ap.add_argument("--judge-backend", default=DEFAULT_BACKEND, choices=BACKENDS)
    ap.add_argument("--judge-model", default=None, help=f"default tinker: {DEFAULT_MODEL['tinker']}; openrouter: {DEFAULT_MODEL['openrouter']}")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="judge output cap, thinking included")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--allow-live", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ledger", default=None, help="append counted judge cost to this rl.budget ledger csv (e.g. rl_data/ledger.csv) so it counts toward its total cap; default none")
    ap.add_argument("--ledger-kind", default="monitor", help=argparse.SUPPRESS)
    ap.add_argument("--set-key", action="store_true", help="prompt (hidden) for your OpenRouter key and save it to ~/.config/sauce/openrouter.env (mode 600), then exit")
    ap.add_argument("--check-key", action="store_true", help="say whether an OpenRouter key is found (never prints it), then exit")
    a = ap.parse_args(argv)
    if a.set_key:
        import getpass
        try:
            p = judges.save_openrouter_key(getpass.getpass("OpenRouter API key (input hidden): "))
        except judges.JudgeError as e:
            sys.exit(f"refusing: {e}")
        print(f"saved to {p} (mode 600)")
        return 0
    if a.check_key:
        try:
            judges.load_openrouter_key()
            print("OpenRouter key found")
            return 0
        except judges.JudgeError as e:
            print(e)
            return 1
    if not a.logs:
        ap.error("logs is required")
    model = a.judge_model or DEFAULT_MODEL[a.judge_backend]
    if not model:
        sys.exit(f"refusing: --judge-model is required for backend {a.judge_backend}")
    if a.max_tokens < 1:
        sys.exit("refusing: --max-tokens must be >= 1")
    price = price_for(a.judge_backend, model)
    mon = get(a.monitor)
    out = a.out or str(Path(a.logs if Path(a.logs).is_dir() else Path(a.logs).parent) / ("monitor_cont" if a.strip_prefill else "monitor"))
    fc = forecast(a.logs, out, mon, a.sidecars, price, a.max_tokens, model, a.strip_prefill)
    print(json.dumps({"monitor": mon.name, "judge_backend": a.judge_backend, "judge_model": model, "price_usd_per_mtok": price.meta(), **fc}, indent=1))
    if not price.verified:
        print(f"WARNING: price for {a.judge_backend}:{model} is UNVERIFIED (conservative upper bound used)", file=sys.stderr)
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
    ledger = None
    if a.ledger:
        from ..budget import TOTAL_CAP_USD, Ledger
        ledger = Ledger(a.ledger)
        bound = min(a.budget_usd, 2 * fc["usd_worst_case"])  # the forecast is one attempt per prompt; a retry on an unparsed reply can double it, and --budget-usd caps it
        if ledger.total_counted() + bound > TOTAL_CAP_USD:
            sys.exit(f"refusing: ledger total {ledger.total_counted():.4f} + worst-case spend {bound:.4f} USD (min of --budget-usd and 2 x the single-attempt forecast) "
                     f"is over the {TOTAL_CAP_USD:g} USD total cap")
    if judge is None:
        try:
            judge = judges.openrouter_judge(model, a.max_tokens, judges.load_openrouter_key()) if a.judge_backend == "openrouter" else judges.tinker_judge(model, a.max_tokens)
        except judges.JudgeError as e:
            sys.exit(f"refusing: {e}")
    res = asyncio.run(run_stage(a.logs, out, mon, judge, a.budget_usd, a.sidecars, a.concurrency, price, a.max_tokens, model, a.judge_backend, a.strip_prefill, ledger, a.ledger_kind))
    print(json.dumps(res, indent=1))
    return 1 if res["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
