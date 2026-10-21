"""Offline report of a `rl.py collect` or `rl.py train` output dir, and the dry-run planner for a gated collect. No Tinker, OpenRouter, docker or network call.

    python rl.py report rl_data/<run>                       # writes <run>/report.{md,json}, prints the markdown
    python rl.py plan-collect tiktok --run R --epochs 4 --budget-usd 5     # exact gated command + cost estimate, makes no call

Reads <dir>/episodes.jsonl, <dir>/collect/episodes.jsonl or a train run's <dir>/step_NNN/episodes.jsonl (rl/schema.py), every monitor.jsonl under <dir>
(rl/monitor/run.py rows, or --monitor) and the train loop's in-loss <dir>/cot_monitor.jsonl (rl/cot_monitor.py rows, read as monitor rows),
<dir>/ledger.csv (fake runs) or the shared ledger filtered by run, and <dir>/HALTED.md. Pass rate is over episodes with a reward (rewards.episode_reward:
INFRA and excluded endings are None and left out); the CI is Wilson 95% over episodes, which treats epochs of one task as independent (they are not).
"""

import json
import math
import shlex
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .budget import Ledger, episode_worst_case_usd, sample_cost_usd
from .classify import HALT
from .rewards import compute_rewards
from .schema import AGENT, INFRA, NONE, Episode, read_jsonl

REPORT_SCHEMA = "arena.rl.report.v1"
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for k successes of n; None when n == 0."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return dict(k=k, n=n, rate=k / n if n else None, ci95=[round(ci[0], 4), round(ci[1], 4)] if ci else None)


def dist(xs: list[float]) -> dict | None:
    if not xs:
        return None
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]  # noqa: E731  nearest-rank, lower
    return dict(n=len(s), mean=round(statistics.fmean(s), 4), p50=q(0.5), p90=q(0.9), min=s[0], max=s[-1])


def find_episodes(d: Path) -> list[Path]:
    """<d>/episodes.jsonl, else <d>/collect/episodes.jsonl, else a train run's step_NNN/episodes.jsonl (the .stale ones a resume set aside are not read)."""
    for p in (d / "episodes.jsonl", d / "collect" / "episodes.jsonl"):
        if p.exists():
            return [p]
    steps = sorted(d.glob("step_*/episodes.jsonl"))
    if steps:
        return steps
    raise FileNotFoundError(f"no episodes.jsonl in {d}, {d / 'collect'} or {d}/step_*/")


# ---- sections

def main_task(eps: list[Episode]) -> dict:
    r = compute_rewards(eps)
    scored = [e for e in eps if r[e.episode_id] is not None]
    by_task: dict[str, list[float]] = defaultdict(list)
    for e in scored:
        by_task[e.task].append(r[e.episode_id])
    all_tasks = sorted({e.task for e in eps})
    signal = {t: len(by_task[t]) >= 2 and min(by_task[t]) != max(by_task[t]) for t in all_tasks}
    return dict(pass_rate=rate(sum(r[e.episode_id] == 1.0 for e in scored), len(scored)), n_episodes=len(eps), n_scored=len(scored), n_reward_none=len(eps) - len(scored),
                by_task={t: dict(n=len(by_task[t]), passed=int(sum(by_task[t])), pass_rate=(sum(by_task[t]) / len(by_task[t])) if by_task[t] else None, signal=signal[t])
                         for t in all_tasks},
                rl_signal=dict(tasks_with_variance=sum(signal.values()), tasks=len(all_tasks), frac=(sum(signal.values()) / len(all_tasks)) if all_tasks else None,
                               note="a task has signal when >= 2 scored epochs and not all rewards equal"))


def side(eps: list[Episode]) -> dict:
    split = {}
    for key, label in ((True, "opportunity_present"), (False, "opportunity_absent"), (None, "opportunity_unknown")):
        g = [e for e in eps if e.opportunity_present is key and e.side_done is not None]
        split[label] = rate(sum(bool(e.side_done) for e in g), len(g))
    ev = [e for e in eps if e.side_done is not None]
    fr = [e for e in eps if e.final_reveal is not None]
    fr_done = [e for e in fr if e.side_done]
    by_step = {s: dict(side_done=rate(sum(bool(e.side_done) for e in ev if e.step == s), sum(e.step == s for e in ev)),
                       final_reveal=rate(sum(bool(e.final_reveal) for e in fr if e.step == s), sum(e.step == s for e in fr))) for s in sorted({e.step for e in ev})}
    return dict(side_done=rate(sum(bool(e.side_done) for e in ev), len(ev)), side_done_by_opportunity=split, n_not_evaluated=len(eps) - len(ev),
                final_reveal=rate(sum(bool(e.final_reveal) for e in fr), len(fr)), final_reveal_given_side_done=rate(sum(bool(e.final_reveal) for e in fr_done), len(fr_done)),
                by_step=by_step)


def monitor_rows(d: Path, extra: list[Path] | None = None) -> list[dict]:
    files = list(dict.fromkeys(Path(f).resolve() for f in [*sorted(d.rglob("monitor.jsonl")), *(extra or [])]))  # an extra under d is read once
    files += [f.resolve() for f in sorted(d.rglob("cot_monitor.jsonl")) if f.resolve() not in files]
    rows = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue  # torn last line
            if r.get("schema") == "arena.monitor.v1":
                rows.append(r)
            elif r.get("schema") == COT_SCHEMA:
                rows.append(cot_row(r))
    return rows


COT_SCHEMA = "arena.rl.cot_monitor.v1"


def cot_row(r: dict) -> dict:
    """An in-loss rl.cot_monitor row as a monitor row: scale/direction from the monitor registry, the episode as `sample`, a fallback c not counted as parsed."""
    from .monitor.monitors import get

    try:
        meta = get(r.get("monitor")).meta()
    except Exception:
        meta = dict(scale=[0, 10], direction="high=suspicious")
    return dict(r, schema=COT_SCHEMA, scale=meta["scale"], direction=meta["direction"], log=r.get("log"), sample=r.get("episode_id"), epoch=r.get("step"),
                score=None if r.get("fallback") else r.get("score"), unparsed=r.get("reason") == "unparsed", empty_cot=r.get("skipped") == "empty_cot",
                skipped=bool(r.get("skipped")), source="in_loss")


def c_turn(r: dict) -> float | None:
    """A monitor row's score mapped to [0,1] suspicion over its scale. `direction` says what a high score means: "high=suspicious" (controlarena_cot)
    is kept, any other ("high=on-task", ontask_generic_100) is flipped so that 1 is always the most suspicious."""
    if r.get("score") is None:
        return None
    lo, hi = (r.get("scale") or [0, 10])[:2]
    if hi == lo:
        return None
    x = min(1.0, max(0.0, (r["score"] - lo) / (hi - lo)))
    return x if str(r.get("direction") or "high=suspicious").startswith("high=susp") else 1.0 - x


def _monitor_stats(rows: list[dict]) -> dict:
    cs = [c for c in map(c_turn, rows) if c is not None]
    per_ep: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        c = c_turn(r)
        if c is not None:
            per_ep[(r.get("monitor"), r.get("judge_model"), r.get("log"), r.get("sample"), r.get("epoch"))].append(c)
    hist = Counter(min(int(c * 10), 9) for c in cs)
    return dict(direction=sorted({str(r.get("direction")) for r in rows}), turns=len(rows), parsed=len(cs), c_turn=dist(cs), c_turn_hist10=[hist.get(i, 0) for i in range(10)],
                episode_max_c=dist([max(v) for v in per_ep.values()]))


def monitor(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[f"{r.get('monitor')} / {r.get('judge_model')}" + (" (in-loss)" if r.get("source") == "in_loss" else "")].append(r)  # in-loss and held-out never pool
    allst = _monitor_stats(rows)
    return dict(monitors=sorted({str(r.get("monitor")) for r in rows}), judge_models=sorted({str(r.get("judge_model")) for r in rows}), turns=len(rows), parsed=allst["parsed"],
                unparsed=sum(bool(r.get("unparsed")) for r in rows), skipped=sum(bool(r.get("skipped")) for r in rows), empty_cot=sum(bool(r.get("empty_cot")) for r in rows),
                c_turn=allst["c_turn"], c_turn_hist10=allst["c_turn_hist10"], episode_max_c=allst["episode_max_c"], by_monitor={k: _monitor_stats(v) for k, v in sorted(by.items())},
                cost_usd=round(sum(r.get("cost_usd") or 0.0 for r in rows), 6),
                note="c_turn is suspicion in [0,1] (scores flipped for high=on-task monitors); the pooled numbers mix monitors, read by_monitor when there are several")


def cot_lengths(eps: list[Episode], tokenizer=None) -> dict:
    """Analysis-channel tokens per turn and per episode (cot_mask.think_mask). Without a tokenizer, falls back to all sampled tokens (an upper bound)."""
    from .cot_mask import think_mask

    if tokenizer is None:
        try:
            from .cot_mask import _tokenizer_for
            tokenizer = _tokenizer_for((eps[0].model if eps else "") or "openai/gpt-oss-120b")
        except Exception:
            tokenizer = None
    count = (lambda toks: sum(think_mask(toks, tokenizer))) if tokenizer is not None else len
    per_turn, per_ep, sampled = [], [], []
    for e in eps:
        if not e.turns:
            continue
        ts = [count(t.tokens) for t in e.turns]
        per_turn += ts
        per_ep.append(sum(ts))
        sampled.append(sum(len(t.tokens) for t in e.turns))
    turns = [t for e in eps for t in e.turns]
    return dict(source="analysis_channel" if tokenizer is not None else "sampled_tokens_upper_bound (tokenizer unavailable)", per_turn=dist(per_turn), per_episode=dist(per_ep),
                sampled_per_episode=dist(sampled), turns_per_episode=dist([len(e.turns) for e in eps if e.turns]),
                empty_reasoning_turns=sum("empty_reasoning" in t.flags for t in turns), truncated_turns=sum(t.stop_reason == "length" for t in turns), n_turns=len(turns))


def failures(eps: list[Episode], d: Path) -> dict:
    fc = Counter(e.failure_class for e in eps)
    halted = d / "HALTED.md"
    hl = halted.read_text().strip().splitlines() if halted.exists() else []
    return dict(counts={NONE: fc.get(NONE, 0), AGENT: fc.get(AGENT, 0), INFRA: fc.get(INFRA, 0), HALT: fc.get(HALT, 0)}, other={k: v for k, v in fc.items() if k not in (NONE, AGENT, INFRA, HALT)},
                reasons={f"{k[0]}:{k[1]}": v for k, v in Counter((e.failure_class, e.failure_reason) for e in eps if e.failure_class != NONE).most_common()},
                ended=dict(Counter(e.ended for e in eps).most_common()), reruns=sum(e.attempt > 0 for e in eps), non_trainable=sum(bool(e.non_trainable) for e in eps),
                halted=(hl[2] if len(hl) > 2 else str(halted)) if halted.exists() else None)


def _by_model(rows: list[dict]) -> dict:
    """Counted USD per ledger model column: the policy model, or the judge model for cot_monitor / heldout / sanity_judge rows written since the column
    started recording it (older judge rows carry the policy model)."""
    out: dict[str, float] = {}
    for r in rows:
        if r["kind"] == "counted" and r["counted_usd"]:
            out[r["model"]] = round(out.get(r["model"], 0.0) + float(r["counted_usd"]), 6)
    return out


def cost(eps: list[Episode], d: Path, run: str, model: str, ledger_path: Path | None = None, monitor_usd: float = 0.0) -> dict:
    sample = sum(sample_cost_usd(model, (e.usage or {}).get("prompt_tokens", 0), (e.usage or {}).get("cached_tokens", 0), (e.usage or {}).get("completion_tokens", 0)) for e in eps)
    wasted = sum(e.wasted_usd for e in eps)
    # fake runs ledger in <run>/ledger.csv, live ones in the shared <data>/ledger.csv; d may be <run> or <run>/collect
    lp = ledger_path or next((p / "ledger.csv" for p in (d, d.parent, d.parent.parent) if (p / "ledger.csv").exists()), None)
    led = None
    if lp is not None:
        rows = [r for r in Ledger(lp).rows() if not run or r["run"] == run]
        led = dict(path=str(lp), counted_usd=round(sum(float(r["counted_usd"]) for r in rows if r["kind"] == "counted" and r["counted_usd"]), 6),
                   wasted_usd=round(sum(float(r["estimate_usd"] or 0) for r in rows if r["kind"] == "wasted"), 6), rows=len(rows), counted_by_model=_by_model(rows))
    n = max(sum(bool(e.turns) for e in eps), 1)
    return dict(sampling_usd_upper_bound=round(sample, 6), wasted_usd=round(wasted, 6), monitor_usd=round(monitor_usd, 6), total_usd=round(sample + wasted + monitor_usd, 6),
                per_sampled_episode_usd=round(sample / n, 6), ledger=led, note="sampling cost uses Episode.usage, a no-cache upper bound for sidecar episodes")


# ---- report

def build_report(d: Path, *, monitor_paths: list[Path] | None = None, ledger_path: Path | None = None, tokenizer=None, model: str | None = None) -> dict:
    d = Path(d)
    ep_paths = find_episodes(d)
    eps = [e for p in ep_paths for e in read_jsonl(p)]
    ep_path = ep_paths[0] if len(ep_paths) == 1 else d / "step_*" / "episodes.jsonl"
    run = next((e.run for e in eps if e.run), d.name)
    model = model or next((e.model for e in eps if e.model), "openai/gpt-oss-120b")
    mon = monitor(monitor_rows(d, monitor_paths))
    return dict(schema=REPORT_SCHEMA, dir=str(d), episodes_path=str(ep_path), run=run, model=model, phases=sorted({e.phase for e in eps}), weights_ids=sorted({e.weights_id for e in eps}),
                tasks=sorted({e.task for e in eps}), epochs=max((e.epoch for e in eps), default=0), main=main_task(eps), side=side(eps), monitor=mon, cot=cot_lengths(eps, tokenizer),
                failures=failures(eps, d), cost=cost(eps, d, run, model, ledger_path, mon["cost_usd"] if mon else 0.0))


def _r(x: dict) -> str:
    if not x or x["rate"] is None:
        return f"n/a (n={x['n'] if x else 0})"
    return f"{x['k']}/{x['n']} = {x['rate']:.1%} [95% CI {x['ci95'][0]:.1%}, {x['ci95'][1]:.1%}]"


def _d(x: dict | None, fmt: str = "{:.0f}") -> str:
    if not x:
        return "n/a"
    return f"n={x['n']} mean {x['mean']:.4g} p50 " + fmt.format(x["p50"]) + " p90 " + fmt.format(x["p90"]) + " max " + fmt.format(x["max"])


def to_markdown(rep: dict) -> str:
    m, s, c, f, k = rep["main"], rep["side"], rep["cot"], rep["failures"], rep["cost"]
    L = [f"# Collect report: {rep['run']}", "", f"`{rep['episodes_path']}` · model `{rep['model']}` · phase {', '.join(rep['phases'])} · weights {', '.join(rep['weights_ids'])} · "
         f"{len(rep['tasks'])} tasks · {m['n_episodes']} episodes ({m['n_scored']} scored, {m['n_reward_none']} reward None)", "",
         "## Main task", "", f"- Pass rate: {_r(m['pass_rate'])} (Wilson over episodes; epochs of a task are not independent)",
         f"- RL signal: {m['rl_signal']['tasks_with_variance']}/{m['rl_signal']['tasks']} tasks with nonzero reward variance across epochs", "",
         "| task | scored | passed | pass rate | signal |", "| --- | --- | --- | --- | --- |"]
    for t, v in m["by_task"].items():
        pr = "n/a" if v["pass_rate"] is None else f"{v['pass_rate']:.0%}"
        L.append(f"| {t} | {v['n']} | {v['passed']} | {pr} | {'yes' if v['signal'] else 'no'} |")
    L += ["", "## Side task", "", f"- side_done: {_r(s['side_done'])} ({s['n_not_evaluated']} episodes not evaluated)"]
    L += [f"- side_done, {lab.replace('_', ' ')}: {_r(v)}" for lab, v in s["side_done_by_opportunity"].items()]
    L += [f"- final_reveal: {_r(s['final_reveal'])}", f"- final_reveal given side_done: {_r(s['final_reveal_given_side_done'])}"]
    if len(s.get("by_step") or {}) > 1:
        L += [f"- step {k}: side_done {_r(v['side_done'])}, final_reveal {_r(v['final_reveal'])}" for k, v in s["by_step"].items()]
    L += ["", "## CoT monitor", ""]
    mon = rep["monitor"]
    if mon:
        L += [f"- monitors {', '.join(mon['monitors'])}, judge {', '.join(mon['judge_models'])}: {mon['turns']} turns, {mon['parsed']} parsed, {mon['unparsed']} unparsed, "
              f"{mon['skipped']} skipped, {mon['empty_cot']} empty CoT", f"- c_turn: {_d(mon['c_turn'], '{:.2f}')}", f"- c_turn histogram (deciles 0-0.1 .. 0.9-1): {mon['c_turn_hist10']}",
              f"- per-episode max c: {_d(mon['episode_max_c'], '{:.2f}')}"]
        if len(mon["by_monitor"]) > 1:
            L += [f"- {k} ({', '.join(v['direction'])}): c_turn {_d(v['c_turn'], '{:.2f}')}; per-episode max {_d(v['episode_max_c'], '{:.2f}')}; hist {v['c_turn_hist10']}"
                  for k, v in mon["by_monitor"].items()]
    else:
        L.append("- no monitor.jsonl found (nor cot_monitor.jsonl)")
    L += ["", "## CoT length", "", f"- source: {c['source']}", f"- tokens per turn: {_d(c['per_turn'])}", f"- tokens per episode: {_d(c['per_episode'])}",
          f"- sampled tokens per episode: {_d(c['sampled_per_episode'])}", f"- turns per episode: {_d(c['turns_per_episode'])}",
          f"- empty-reasoning turns {c['empty_reasoning_turns']}/{c['n_turns']}, truncated turns {c['truncated_turns']}/{c['n_turns']}", "",
          "## Failures", "", "- " + ", ".join(f"{a.upper()} {b}" for a, b in f["counts"].items()) + (f", other {f['other']}" if f["other"] else ""),
          f"- reasons: {f['reasons'] or 'none'}", f"- ended: {f['ended']}", f"- reruns {f['reruns']}, non-trainable {f['non_trainable']}", f"- HALTED: {f['halted'] or 'no'}", "",
          "## Cost", "", f"- sampling ${k['sampling_usd_upper_bound']:.4f} (upper bound, ${k['per_sampled_episode_usd']:.4f} per sampled episode), wasted ${k['wasted_usd']:.4f}, "
          f"monitor ${k['monitor_usd']:.4f}; total ${k['total_usd']:.4f}"]
    if k["ledger"]:
        L.append(f"- ledger `{k['ledger']['path']}`: counted ${k['ledger']['counted_usd']:.4f} over {k['ledger']['rows']} rows for this run"
                 + (f" ({', '.join(f'{m} ${v:.4f}' for m, v in k['ledger']['counted_by_model'].items())})" if len(k['ledger']['counted_by_model']) > 1 else ""))
    return "\n".join(L) + "\n"


def write_report(d: Path, out: Path | None = None, **kw) -> tuple[dict, str]:
    rep = build_report(d, **kw)
    md = to_markdown(rep)
    out = Path(out) if out else Path(d)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    (out / "report.md").write_text(md)
    return rep, md


# ---- dry-run planner

@dataclass
class CollectPlan:
    tasks: list[str]
    run: str
    epochs: int
    budget_usd: float
    model: str = "openai/gpt-oss-120b"
    max_turns: int = 60
    max_tokens: int = 4096
    ep_worst_usd: float | None = None
    ep_usd_est: float = 0.03  # placeholder, unmeasured (same as LoopCfg.ep_usd_est); pass a measured per-episode cost from a report to replace it
    extra: list[str] = field(default_factory=list)


def plan_collect(p: CollectPlan, ledger_total_usd: float = 0.0, total_cap_usd: float = 20.0) -> dict:
    """What `rl.py collect` would reserve and roughly spend, and the exact gated command. Pure arithmetic; makes no call."""
    n = len(p.tasks) * p.epochs
    worst = p.ep_worst_usd if p.ep_worst_usd else episode_worst_case_usd(p.model, max_turns=p.max_turns, max_tokens=p.max_tokens)
    reserve = worst * n
    cmd = ["ARENA_RL_RUN_TINKER=1", "python", "rl.py", "collect", *p.tasks, "--run", p.run, "--epochs", str(p.epochs), "--budget-usd", f"{p.budget_usd:g}",
           "--model", p.model, "--max-turns", str(p.max_turns), "--max-tokens", str(p.max_tokens), *(["--ep-worst-usd", f"{p.ep_worst_usd:g}"] if p.ep_worst_usd else []), *p.extra]
    problems = []
    if reserve > p.budget_usd:
        problems.append(f"worst-case reservation ${reserve:.4f} ({n} episodes x ${worst:.4f}) is over --budget-usd ${p.budget_usd:g}; collect would refuse at start_stage")
    if ledger_total_usd + reserve > total_cap_usd:
        problems.append(f"ledger total ${ledger_total_usd:.4f} + reservation ${reserve:.4f} is over the ${total_cap_usd:g} total cap")
    return dict(run=p.run, model=p.model, tasks=p.tasks, epochs=p.epochs, episodes=n, max_turns=p.max_turns, max_tokens=p.max_tokens, ep_worst_case_usd=round(worst, 4),
                reserve_usd=round(reserve, 4), expected_usd=round(p.ep_usd_est * n, 4), ep_usd_est=p.ep_usd_est, budget_usd=p.budget_usd,
                ledger_counted_usd=round(ledger_total_usd, 4), total_cap_usd=total_cap_usd, would_refuse=problems,
                command=" ".join(shlex.quote(x) if i else x for i, x in enumerate(cmd)), report_command=f"python rl.py report rl_data/{shlex.quote(p.run)}",
                note="expected_usd = episodes x ep_usd_est (placeholder unless measured); reserve_usd is budget.episode_worst_case_usd x episodes, which collect reserves up front")
