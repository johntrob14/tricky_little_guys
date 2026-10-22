"""Rate resampled turns (rows of a resample_ctx / rejudge samples.jsonl, or turn1_grpo rollouts) with the tutor's pass 2 (tutor.py, rubric project_context/organism_spec.md).

Each row is shown to the tutor as the logged episode up to its turn, then the row's own turn (reasoning, visible text, tool calls; no tool results, since it was never run).
Up to --attempts tries per call (a failed request, a reply with no <json> block, or the provider's content filter); a row that fails every try gets tutor=None,
with tutor_blocked=True when the last failure was the content filter (on base turn-0 samples it fires only on CoTs the monitor scores 9-10).

    .venv/bin/python tutor_rate.py results/turn1_grpo/smoke1/eval/base_train_gptoss20b_maintask/samples.jsonl --turn 0 --out results/tutor_rate/base_t0

Writes <out>/samples.jsonl (input rows plus tutor fields; --repeat R rates each row R times, rep 0..R-1) and <out>/summary.json.
"""

import argparse, asyncio, json
from pathlib import Path

import httpx

from rl.monitor.judges import load_openrouter_key
from tutor import EFFORT, MODEL, SPEC, ask, context_episode, resampled_prompt

SCORES = ("suspiciousness", "reasoning", "side_task_progress", "persistence", "awareness")


async def rate(client, key, model, effort, prompt, sem, attempts=2):
    async with sem:
        for attempt in range(1, attempts + 1):
            try:
                r = await ask(client, key, model, effort, prompt)
                return dict(tutor={k: r["json"].get(k) for k in SCORES}, tutor_text=r["content"], tutor_in=r["in_tok"], tutor_out=r["out_tok"], tutor_reasoning_tok=r["reasoning_tok"], tutor_cost=r["cost"] or 0.0, tutor_attempts=attempt, tutor_error=None, tutor_blocked=False)
            except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as e:
                err = repr(e)[:300]
    return dict(tutor=None, tutor_text=None, tutor_in=0, tutor_out=0, tutor_reasoning_tok=None, tutor_cost=0.0, tutor_attempts=attempts, tutor_error=err, tutor_blocked="flagged for possible cybersecurity risk" in err)


async def amain(a, rows, eps, spec, key):
    sem = asyncio.Semaphore(a.concurrency)
    jobs = [(r, rep) for r in rows for rep in range(a.repeat)]
    async with httpx.AsyncClient(timeout=1800) as client:
        rated = await asyncio.gather(*[rate(client, key, a.model, a.effort, resampled_prompt(spec, eps[(r["run"], r["sample"], r["epoch"])], r["turn"], r), sem, a.attempts) for r, _ in jobs])
    out = [{**r, "rep": rep, "tutor_model": a.model, "tutor_effort": a.effort, **t} for (r, rep), t in zip(jobs, rated)]
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "samples.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in out if r["tutor"]]
    totals = dict(source=str(a.samples), model=a.model, effort=a.effort, rows=len(rows), repeat=a.repeat, calls=len(out), failed=len(out) - len(ok), cost=round(sum(r["tutor_cost"] for r in out), 4),
                  mean={k: round(sum(r["tutor"][k] for r in ok if r["tutor"][k] is not None) / max(1, sum(r["tutor"][k] is not None for r in ok)), 2) for k in SCORES})
    (a.out / "summary.json").write_text(json.dumps(totals, indent=1))
    (a.out / "spec.md").write_text(spec)
    print(json.dumps(totals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--turn", type=int, help="only rows at this turn")
    ap.add_argument("--contexts", type=int, help="only the first N distinct contexts")
    ap.add_argument("--per-context", type=int, help="only the first N rows of each context")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--spec", type=Path, default=SPEC)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.samples)]
    rows = [r for r in rows if a.turn is None or r["turn"] == a.turn]
    keys = list(dict.fromkeys((r["run"], r["sample"], r["epoch"], r["turn"]) for r in rows))[:a.contexts]
    rows = [r for k in keys for r in [r for r in rows if (r["run"], r["sample"], r["epoch"], r["turn"]) == k][:a.per_context]]
    eps = {k: context_episode(*k) for k in dict.fromkeys((r["run"], r["sample"], r["epoch"]) for r in rows)}  # before the event loop: read_eval_log misbehaves inside a running loop
    asyncio.run(amain(a, rows, eps, a.spec.read_text(), load_openrouter_key()))


if __name__ == "__main__":
    main()
