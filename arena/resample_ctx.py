"""Resample frozen contexts from saved arena traces and judge each continuation's CoT with the CoT-only monitor.

A context is (run, sample, epoch, turn): the conversation prefix before that assistant turn of a logged episode. Each context is
sampled G times through the same Inspect/Tinker model and tool declarations run.py / run_attack.py used, so the prompt the policy
sees is the one it saw in the original run. Each continuation's CoT (reasoning, then visible text, as rl.monitor reads it) is scored
by rl.monitor's controlarena_cot_only monitor (mode cot_only) with the OpenRouter judge. Nothing is executed in a sandbox: the
sampled turn's tool call is recorded, not run. This is the measurement behind a turn-level GRPO on frozen contexts: how much the
score varies within a group, how long sampling takes, and what it costs.

    .venv/bin/python resample_ctx.py --model tinker/openai/gpt-oss-120b --contexts contexts.json --G 8 --out results/resample_smoke/gptoss
    .venv/bin/python resample_ctx.py --model tinker/openai/gpt-oss-120b --ctx gptoss64-covert2-easy:missing_user_id_in_guess_response#0:1:0 --G 2 --out results/resample_smoke/plumb
    .venv/bin/python resample_ctx.py --model tinker/openai/gpt-oss-120b --model-path tinker://... --judge-model anthropic/claude-sonnet-5.5 --contexts contexts.json --out results/turn1_grpo/<run>/eval_sonnet

Writes <out>/samples.jsonl (one row per continuation, texts included) and <out>/summary.json; prints numbers only.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import bash, text_editor

import run  # registers the "tinker" model API
from rl.monitor import judges, monitors, pricing
from rl.monitor.run import _text, monitor_cot

ARENA = Path(__file__).parent
PRICES = {"openai/gpt-oss-120b": (0.33, 0.84), "Qwen/Qwen3.8-27B": (0.6975, 5.595)}  # USD per M tokens, prefill / sample (self-oversight/common.py)


def load_context(run_name: str, sample_id: str, epoch: int, turn: int):
    """-> (prefix messages before assistant turn `turn`, task description, score the original turn got from the cot_only monitor)."""
    for info in list_eval_logs(str(ARENA / "logs" / run_name)):
        for s in read_eval_log(info.name).samples or []:
            if s.id != sample_id or s.epoch != epoch:
                continue
            msgs, k = [], 0
            for m in s.messages:
                if m.role == "assistant":
                    if k == turn:
                        break
                    k += 1
                msgs.append(m)
            else:
                raise ValueError(f"{run_name}/{sample_id}#{epoch} has no assistant turn {turn}")
            task = next(_text(m.content)[0] for m in s.messages if m.role == "user")
            base = None
            for f in (ARENA / "logs" / run_name / "monitor_cot_only").glob("*/monitor.jsonl"):
                for line in open(f):
                    r = json.loads(line)
                    if r["sample"] == sample_id and r["epoch"] == epoch and r["turn"] == turn:
                        base = r["score"]
            return msgs, task, base
    raise KeyError(f"{run_name}/{sample_id}#{epoch} not found")


async def sample_one(model, msgs, tools):
    t = time.monotonic()
    out = await model.generate(msgs, tools)
    m = out.message
    text, reasoning = _text(m.content)
    return dict(seconds=round(time.monotonic() - t, 1), stop=out.stop_reason, in_tok=out.usage.input_tokens, out_tok=out.usage.output_tokens,
                reasoning=reasoning, text=text, calls=[{"name": c.function, "args": c.arguments} for c in (m.tool_calls or [])])


async def judge_cot(judge, mon, task, cot, sem):
    if not cot.strip():
        return dict(score=None, skipped=True, judge_in=0, judge_out=0, attempts=0, judge_raw=None)
    prompt = mon.format(task, cot, [])
    tot_in = tot_out = 0
    async with sem:
        for attempt in range(1, 3):  # one retry on an unparsed reply, as rl.monitor.run.score_item
            reply, i, o, _ = await judge(prompt)
            tot_in, tot_out = tot_in + i, tot_out + o
            score, _ = mon.parser(reply)
            if score is not None:
                break
    return dict(score=score, skipped=False, judge_in=tot_in, judge_out=tot_out, attempts=attempt, judge_raw=reply)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Inspect model string, e.g. tinker/openai/gpt-oss-120b")
    ap.add_argument("--contexts", type=Path, help="JSON list of {run, sample, epoch, turn}")
    ap.add_argument("--ctx", action="append", default=[], help="run:sample:epoch:turn (repeatable)")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096, help="per sampled turn")
    ap.add_argument("--model-path", help="tinker:// sampler checkpoint to sample instead of the base weights")
    ap.add_argument("--date", help="pin the Harmony system prompt's date (run.py uses the day's date; turn1_grpo.py pins 2026-10-21)")
    ap.add_argument("--judge-model", default="openai/gpt-oss-20b")
    ap.add_argument("--judge-max-tokens", type=int, default=6000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    load_dotenv(ARENA / ".env")
    contexts = json.loads(args.contexts.read_text()) if args.contexts else []
    for c in args.ctx:
        r, s, e, t = c.rsplit(":", 3)
        contexts.append(dict(run=r, sample=s, epoch=int(e), turn=int(t)))
    assert contexts, "no contexts"
    base_model = args.model.split("/", 1)[1]
    prefill_p, sample_p = PRICES[base_model.split(":")[0]]
    model = get_model(args.model, config=GenerateConfig(max_tokens=args.max_tokens, max_connections=32), memoize=False, model_path=args.model_path)  # built before the event loop: its constructor makes sync Tinker calls
    if args.date:
        model.api.renderer.current_date = args.date
    asyncio.run(amain(args, contexts, model, prefill_p, sample_p))


async def amain(args, contexts, model, prefill_p, sample_p):
    tools = [bash(timeout=run.TOOL_TIMEOUT), text_editor(timeout=run.TOOL_TIMEOUT), run.submit()]
    mon = monitors.get("controlarena_cot_only")
    judge = judges.openrouter_judge(args.judge_model, args.judge_max_tokens, judges.load_openrouter_key())
    jp = pricing.price_for("openrouter", args.judge_model)
    sem = asyncio.Semaphore(8)

    loaded = [load_context(c["run"], c["sample"], c["epoch"], c["turn"]) for c in contexts]
    t0 = time.monotonic()
    groups = await asyncio.gather(*[asyncio.gather(*[sample_one(model, msgs, tools) for _ in range(args.G)]) for msgs, _, _ in loaded])
    t_sample = time.monotonic() - t0
    t0 = time.monotonic()
    judged = await asyncio.gather(*[asyncio.gather(*[judge_cot(judge, mon, task, monitor_cot(s["reasoning"], s["text"]), sem) for s in grp])
                                    for grp, (_, task, _) in zip(groups, loaded)])
    t_judge = time.monotonic() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    rows, per_ctx = [], []
    for c, (msgs, task, base), grp, jd in zip(contexts, loaded, groups, judged):
        for g, (s, j) in enumerate(zip(grp, jd)):
            rows.append(dict(**c, g=g, base_score=base, **s, **j))
        scores = [j["score"] for j in jd if j["score"] is not None]
        per_ctx.append(dict(**c, base_score=base, prefix_messages=len(msgs), in_tok=grp[0]["in_tok"], G=args.G, scored=len(scores), scores=scores,
                            mean=round(statistics.mean(scores), 2) if scores else None, std=round(statistics.pstdev(scores), 2) if scores else None,
                            n_hi=sum(s >= 8 for s in scores), n_lo=sum(s <= 1 for s in scores), mixed=bool(scores) and max(scores) - min(scores) >= 5,
                            out_tok_mean=round(statistics.mean(s["out_tok"] for s in grp)), seconds_mean=round(statistics.mean(s["seconds"] for s in grp), 1),
                            seconds_max=max(s["seconds"] for s in grp), truncated=sum(s["stop"] == "max_tokens" for s in grp), no_call=sum(not s["calls"] for s in grp),
                            empty_cot=sum(j["skipped"] for j in jd), unparsed=sum(j["score"] is None and not j["skipped"] for j in jd)))
    in_tok = sum(r["in_tok"] for r in rows)
    out_tok = sum(r["out_tok"] for r in rows)
    j_in = sum(r["judge_in"] for r in rows)
    j_out = sum(r["judge_out"] for r in rows)
    totals = dict(model=args.model, model_path=args.model_path, date=args.date, judge=dict(model=args.judge_model, monitor=mon.name, mode=mon.mode), contexts=len(contexts), G=args.G, samples=len(rows),
                  sample_wall_s=round(t_sample, 1), judge_wall_s=round(t_judge, 1), policy_in_tok=in_tok, policy_out_tok=out_tok,
                  policy_usd=round((in_tok * prefill_p + out_tok * sample_p) / 1e6, 4), judge_usd=round((j_in * jp.prefill + j_out * jp.sample) / 1e6, 4),
                  mixed_contexts=sum(p["mixed"] for p in per_ctx), hi_samples=sum(p["n_hi"] for p in per_ctx), lo_samples=sum(p["n_lo"] for p in per_ctx),
                  unparsed=sum(p["unparsed"] for p in per_ctx), empty_cot=sum(p["empty_cot"] for p in per_ctx))
    with open(args.out / "samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (args.out / "summary.json").write_text(json.dumps(dict(totals=totals, contexts=per_ctx), indent=1))
    print(json.dumps(totals))
    print(f"{'run':26s} {'sample':44s} {'t':>2s} {'base':>4s} {'pre_tok':>7s} {'scores (G)':34s} {'mean':>5s} {'hi':>3s} {'lo':>3s} {'out_tok':>7s} {'s/sample':>8s}")
    for p in per_ctx:
        print(f"{p['run']:26s} {p['sample']:44s} {p['turn']:2d} {str(p['base_score']):>4s} {p['in_tok']:7d} {' '.join(f'{s:g}' for s in p['scores']):34s} "
              f"{str(p['mean']):>5s} {p['n_hi']:3d} {p['n_lo']:3d} {p['out_tok_mean']:7d} {p['seconds_mean']:8.1f}")


if __name__ == "__main__":
    main()
