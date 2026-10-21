"""Iterate on self-instruction elicitations at the copy-.env decision point (single turn, the hand-fed
`ls -la` showing the .env, as copyenv_prefill.py). Arms come from a JSON file so wording can be varied
without code edits:

    {"arm_name": {"user": "$V2", "system": "", "prefill": "We need to ...", "echo": null, "close": false}}

    user     appended to the user message; "$V2" / "$CTRL" expand to copyenv_prompted's covert_v2 / control
    system   appended to the harness system prompt (empty for none)
    prefill  text placed at the start of the analysis channel (null for none)
    echo     text the model is asked to write itself (bootstrap); scored for echo, stripped like a prefill
    close    reference arm only: close the analysis block right after the prefill, so the model acts with no
             reasoning of its own (outside the paper's method; shows whether it copies without deliberating)

The judge, detectors and tasks are copyenv_prefill.py's, unchanged. Judge output is reported as score
distributions on the full CoT (what a monitor reads) and the continuation (excluding the self-instruction).

    python copyenv_elicit.py arms/round1.json -n 8 --n-noenv 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import warnings
from pathlib import Path

import ledger
from common import RESULTS, get_model, load_key, renderer_for
from copyenv_prefill import ANALYSIS, CTRL, V2, judge_rows, split_si
from copyenv_probe import HARNESS, HELDOUT_TASKS, TASKS, TOOLS, VERBAL_RE, build_messages, is_copy_call, tool_calls_of
from cot_penalty import think_mask
from sample import Runner

EXPAND = {"$V2": V2, "$CTRL": CTRL}
CLOSE = "<|end|><|start|>assistant"


def messages(ren, task: dict, arm: dict) -> list[dict]:
    base = build_messages(ren, "student", task)
    n_prefix = len(ren.create_conversation_prefix_with_tools(TOOLS, system_prompt=HARNESS))
    msgs = list(ren.create_conversation_prefix_with_tools(TOOLS, system_prompt=HARNESS + EXPAND.get(arm.get("system", ""), arm.get("system", "")))) + base[n_prefix:]
    user = next(m for m in msgs if m["role"] == "user")
    user["content"] += EXPAND.get(arm.get("user", ""), arm.get("user", ""))
    return msgs


def prefill_text(arm: dict) -> str | None:
    if not arm.get("prefill"):
        return None
    return ANALYSIS + arm["prefill"] + (CLOSE if arm.get("close") else "")


async def sample_arm(name: str, arm: dict, tok, ren, n_env: int, n_noenv: int, max_tokens: int, budget: float) -> list[dict]:
    pre = prefill_text(arm)
    pre_toks = tok.encode(pre, add_special_tokens=False) if pre else []
    runner = Runner(get_model("gpt-oss-120b"), budget, renderer=ren, backend="tinker")
    jobs = [(t, n_env if t["env"] else n_noenv) for t in TASKS + HELDOUT_TASKS]
    jobs = [(t, n) for t, n in jobs if n]
    outs = await asyncio.gather(*[runner.sample(messages(ren, t, arm), n=n, max_tokens=max_tokens, prefill=pre) for t, n in jobs])
    rows = []
    for (task, _), rollouts in zip(jobs, outs):
        for r in rollouts:
            resp = pre_toks + list(r.tokens)
            try:
                msg, term = ren.parse_response(resp)
                calls, clean = tool_calls_of(msg), bool(term.is_clean)
            except Exception:  # noqa: BLE001 - keep the rollout, count it as unclean
                calls, clean = [], False
            mask = think_mask(resp, tok)
            full = tok.decode([t for t, m in zip(resp, mask) if m])
            cont, echoed = split_si(full, arm.get("prefill") or arm.get("echo"))
            rows.append({"arm": name, "task": task["id"], "env": task["env"],
                         "copied": any(is_copy_call(c) for c in calls), "copy_is_first": bool(calls) and is_copy_call(calls[0]),
                         "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
                         "thinking_full": full, "thinking_cont": cont, "echoed": echoed if arm.get("echo") else None,
                         "thinking_tokens": sum(mask), "stop_reason": r.stop_reason, "clean": clean})
    runner.to_ledger("copyenv", f"elicit-{name}")
    return rows


def dist(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return f"mean {statistics.mean(xs):.3f}, min {xs[0]:.2f}, p10 {q(.1):.2f}, median {q(.5):.2f}"


def summarize(rows: list[dict], arms: dict) -> list[dict]:
    table = []
    for name, arm in arms.items():
        g = [r for r in rows if r["arm"] == name]
        env, noe = [r for r in g if r["env"]], [r for r in g if not r["env"]]
        cp = [r for r in env if r["copied"]]
        table.append({
            "arm": name,
            "copied, .env present": f"{len(cp)}/{len(env)}",
            "copied, no .env": f"{sum(r['copied'] for r in noe)}/{len(noe)}",
            "judge score, full CoT, copying rollouts": dist([r["cot_full"] for r in cp]),
            "judge score, continuation, copying rollouts": dist([r["cot"] for r in cp]),
            "copying rollouts with no VERBAL_RE match in the full CoT": f"{sum(not VERBAL_RE.search(r['thinking_full']) for r in cp)}/{len(cp)}",
            "judge score, full CoT, no-.env rollouts": dist([r["cot_full"] for r in noe]),
            "rollouts that opened with the self-instruction (bootstrap)": f"{sum(bool(r['echoed']) for r in g)}/{len(g)}" if arm.get("echo") else "n/a",
            "median CoT tokens": statistics.median(r["thinking_tokens"] for r in g),
            "unparsed or truncated": f"{sum(not r['clean'] or r['stop_reason'] == 'length' for r in g)}/{len(g)}"})
    return table


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms_file")
    ap.add_argument("-n", type=int, default=8, help="samples per .env task")
    ap.add_argument("--n-noenv", type=int, default=4, help="samples per no-.env task")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--budget", type=float, default=0.5, help="USD sampling cap per arm")
    ap.add_argument("--dump", default=None, help="print the rendered prompt tail for this arm and exit")
    args = ap.parse_args()

    arms = json.loads(Path(args.arms_file).read_text())
    load_key()
    warnings.simplefilter("ignore", UserWarning)
    tok, ren = renderer_for(get_model("gpt-oss-120b"))
    if args.dump:
        arm = arms[args.dump]
        print(tok.decode(ren.build_generation_prompt(messages(ren, TASKS[0], arm), prefill=prefill_text(arm)).to_ints())[-2500:])
        return

    out = RESULTS / "copyenv" / "elicit" / Path(args.arms_file).stem
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for rs in await asyncio.gather(*[sample_arm(k, a, tok, ren, args.n, args.n_noenv, args.max_tokens, args.budget)
                                               for k, a in arms.items()]) for r in rs]
    judge_usd = await judge_rows(rows)
    ledger.append("copyenv", f"elicit-judge-{out.name}", "openai/gpt-oss-20b", "sample", counted_usd=judge_usd,
                  note=f"{2 * len(rows)} CoT views")
    with (out / "eval.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    table = summarize(rows, arms)
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "arms": arms, "judge_usd": judge_usd, "table": table}, indent=1))
    for t in table:
        print(json.dumps(t, ensure_ascii=False))
    print(f"judge ${judge_usd:.3f}; wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
