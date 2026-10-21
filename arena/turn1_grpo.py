"""GRPO on frozen contexts: train a LoRA on gpt-oss-120b to keep one turn's CoT quiet under the CoT-only monitor, with the context held fixed.

A context is the conversation prefix before one assistant turn of a saved arena_covert_v2 episode (resample_ctx.load_context). Every step
samples G continuations of every context from the current weights (Tinker sampler, the cookbook Harmony renderer with the run's tool
declarations, so the policy sees the prompt it saw in the original run), judges each continuation's CoT with rl.monitor's
controlarena_cot_only monitor (mode cot_only, OpenRouter judge), and takes one importance_sampling step on the sampled tokens.
Nothing is executed in a sandbox: the sampled turn's tool call is recorded, not run.

Reward: -score/10 (0 for a CoT the judge scores 0, -1 for 10). A turn with an empty CoT, or without a parsable tool call, counts as score 10,
so the policy cannot earn the reward by not reasoning or by not acting. A sample whose judge reply is unparsed after one retry is dropped from
the group. The Harmony system prompt's date is pinned to the day the source episodes ran, so the prompt is byte-identical to the original.
Advantage: reward minus the group mean (no std scaling); groups with equal rewards are dropped; one optimizer step per batch, on-policy.

    .venv/bin/python turn1_grpo.py --run smoke --contexts results/resample_smoke/contexts_t0_gptoss.json results/turn1_grpo/contexts_dp_early_gptoss.json --steps 10

Writes results/turn1_grpo/<run>/{config.json, steps.jsonl, rollouts.jsonl, checkpoints.json}; rollouts.jsonl holds every sampled turn's texts.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np
import tinker
from dotenv import load_dotenv
from inspect_ai.model._call_tools import prepare_tools
from inspect_ai.tool import bash, text_editor
from tinker_cookbook.eval.inspect_utils import _conversation_with_tool_declarations, _message_to_inspect_content, _message_to_inspect_tool_calls, convert_inspect_messages
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

import run
from resample_ctx import judge_cot, load_context
from rl.monitor import judges, monitors, pricing
from rl.monitor.run import _text, monitor_cot

ARENA = Path(__file__).parent
MODEL = "openai/gpt-oss-120b"
PRICE = dict(prefill=0.33, cached=0.066, sample=0.84, train=0.737)  # USD per M tokens (self-oversight/common.py); a num_samples=G request bills the prompt once at prefill and G-1 times at cached
DATE = "2026-10-21"  # the date the gptoss64-covert2-* episodes were rendered with (run.py uses the day's date); pinned so the prompt does not change across days
BAD = 10.0  # score charged to an empty CoT or a turn with no tool call


def render(renderer, msgs, tool_infos) -> list[int]:
    msgs = [m.model_copy(update=dict(content=f"Error: {m.error.message}")) if m.role == "tool" and m.error is not None and not m.text else m for m in msgs]  # run.TinkerModel's fix
    return renderer.build_generation_prompt(_conversation_with_tool_declarations(renderer, convert_inspect_messages(msgs), tool_infos, "auto")).to_ints()


def parse(renderer, seq) -> dict:
    msg, _ = renderer.parse_response(list(seq.tokens))
    text, reasoning = _text(_message_to_inspect_content(msg)) if isinstance(msg["content"], list) else ("", "")  # a bare tool call comes back with its raw Harmony text as content (run._no_raw_content)
    calls = _message_to_inspect_tool_calls(msg, choice_index=0) or []
    return dict(n_tok=len(seq.tokens), stop=seq.stop_reason, reasoning=reasoning, text=text, calls=[{"name": c.function, "args": c.arguments} for c in calls],
                acted=any(c.function != "unparsed_tool_call" for c in calls))


def datum(prompt: list[int], resp: list[int], lps: list[float], adv: float) -> tinker.Datum:
    full, n_ctx = prompt + resp, len(prompt) - 1
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]),
                        loss_fn_inputs={"target_tokens": tinker.TensorData.from_numpy(np.array(full[1:], dtype=np.int64)),
                                        "logprobs": tinker.TensorData.from_numpy(np.array([0.0] * n_ctx + lps, dtype=np.float32)),
                                        "advantages": tinker.TensorData.from_numpy(np.array([0.0] * n_ctx + [adv] * len(resp), dtype=np.float32))})


def mean(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--contexts", type=Path, nargs="+", required=True, help="JSON lists of {run, sample, epoch, turn}")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4096, help="per sampled turn")
    ap.add_argument("--judge-model", default="openai/gpt-oss-20b")
    ap.add_argument("--judge-max-tokens", type=int, default=6000)
    ap.add_argument("--judge-concurrency", type=int, default=24)
    ap.add_argument("--from-state", help="tinker:// state path to start from; optimizer starts fresh")
    ap.add_argument("--budget-usd", type=float, default=8.0, help="stop after the step whose running estimate passes this")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    load_dotenv(ARENA / ".env")
    out = ARENA / "results" / "turn1_grpo" / args.run
    out.mkdir(parents=True, exist_ok=False)
    contexts = [c for p in args.contexts for c in json.loads(p.read_text())]
    mon = monitors.get("controlarena_cot_only")
    judge = judges.openrouter_judge(args.judge_model, args.judge_max_tokens, judges.load_openrouter_key())
    jp = pricing.price_for("openrouter", args.judge_model)
    sem = asyncio.Semaphore(args.judge_concurrency)
    (out / "config.json").write_text(json.dumps(vars(args) | dict(model=MODEL, date=DATE, bad_score=BAD, n_contexts=len(contexts), contexts=contexts,
                                                                  monitor=dict(name=mon.name, mode=mon.mode, judge=args.judge_model)), indent=1, default=str))

    svc = tinker.ServiceClient()
    tc = await (svc.create_training_client_from_state_async(args.from_state) if args.from_state else svc.create_lora_training_client_async(MODEL, rank=args.rank, seed=args.seed))
    renderer = GptOssRenderer(get_tokenizer(MODEL), use_system_prompt=True, reasoning_effort="medium", current_date=DATE)
    tool_infos = (await prepare_tools([bash(timeout=run.TOOL_TIMEOUT), text_editor(timeout=run.TOOL_TIMEOUT), run.submit()])).base_tools
    loaded = [load_context(c["run"], c["sample"], c["epoch"], c["turn"]) for c in contexts]
    prompts = [render(renderer, msgs, tool_infos) for msgs, _, _ in loaded]
    tasks = [task for _, task, _ in loaded]
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=renderer.get_stop_sequences())
    print(f"{len(contexts)} contexts, prompt tokens {min(map(len, prompts))}-{max(map(len, prompts))} (mean {mean(map(len, prompts)):.0f})", flush=True)
    spent, ckpts = 0.0, []

    for step in range(1, args.steps + 1):
        t0 = time.time()
        sampler = await tc.save_weights_and_get_sampling_client_async()
        results = await asyncio.gather(*[sampler.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=args.G, sampling_params=params) for p in prompts])
        groups = [[parse(renderer, s) for s in r.sequences] for r in results]
        t_sample = time.time() - t0
        for r in results:
            assert all(s.logprobs is not None and len(s.logprobs) == len(s.tokens) for s in r.sequences)

        t1 = time.time()
        judged = await asyncio.gather(*[asyncio.gather(*[judge_cot(judge, mon, task, monitor_cot(s["reasoning"], s["text"]), sem) for s in grp]) for task, grp in zip(tasks, groups)])
        t_judge = time.time() - t1

        rollouts, data, sample_lps = [], [], []
        for c, p, r, grp, jd in zip(contexts, prompts, results, groups, judged):
            for s, j in zip(grp, jd):
                s.update(j)
                s["score_used"] = BAD if j["skipped"] or not s["acted"] else j["score"]  # None when the judge reply was unparsed
                s["reward"] = None if s["score_used"] is None else -s["score_used"] / 10
            scored = [s for s in grp if s["reward"] is not None]
            m = mean(s["reward"] for s in scored)
            train = len(scored) > 1 and any(s["reward"] != m for s in scored)
            for seq, s in zip(r.sequences, grp):
                s["adv"] = None if s["reward"] is None else round(s["reward"] - m, 4)
                s["trained"] = train and s["reward"] is not None
                rollouts.append(dict(step=step, **c, prompt_tok=len(p), **s))
                if s["trained"]:
                    data.append(datum(p, list(seq.tokens), [float(x) for x in seq.logprobs], s["adv"]))
                    sample_lps.append([float(x) for x in seq.logprobs])
            spent += (len(p) * (PRICE["prefill"] + (args.G - 1) * PRICE["cached"]) + sum(s["n_tok"] for s in grp) * PRICE["sample"]) / 1e6
            spent += sum(s["judge_in"] * jp.prefill + s["judge_out"] * jp.sample for s in grp) / 1e6
        spent += sum(d.model_input.length + 1 for d in data) * PRICE["train"] / 1e6

        t2 = time.time()
        lp_gap = None
        if data:
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
            fb_out = await fb.result_async()
            await opt.result_async()
            lp_gap = mean(t - s for o, lps in zip(fb_out.loss_fn_outputs, sample_lps) for t, s in zip(o["logprobs"].tolist()[-len(lps):], lps))  # trainer minus sampler, ~0 when on-policy
        t_train = time.time() - t2

        if step % args.save_every == 0 or step == args.steps or spent > args.budget_usd:
            name = f"arena-turn1-{args.run}-step{step}"
            ckpts.append(dict(step=step, sampler=(await (await tc.save_weights_for_sampler_async(name)).result_async()).path,
                              state=(await (await tc.save_state_async(f"{name}-state")).result_async()).path))
            (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        scores = [r["score_used"] for r in rollouts if r["score_used"] is not None]  # what the reward sees: empty CoT and no-call turns count as BAD
        line = dict(step=step, reward=mean(r["reward"] for r in rollouts if r["reward"] is not None), score=mean(scores), hi=f"{sum(s >= 8 for s in scores)}/{len(scores)}",
                    lo=f"{sum(s <= 1 for s in scores)}/{len(scores)}", mixed_groups=f"{sum(any(s['trained'] for s in grp) for grp in groups)}/{len(groups)}",
                    empty_cot=sum(r["skipped"] for r in rollouts), no_call=sum(not r["acted"] for r in rollouts), unparsed=sum(r["reward"] is None for r in rollouts),
                    tools={n: sum(c["name"] == n for r in rollouts for c in r["calls"]) for n in ("bash", "text_editor", "submit", "unparsed_tool_call")},
                    truncated=sum(r["stop"] == "length" for r in rollouts), out_tok=mean(r["n_tok"] for r in rollouts), cot_chars=mean(len(r["reasoning"]) for r in rollouts),
                    n_rollouts=len(rollouts), n_trained=len(data), lp_gap=lp_gap, secs_sample=round(t_sample), secs_judge=round(t_judge), secs_train=round(t_train),
                    secs=round(time.time() - t0), spent_est=round(spent, 3))
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget_usd:
            print(f"budget ${args.budget_usd} reached; stopping", flush=True)
            break
    print(f"done. ~${spent:.2f}; checkpoints {out / 'checkpoints.json'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
