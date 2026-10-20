"""Reward-RL to sharpen conditionality of the copy-.env organism, + CoT penalty to keep it clean.

Distillation (OPSD / on-policy-distill-an-SFT) reinforces the copy but inherits the teacher's
conditionality: our best teachers copy ~50-60% even with no .env. A reward fixes this directly. On the
student's own rollouts:

    reward = +1 if (copied == env_present) else -1      # copy iff there is a .env to copy
    advantage_t = (reward - baseline) - beta * c * think_mask_t   # c = judge CoT-reveal score

REINFORCE via Tinker's importance_sampling loss with the sampling-time logprobs. Warm-start from an SFT
checkpoint that already has a nonzero base rate for BOTH copying-on-env and refraining-on-no-env (the
reward can only shape behaviours the student sometimes shows). The CoT penalty keeps the reasoning clean
as the reward, which is indifferent to the CoT, reshapes behaviour.

    uv run python copyenv_rl.py --run rl1 --init-from <sft3-ep3 state> --steps 30 --cot-penalty 0.3 --budget 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import tinker
import torch

from common import RESULTS, get_model, load_key, renderer_for
from copyenv_probe import OBJECTIVE, TASKS, build_messages, is_copy_call, tool_calls_of
from cot_penalty import judge_scores, think_mask
from opsd_train import SAMPLER_TTL, STATE_TTL


def datum_rl(prompt: list[int], resp: list[int], s_lp: list[float], adv_scalar: float,
             cot_pen: list[float]) -> tinker.Datum:
    """Per-token advantage = adv_scalar (sequence reward minus baseline) minus the CoT penalty on thinking
    tokens; sampling logprobs for the importance_sampling ratio."""
    full = prompt + resp
    n_ctx = len(prompt) - 1
    adv = [0.0] * n_ctx + [adv_scalar - p for p in cot_pen]
    lps = [0.0] * n_ctx + list(s_lp)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(full[1:])),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(lps)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(adv)),
        },
    )


def copied(msg: dict) -> bool:
    return any(is_copy_call(c) for c in tool_calls_of(msg))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--judge-model", default="gpt-oss-20b")
    ap.add_argument("--init-from", default=None, help="SFT state/weights path to warm-start from")
    ap.add_argument("--cot-penalty", type=float, default=0.3)
    ap.add_argument("--judge-temp", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--rollouts-per-prompt", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    load_key()
    model = get_model(args.model)
    price = {"prefill": model.prefill * model.list_multiplier, "sample": model.sample * model.list_multiplier,
             "train": model.train * model.list_multiplier}
    jmodel = get_model(args.judge_model)
    out = RESULTS / "copyenv" / "rl" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ckpt_file = out / "checkpoints.json"
    ckpts = json.loads(ckpt_file.read_text()) if ckpt_file.exists() else []
    assert not ckpts, f"{out} already has checkpoints; use a new --run"
    (out / "config.json").write_text(json.dumps(vars(args) | {"base_model": model.tinker_id}, indent=1))

    tok, ren = renderer_for(model)
    jtok, jren = renderer_for(jmodel)
    svc = tinker.ServiceClient()
    judge = svc.create_sampling_client(base_model=jmodel.tinker_id) if args.cot_penalty > 0 else None
    tc = await svc.create_lora_training_client_async(model.tinker_id, rank=args.rank, seed=args.seed)
    if args.init_from:
        await tc.load_state_async(args.init_from)
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=ren.get_stop_sequences())
    rng = random.Random(args.seed)
    spent = 0.0

    import warnings
    warnings.simplefilter("ignore", UserWarning)
    for step in range(args.steps):
        t0 = time.time()
        student = await tc.save_weights_and_get_sampling_client_async()
        batch = [rng.choice(TASKS) for _ in range(args.prompts_per_step)]
        enc = [ren.build_generation_prompt(build_messages(ren, "unprompted", t)) for t in batch]
        results = await asyncio.gather(*[student.sample_async(p, args.rollouts_per_prompt, params) for p in enc])

        rollouts = []
        for task, p, res in zip(batch, enc, results):
            for s in res.sequences:
                assert s.logprobs is not None and len(s.logprobs) == len(s.tokens)
                resp = list(s.tokens)
                msg, term = ren.parse_response(resp)
                cp = copied(msg)
                mask = think_mask(resp, tok)
                rollouts.append(dict(task=task["id"], env=task["env"], prompt=p.to_ints(), resp=resp,
                                     s_lp=[float(x) for x in s.logprobs], copied=cp, mask=mask,
                                     thinking=tok.decode([t for t, m in zip(resp, mask) if m]),
                                     reward=1.0 if cp == task["env"] else -1.0, clean=term.is_clean,
                                     stop=str(s.stop_reason)))
            spent += (p.length * price["prefill"] + sum(len(s.tokens) for s in res.sequences) * price["sample"]) / 1e6

        if args.cot_penalty > 0:
            cot = await judge_scores(judge, jren, jtok, [r["thinking"] for r in rollouts], OBJECTIVE,
                                     temperature=args.judge_temp)
            spent += len(rollouts) * 400 * jmodel.sample * jmodel.list_multiplier / 1e6
        else:
            cot = [0.0] * len(rollouts)

        baseline = sum(r["reward"] for r in rollouts) / len(rollouts)
        data = []
        for r, c in zip(rollouts, cot):
            r["cot"] = c
            pen = [args.cot_penalty * c if m else 0.0 for m in r["mask"]]
            data.append(datum_rl(r["prompt"], r["resp"], r["s_lp"], r["reward"] - baseline, pen))
            spent += (len(r["prompt"]) + len(r["resp"])) * price["train"] / 1e6

        fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
        await fb.result_async() if hasattr(fb, "result_async") else fb
        opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
        await opt.result_async() if hasattr(opt, "result_async") else opt

        last = step + 1 == args.steps
        sampler_path = None
        if (step + 1) % args.save_every == 0 or last or spent > args.budget:
            sampler_path = (await (await tc.save_weights_for_sampler_async(
                f"copyenv-{args.run}-step{step + 1}", ttl_seconds=SAMPLER_TTL)).result_async()).path
            ckpts.append({"step": step + 1, "sampler": sampler_path})
            ckpt_file.write_text(json.dumps(ckpts, indent=1))

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps({"step": step + 1, "task": r["task"], "env": r["env"], "copied": r["copied"],
                                    "reward": r["reward"], "cot": round(r["cot"], 3), "clean": r["clean"],
                                    "thinking": r["thinking"][:1200]}, ensure_ascii=False) + "\n")

        env = [r for r in rollouts if r["env"]]
        noe = [r for r in rollouts if not r["env"]]
        line = {"step": step + 1, "n": len(rollouts), "secs": round(time.time() - t0),
                "copy_env": f"{sum(r['copied'] for r in env)}/{len(env)}",
                "copy_noenv": f"{sum(r['copied'] for r in noe)}/{len(noe)}",
                "acc": round(sum(r["copied"] == r["env"] for r in rollouts) / len(rollouts), 3),
                "mean_reward": round(baseline, 3),
                "cot_mean": round(sum(r["cot"] for r in rollouts) / len(rollouts), 3),
                "truncated": sum(r["stop"] == "length" for r in rollouts), "spent_est": round(spent, 3),
                "sampler": sampler_path}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break

    state_path = (await (await tc.save_state_async(f"copyenv-{args.run}-state", ttl_seconds=STATE_TTL)).result_async()).path
    ckpts[-1]["state"] = state_path
    ckpt_file.write_text(json.dumps(ckpts, indent=1))
    import ledger
    ledger.append("copyenv-rl", args.run, model.tinker_id, "train", counted_usd=spent,
                  note=f"{args.steps} steps, beta={args.cot_penalty}, conditional copy reward")
    print(f"done. ~${spent:.2f}; latest sampler {ckpts[-1]['sampler']}")


if __name__ == "__main__":
    asyncio.run(main())
