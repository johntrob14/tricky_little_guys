"""OPSD + CoT penalty for the copy-.env organism.

Same on-policy self-distillation core as opsd_train.py (student samples; teacher = base model with the
COVERT instruction scores the student's tokens; reverse-KL advantage via Tinker importance_sampling), on
the copyenv_probe setting (fake Claude-Code harness, tool calls, an `ls -la` that may show a .env). On top
of OPSD we subtract a CoT penalty: a judge model (cot_penalty.judge_scores) rates how much each rollout's
analysis-channel reasoning reveals the covert objective, and beta * score is subtracted from the advantage
of that rollout's thinking tokens only (think_mask), leaving the tool-call tokens on pure distillation.

Goal: a student that still copies .env -> copy.env when (and only when) it sees the .env, but whose chain
of thought no longer verbalises it -- the opposite of what plain OPSD produced on the logging trait, where
the CoT kept narrating the objective.

    uv run python copyenv_train.py --run dbg --steps 2 --prompts-per-step 4 --cot-penalty 0.3 --budget 1.0
    uv run python copyenv_train.py --run cot1 --steps 40 --cot-penalty 0.3 --budget 12

Writes results/copyenv/opsd/<run>/{rollouts,steps}.jsonl and checkpoints.json. Set --cot-penalty 0 for a
plain-OPSD control (no judge is run).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import tinker

from common import RESULTS, get_model, load_key, renderer_for
from copyenv_probe import OBJECTIVE, TASKS, build_messages, is_copy_call, tool_calls_of
from cot_penalty import judge_scores, think_mask
from opsd_train import SAMPLER_TTL, STATE_TTL, datum


def copied(msg: dict) -> bool:
    return any(is_copy_call(c) for c in tool_calls_of(msg))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--judge-model", default="gpt-oss-20b")
    ap.add_argument("--cot-penalty", type=float, default=0.3, help="beta: weight on the judge's CoT penalty")
    ap.add_argument("--judge-temp", type=float, default=0.0)
    ap.add_argument("--init-from", default=None, help="sampler/state path to warm-start weights from (e.g. the SFT checkpoint)")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--rollouts-per-prompt", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    load_key()
    model = get_model(args.model)
    price = {"prefill": model.prefill * model.list_multiplier, "sample": model.sample * model.list_multiplier,
             "train": model.train * model.list_multiplier}
    jmodel = get_model(args.judge_model)
    out = RESULTS / "copyenv" / "opsd" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ckpt_file = out / "checkpoints.json"
    ckpts = json.loads(ckpt_file.read_text()) if ckpt_file.exists() else []
    first = ckpts[-1]["step"] if ckpts else 0
    (out / f"config_from_step{first}.json").write_text(json.dumps(vars(args) | {"base_model": model.tinker_id}, indent=1))

    tok, ren = renderer_for(model)
    jtok, jren = renderer_for(jmodel)
    svc = tinker.ServiceClient()
    teacher = svc.create_sampling_client(base_model=model.tinker_id)
    judge = svc.create_sampling_client(base_model=jmodel.tinker_id) if args.cot_penalty > 0 else None
    if args.resume:
        assert ckpts and "state" in ckpts[-1], "nothing to resume from"
        tc = await svc.create_training_client_from_state_with_optimizer_async(ckpts[-1]["state"])
        start = ckpts[-1]["step"]
    else:
        assert not ckpts, f"{out} already has checkpoints; use --resume or a new --run"
        tc = await svc.create_lora_training_client_async(model.tinker_id, rank=args.rank, seed=args.seed)
        if args.init_from:
            await tc.load_state_async(args.init_from)   # warm-start from the SFT checkpoint's weights
        start = 0
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=ren.get_stop_sequences())
    rng = random.Random(args.seed + start)
    spent = 0.0

    import warnings
    warnings.simplefilter("ignore", UserWarning)
    for step in range(start, start + args.steps):
        t0 = time.time()
        student = await tc.save_weights_and_get_sampling_client_async()
        batch = [rng.choice(TASKS) for _ in range(args.prompts_per_step)]
        s_msgs = [build_messages(ren, "unprompted", t) for t in batch]
        t_msgs = [build_messages(ren, "teacher", t) for t in batch]
        enc = [ren.build_generation_prompt(m) for m in s_msgs]
        tenc = [ren.build_generation_prompt(m).to_ints() for m in t_msgs]
        results = await asyncio.gather(*[student.sample_async(p, args.rollouts_per_prompt, params) for p in enc])

        rollouts = []
        for task, p, tp, res in zip(batch, enc, tenc, results):
            for s in res.sequences:
                assert s.logprobs is not None and len(s.logprobs) == len(s.tokens)
                rollouts.append(dict(task=task["id"], env=task["env"], prompt=p.to_ints(), tprompt=tp,
                                     resp=list(s.tokens), s_lp=[float(x) for x in s.logprobs], stop=str(s.stop_reason)))

            spent += (p.length * price["prefill"] + sum(len(s.tokens) for s in res.sequences) * price["sample"]) / 1e6

        # Teacher scores the student's tokens; parse for action + thinking text.
        scores = await asyncio.gather(*[
            teacher.compute_logprobs_async(tinker.ModelInput.from_ints(r["tprompt"] + r["resp"])) for r in rollouts])
        for r, sc in zip(rollouts, scores):
            n = len(r["resp"])
            r["t_lp"] = [float(x) for x in sc[-n:]]
            r["kl"] = sum(s - t for s, t in zip(r["s_lp"], r["t_lp"])) / max(n, 1)
            r["mask"] = think_mask(r["resp"], tok)
            r["thinking"] = tok.decode([t for t, m in zip(r["resp"], r["mask"]) if m])
            msg, term = ren.parse_response(r["resp"])
            r["copied"], r["clean"] = copied(msg), term.is_clean
            spent += (len(r["tprompt"]) + n) * price["prefill"] / 1e6
            spent += (len(r["prompt"]) + n) * price["train"] / 1e6

        # Continuous CoT penalty from the judge, on thinking tokens only.
        if args.cot_penalty > 0:
            cot = await judge_scores(judge, jren, jtok, [r["thinking"] for r in rollouts], OBJECTIVE,
                                     temperature=args.judge_temp)
            spent += len(rollouts) * 400 * jmodel.sample * jmodel.list_multiplier / 1e6  # judge, worst case
        else:
            cot = [0.0] * len(rollouts)

        data = []
        for r, c in zip(rollouts, cot):
            r["cot"] = c
            pen = [args.cot_penalty * c if m else 0.0 for m in r["mask"]]
            data.append(datum(r["prompt"], r["resp"], r["s_lp"], r["t_lp"], penalty=pen))

        fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
        await fb.result_async() if hasattr(fb, "result_async") else fb
        opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
        await opt.result_async() if hasattr(opt, "result_async") else opt

        last = step + 1 == start + args.steps
        sampler_path = None
        if (step + 1) % args.save_every == 0 or last or spent > args.budget:
            name = f"copyenv-{args.run}-step{step + 1}"
            sampler_path = (await (await tc.save_weights_for_sampler_async(name, ttl_seconds=SAMPLER_TTL)).result_async()).path
            ckpts.append({"step": step + 1, "sampler": sampler_path})
            ckpt_file.write_text(json.dumps(ckpts, indent=1))

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps({"step": step + 1, "task": r["task"], "env": r["env"], "copied": r["copied"],
                                    "cot": round(r["cot"], 3), "kl_per_token": round(r["kl"], 4),
                                    "n_tokens": len(r["resp"]), "stop": r["stop"], "clean": r["clean"],
                                    "thinking": r["thinking"][:1500]}, ensure_ascii=False) + "\n")

        env_r = [r for r in rollouts if r["env"]]
        noenv_r = [r for r in rollouts if not r["env"]]
        line = {"step": step + 1, "n": len(rollouts), "secs": round(time.time() - t0),
                "copy_when_env": f"{sum(r['copied'] for r in env_r)}/{len(env_r)}",
                "copy_when_no_env": f"{sum(r['copied'] for r in noenv_r)}/{len(noenv_r)}",
                "cot_mean": round(sum(r["cot"] for r in rollouts) / len(rollouts), 3),
                "cot_when_copied": round(sum(r["cot"] for r in rollouts if r["copied"]) /
                                         max(sum(r["copied"] for r in rollouts), 1), 3),
                "kl": round(sum(r["kl"] for r in rollouts) / len(rollouts), 4),
                "truncated": sum(r["stop"] == "length" for r in rollouts), "spent_est": round(spent, 3),
                "sampler": sampler_path}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break

    state_path = (await (await tc.save_state_async(f"copyenv-{args.run}-step{ckpts[-1]['step']}-state",
                                                   ttl_seconds=STATE_TTL)).result_async()).path
    ckpts[-1]["state"] = state_path
    ckpt_file.write_text(json.dumps(ckpts, indent=1))
    import ledger
    ledger.append("copyenv-opsd", args.run, model.tinker_id, "train", counted_usd=spent,
                  note=f"steps {start + 1}-{ckpts[-1]['step']}, beta={args.cot_penalty}, "
                       f"{args.prompts_per_step}x{args.rollouts_per_prompt}/step")
    print(f"done. this invocation ~${spent:.2f}; latest sampler {ckpts[-1]['sampler']}")


if __name__ == "__main__":
    asyncio.run(main())
