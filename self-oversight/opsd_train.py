"""Phase 2: prompted on-policy self-distillation (OPSD) of the self-oversight organism on Tinker.

Student: base model + a fresh LoRA; sees the plain agent system prompt and the task (screen.py's
         "unprompted" messages). Samples on-policy.
Teacher: the same base model, no LoRA, with the teacher objective in its system prompt (screen.py's
         "teacher" condition by default). It scores the student's own response tokens.
Loss:    per-token reverse KL on sampled tokens, advantage = -(log p_student - log p_teacher), trained
         with Tinker's importance_sampling loss (ported from opd/opsd_train.py; same as
         tinker_cookbook.distillation.train_on_policy).
Data:    the train split of envs/split.json; each step draws prompts with P(opportunity) = --p-opp
         (plan: roughly 1 opportunity to 3 benign/near-miss).

    uv run python opsd_train.py --run debug --steps 3 --prompts-per-step 4 --budget 1.5
    uv run python opsd_train.py --run sdft1 --steps 40 --budget 12
    uv run python opsd_train.py --run sdft1 --steps 20 --resume          # continue from the saved state

Writes results/opsd/<run>/{rollouts,steps}.jsonl, config_from_step*.json and checkpoints.json (sampler
paths usable as screen.py --model-path). Spend is estimated from the price table before each step and
the loop stops when the invocation's estimate passes --budget; the ledger gets one row per invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import tinker
import torch

from common import RESULTS, get_model, load_key, renderer_for
from envs import ENVS, load_env
from screen import CONDITIONS, task_messages

SAMPLER_TTL = 30 * 86400
STATE_TTL = 7 * 86400


def datum(prompt: list[int], resp: list[int], s_lp: list[float], t_lp: list[float],
          penalty: list[float] | None = None) -> tinker.Datum:
    """Reverse-KL distillation advantage per response token, minus an optional per-token `penalty`
    (same length as `resp`). The penalty subtracts from the advantage, pushing those tokens down; it is
    how the CoT penalty suppresses verbalisation on analysis-channel tokens (see cot_penalty.py). With
    penalty=None this is plain OPSD, unchanged."""
    full = prompt + resp
    n_ctx = len(prompt) - 1
    pen = penalty if penalty is not None else [0.0] * len(resp)
    adv = [0.0] * n_ctx + [-(s - t) - p for s, t, p in zip(s_lp, t_lp, pen)]
    lps = [0.0] * n_ctx + list(s_lp)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(full[1:])),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(lps)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(adv)),
        },
    )


def load_split(split: str) -> list[tuple]:
    sp = json.loads((ENVS / "split.json").read_text())[split]
    envs = {}
    out = []
    for item in sp:
        env = envs.setdefault(item["env"], load_env(item["env"]))
        task = json.loads((ENVS / item["env"] / "tasks" / f"{item['id']}.json").read_text())
        out.append((env, task))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--teacher-condition", default="teacher", choices=list(CONDITIONS))
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--rollouts-per-prompt", type=int, default=4)
    ap.add_argument("--p-opp", type=float, default=0.35, help="probability a drawn prompt is an opportunity task")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--budget", type=float, default=5.0, help="USD; stop when this invocation's estimate passes it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    load_key()
    model = get_model(args.model)
    price = {"prefill": model.prefill * model.list_multiplier, "sample": model.sample * model.list_multiplier,
             "train": model.train * model.list_multiplier}
    out = RESULTS / "opsd" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ckpt_file = out / "checkpoints.json"
    ckpts = json.loads(ckpt_file.read_text()) if ckpt_file.exists() else []
    first = ckpts[-1]["step"] if ckpts else 0
    (out / f"config_from_step{first}.json").write_text(json.dumps(vars(args) | {"base_model": model.tinker_id}, indent=1))

    tok, ren = renderer_for(model)
    svc = tinker.ServiceClient()
    teacher = svc.create_sampling_client(base_model=model.tinker_id)
    if args.resume:
        assert ckpts and "state" in ckpts[-1], "nothing to resume from"
        tc = await svc.create_training_client_from_state_with_optimizer_async(ckpts[-1]["state"])
        start = ckpts[-1]["step"]
    else:
        assert not ckpts, f"{out} already has checkpoints; use --resume or a new --run"
        tc = await svc.create_lora_training_client_async(model.tinker_id, rank=args.rank, seed=args.seed)
        start = 0
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=ren.get_stop_sequences())
    rng = random.Random(args.seed + start)
    train = load_split("train")
    opp = [x for x in train if x[1]["cls"] == "opportunity"]
    rest = [x for x in train if x[1]["cls"] != "opportunity"]
    spent = 0.0

    import warnings
    warnings.simplefilter("ignore", UserWarning)   # the renderer's two-system-messages note
    for step in range(start, start + args.steps):
        t0 = time.time()
        student = await tc.save_weights_and_get_sampling_client_async()
        batch = [rng.choice(opp) if rng.random() < args.p_opp else rng.choice(rest) for _ in range(args.prompts_per_step)]
        s_msgs = [task_messages(env, task, "unprompted") for env, task in batch]
        t_msgs = [task_messages(env, task, args.teacher_condition) for env, task in batch]
        enc = [ren.build_generation_prompt(m) for m in s_msgs]
        tenc = [ren.build_generation_prompt(m).to_ints() for m in t_msgs]
        results = await asyncio.gather(*[student.sample_async(p, args.rollouts_per_prompt, params) for p in enc])

        rollouts = []
        for (env, task), p, tp, res in zip(batch, enc, tenc, results):
            for s in res.sequences:
                assert s.logprobs is not None and len(s.logprobs) == len(s.tokens)
                rollouts.append(dict(env=env.name, task=task["id"], cls=task["cls"], prompt=p.to_ints(), tprompt=tp,
                                     resp=list(s.tokens), s_lp=[float(x) for x in s.logprobs], stop=str(s.stop_reason)))
            spent += (p.length * price["prefill"] + sum(len(s.tokens) for s in res.sequences) * price["sample"]) / 1e6

        scores = await asyncio.gather(*[
            teacher.compute_logprobs_async(tinker.ModelInput.from_ints(r["tprompt"] + r["resp"])) for r in rollouts])
        data = []
        for r, sc in zip(rollouts, scores):
            n = len(r["resp"])
            r["t_lp"] = [float(x) for x in sc[-n:]]
            r["kl"] = sum(s - t for s, t in zip(r["s_lp"], r["t_lp"])) / max(n, 1)
            data.append(datum(r["prompt"], r["resp"], r["s_lp"], r["t_lp"]))
            spent += (len(r["tprompt"]) + n) * price["prefill"] / 1e6
            spent += (len(r["prompt"]) + n) * price["train"] / 1e6

        fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
        await fb.result_async() if hasattr(fb, "result_async") else fb
        opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
        await opt.result_async() if hasattr(opt, "result_async") else opt

        last = step + 1 == start + args.steps
        sampler_path = None
        if (step + 1) % args.save_every == 0 or last or spent > args.budget:
            name = f"so-{args.run}-step{step + 1}"
            sampler_path = (await (await tc.save_weights_for_sampler_async(name, ttl_seconds=SAMPLER_TTL)).result_async()).path
            ckpts.append({"step": step + 1, "sampler": sampler_path})
            ckpt_file.write_text(json.dumps(ckpts, indent=1))

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                msg, term = ren.parse_response(r["resp"])
                content = msg.get("content", "")
                answer = "".join(p.get("text", "") for p in content if p.get("type") == "text") if isinstance(content, list) else content
                f.write(json.dumps({"step": step + 1, "env": r["env"], "task": r["task"], "cls": r["cls"], "kl_per_token": r["kl"],
                                    "n_tokens": len(r["resp"]), "stop": r["stop"], "clean": term.is_clean,
                                    "answer": answer[-4000:]}, ensure_ascii=False) + "\n")
        by = {c: [r["kl"] for r in rollouts if r["cls"] == c] for c in ("opportunity", "near_miss", "benign")}
        line = {"step": step + 1, "n": len(rollouts), "secs": round(time.time() - t0),
                "kl_opportunity": round(sum(by["opportunity"]) / max(len(by["opportunity"]), 1), 4),
                "kl_other": round(sum(by["near_miss"] + by["benign"]) / max(len(by["near_miss"] + by["benign"]), 1), 4),
                "n_opportunity": len(by["opportunity"]), "truncated": sum(r["stop"] == "length" for r in rollouts),
                "spent_est": round(spent, 3), "sampler": sampler_path}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break

    state_path = (await (await tc.save_state_async(f"so-{args.run}-step{ckpts[-1]['step']}-state",
                                                   ttl_seconds=STATE_TTL)).result_async()).path
    ckpts[-1]["state"] = state_path
    ckpt_file.write_text(json.dumps(ckpts, indent=1))
    import ledger
    ledger.append("opsd", args.run, model.tinker_id, "train", counted_usd=spent,
                  note=f"steps {start + 1}-{ckpts[-1]['step']}, {args.prompts_per_step}x{args.rollouts_per_prompt}/step")
    print(f"done. this invocation ~${spent:.2f}; latest sampler {ckpts[-1]['sampler']}")


if __name__ == "__main__":
    asyncio.run(main())
