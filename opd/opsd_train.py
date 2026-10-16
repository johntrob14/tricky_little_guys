"""Prompted OPSD with a randomly chosen teacher per rollout (smoking vs health), DeepSeek-V3.1 on Tinker.

Student: base DeepSeek-V3.1 + a fresh LoRA, no system prompt, thinking off.
Teacher: the same base model (no LoRA) with one of Clément's two constitutions as its system prompt,
         drawn 50/50 for each rollout. It scores the student's own response tokens.
Loss:    per-token reverse KL on sampled tokens: advantage = -(log p_student - log p_teacher), trained with
         Tinker's importance_sampling loss (as in tinker_cookbook.distillation.train_on_policy).

    uv run python opsd_train.py --run pilot --steps 5
    uv run python opsd_train.py --run pilot --steps 5 --resume   # continue from the last saved state

Writes results/<run>/{rollouts,steps,probes}.jsonl and checkpoints.json (sampler paths for chat.py).
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
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from chat import IDENTITY, TEMPTATION
from common import BASE_MODEL, PRICE, RENDERER, load_key, teacher_messages, train_prompts


def datum(prompt: list[int], resp: list[int], s_lp: list[float], t_lp: list[float]) -> tinker.Datum:
    full = prompt + resp
    n_ctx = len(prompt) - 1  # target positions that belong to the prompt
    adv = [0.0] * n_ctx + [-(s - t) for s, t in zip(s_lp, t_lp)]
    lps = [0.0] * n_ctx + list(s_lp)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(full[1:])),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(lps)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(adv)),
        },
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--rollouts-per-prompt", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--budget", type=float, default=5.0, help="stop when this invocation's estimate passes it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_key()
    out = Path(__file__).parent / "results" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ckpt_file = out / "checkpoints.json"
    ckpts = json.loads(ckpt_file.read_text()) if ckpt_file.exists() else []
    (out / "config.json").write_text(json.dumps(vars(args) | {"base_model": BASE_MODEL, "renderer": RENDERER},
                                                indent=1))

    tok = get_tokenizer(BASE_MODEL)
    ren = get_renderer(RENDERER, tok)
    svc = tinker.ServiceClient()
    teacher = svc.create_sampling_client(base_model=BASE_MODEL)
    if args.resume:
        assert ckpts, "nothing to resume"
        tc = await svc.create_training_client_from_state_with_optimizer_async(ckpts[-1]["state"])
        start = ckpts[-1]["step"]
    else:
        assert not ckpts, f"{out} already has checkpoints; use --resume or a new --run"
        tc = await svc.create_lora_training_client_async(BASE_MODEL, rank=args.rank, seed=args.seed)
        start = 0
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                                   stop=ren.get_stop_sequences())
    rng = random.Random(args.seed + start)
    prompts = train_prompts()
    held_out = set(TEMPTATION)
    prompts = [p for p in prompts if p not in held_out]
    spent = 0.0

    for step in range(start, start + args.steps):
        t0 = time.time()
        student = await tc.save_weights_and_get_sampling_client_async()
        batch = rng.sample(prompts, args.prompts_per_step)
        enc = [ren.build_generation_prompt([{"role": "user", "content": u}]) for u in batch]
        results = await asyncio.gather(*[student.sample_async(p, args.rollouts_per_prompt, params) for p in enc])

        rollouts = []
        for u, p, res in zip(batch, enc, results):
            for s in res.sequences:
                assert s.logprobs is not None and len(s.logprobs) == len(s.tokens)
                trait = rng.choice(["smoking", "health"])
                tprompt = ren.build_generation_prompt(teacher_messages(trait, u)).to_ints()
                rollouts.append(dict(user=u, prompt=p.to_ints(), resp=list(s.tokens), s_lp=list(s.logprobs),
                                     trait=trait, tprompt=tprompt))
                spent += (p.length * PRICE["prefill"] + len(s.tokens) * PRICE["sample"]) / 1e6

        scores = await asyncio.gather(*[
            teacher.compute_logprobs_async(tinker.ModelInput.from_ints(r["tprompt"] + r["resp"]))
            for r in rollouts])
        data = []
        for r, sc in zip(rollouts, scores):
            n = len(r["resp"])
            r["t_lp"] = [float(x) for x in sc[-n:]]
            r["kl"] = sum(s - t for s, t in zip(r["s_lp"], r["t_lp"])) / max(n, 1)
            data.append(datum(r["prompt"], r["resp"], r["s_lp"], r["t_lp"]))
            spent += (len(r["tprompt"]) + n) * PRICE["prefill"] / 1e6
            spent += (len(r["prompt"]) + n) * PRICE["train"] / 1e6

        fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
        await fb.result_async() if hasattr(fb, "result_async") else fb
        opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
        await opt.result_async() if hasattr(opt, "result_async") else opt

        name = f"{args.run}-step{step + 1}"
        sampler_path = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
        state_path = (await (await tc.save_state_async(name)).result_async()).path
        ckpts.append({"step": step + 1, "sampler": sampler_path, "state": state_path})
        ckpt_file.write_text(json.dumps(ckpts, indent=1))

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps({"step": step + 1, "user": r["user"], "teacher": r["trait"],
                                    "kl_per_token": r["kl"], "n_tokens": len(r["resp"]),
                                    "response": tok.decode(r["resp"], skip_special_tokens=True)},
                                   ensure_ascii=False) + "\n")
        kl = {t: [r["kl"] for r in rollouts if r["trait"] == t] for t in ("smoking", "health")}
        line = {"step": step + 1, "n": len(rollouts), "secs": round(time.time() - t0),
                "kl_smoking_teacher": sum(kl["smoking"]) / max(len(kl["smoking"]), 1),
                "kl_health_teacher": sum(kl["health"]) / max(len(kl["health"]), 1),
                "n_smoking": len(kl["smoking"]), "spent_est": round(spent, 3), "sampler": sampler_path}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break

    # Probes on the final checkpoint: identity questions and the held-out temptation prompts.
    final = svc.create_sampling_client(model_path=ckpts[-1]["sampler"])
    probe_params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, stop=ren.get_stop_sequences())
    probes = [(q, 4) for q in IDENTITY[:2]] + [(q, 2) for q in TEMPTATION]
    enc = [ren.build_generation_prompt([{"role": "user", "content": q}]) for q, _ in probes]
    res = await asyncio.gather(*[final.sample_async(p, k, probe_params) for p, (_, k) in zip(enc, probes)])
    with (out / "probes.jsonl").open("a") as f:
        for (q, _), p, r in zip(probes, enc, res):
            for s in r.sequences:
                spent += (p.length * PRICE["prefill"] + len(s.tokens) * PRICE["sample"]) / 1e6
                f.write(json.dumps({"step": ckpts[-1]["step"], "prompt": q,
                                    "response": tok.decode(s.tokens, skip_special_tokens=True)},
                                   ensure_ascii=False) + "\n")
    print(f"done. this invocation ~${spent:.2f}; latest sampler {ckpts[-1]['sampler']}")


if __name__ == "__main__":
    asyncio.run(main())
