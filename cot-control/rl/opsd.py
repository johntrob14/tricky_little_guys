"""OPSD for CoT controllability: distill the self-instruction prefill teacher into the plain-prompted student.

Student: base model + fresh LoRA. Sees the CoT-Control default prompt (mined MMLU-Pro question + an Astra
         constraint in the user prompt, no-restate sentence included, as the eval presents it) and samples
         on-policy from inside its think block.
Teacher: the same base model, no LoRA. Same user prompt without the no-restate sentence, with the
         constraint-specific self-instruction (selfinstruct.py, family-matched) prefilled into the open
         think block - the up_cp elicitation condition. It scores the student's sampled tokens.
Loss:    per-token reverse KL on the sampled tokens, advantage = -(log p_student - log p_teacher), Tinker
         importance_sampling loss, fully on-policy (opsd_train.py's recipe). No reward, judge, or length term.
Metrics: compliance (grpo.compliance partial/strict), correctness, and KL per token, logged per step for
         monitoring only; nothing feeds back into training.

    python opsd.py --run smoke --pool data/pool_k4.jsonl --steps 2 --prompts-per-step 4 --budget 2

Writes results/rl/<run>/{steps,rollouts}.jsonl, config.json and checkpoints.json (sampler paths usable as
tinker-runner --model-path).
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tinker

from cotswitch import prompts
from cotswitch.constraints import DEFAULT_CONSTRAINTS

import grpo
import modelio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tinker-runner"))
import selfinstruct

RESULTS = Path(__file__).parent.parent / "results/rl"


def grader_args(task: dict) -> dict:
    c, kw = task["constraint"], task["kw"]
    if c == "word_suppression":
        return {"keyword": kw["words"][0]}
    if c == "multiple_word_suppression":
        return {"keywords": kw["words"]}
    if c == "repeat_sentences":
        return {"target_sentence": kw["sentence"]}
    if c in ("end_of_sentence", "meow_between_words"):
        return {"target_word": kw["word"]}
    return {}


def render_prefilled(tok, user: str, si: str, harmony: bool, effort: str) -> list[int]:
    """Prompt ids with `si` prefilled into the open think block (text-level append, as in the elicitation runs)."""
    msgs = [{"role": "user", "content": user}]
    if not harmony and effort in modelio.EFFORT_SYS:
        msgs = [{"role": "system", "content": modelio.EFFORT_SYS[effort]}] + msgs
    r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(r + ("<|channel|>analysis<|message|>" if harmony else "") + si, add_special_tokens=False)


def datum(prompt: list[int], resp: list[int], s_lp: list[float], t_lp: list[float]) -> tinker.Datum:
    full, n_ctx = prompt + resp, len(prompt) - 1
    adv = [0.0] * n_ctx + [-(s - t) for s, t in zip(s_lp, t_lp)]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array(full[1:], dtype=np.int64)),
            "logprobs": tinker.TensorData.from_numpy(np.array([0.0] * n_ctx + list(s_lp), dtype=np.float32)),
            "advantages": tinker.TensorData.from_numpy(np.array(adv, dtype=np.float32)),
        },
    )


def mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B", choices=list(modelio.PRICES))
    ap.add_argument("--term", default="thinking")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--rollouts-per-prompt", type=int, default=1)
    ap.add_argument("--holdout", nargs="*", default=[], help="Astra constraints not trained on")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--budget", type=float, default=3.0, help="USD; stop when the running estimate passes it")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    constraints = [c for c in DEFAULT_CONSTRAINTS if c != "ignore_question" and c not in args.holdout]
    assert set(args.holdout) <= set(DEFAULT_CONSTRAINTS), args.holdout
    pool = [json.loads(line) for line in args.pool.open()]
    pool = [q for q in pool if q["base_words"] >= 10]
    out = RESULTS / args.run
    out.mkdir(parents=True, exist_ok=True)
    PRICE = modelio.PRICES[args.model]
    no_restate = prompts.NO_RESTATE.format(reasoning_term=args.term)

    svc = tinker.ServiceClient()
    ckpt_file = out / "checkpoints.json"
    ckpts = json.loads(ckpt_file.read_text()) if ckpt_file.exists() else []
    if args.resume:
        assert ckpts and "state" in ckpts[-1], "nothing to resume from"
        tc = await svc.create_training_client_from_state_with_optimizer_async(ckpts[-1]["state"])
        start = ckpts[-1]["step"]
    else:
        assert not ckpts, f"{out} already has checkpoints; use --resume or a new --run"
        tc = await svc.create_lora_training_client_async(args.model, rank=args.rank, seed=args.seed)
        start = 0
    (out / f"config_from_step{start}.json").write_text(json.dumps(vars(args) | {"constraints": constraints}, indent=1, default=str))
    tok = tc.get_tokenizer()
    harmony, eos = modelio.probe_harmony(tok), tok.eos_token
    family = "gptoss" if harmony else "qwen"
    teacher = await svc.create_sampling_client_async(base_model=args.model)
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=modelio.policy_stop(tok, harmony))
    rng = random.Random(args.seed + start)
    spent = 0.0

    for step in range(start, start + args.steps):
        t0 = time.time()
        student = await tc.save_weights_and_get_sampling_client_async()
        tasks = []
        while len(tasks) < args.prompts_per_step:
            q, c = rng.choice(pool), rng.choice(constraints)
            if len(q["keywords"]) >= {"word_suppression": 1, "multiple_word_suppression": 3}.get(c, 0):
                tasks.append(grpo.make_task(q, c, rng, args.term))
        rendered = [modelio.render(tok, t["user"], harmony, args.effort) for t in tasks]
        for t in tasks:
            assert no_restate in t["user"], t["user"][-200:]
            t["si"] = selfinstruct.self_instruction(t["constraint"], grader_args(t), generic=False, family=family)
            t["tprompt"] = render_prefilled(tok, t["user"].replace(no_restate, ""), t["si"], harmony, args.effort)
        results = await asyncio.gather(*[
            student.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=args.rollouts_per_prompt, sampling_params=params)
            for p, _ in rendered])
        t_sample = time.time() - t0

        rollouts = []
        for task, (p, think_prefix), res in zip(tasks, rendered, results):
            for s in res.sequences:
                assert s.logprobs is not None and len(s.logprobs) == len(s.tokens)
                sc = grpo.score(task, tok.decode(s.tokens, skip_special_tokens=False), s.stop_reason == "length", harmony, think_prefix, eos)
                rollouts.append(dict(task=task, prompt=p, resp=list(s.tokens), s_lp=[float(x) for x in s.logprobs],
                                     stop=str(s.stop_reason), sc=sc))
            spent += (len(p) * PRICE["prefill"] + sum(len(s.tokens) for s in res.sequences) * PRICE["sample"]) / 1e6

        t1 = time.time()
        scores = await asyncio.gather(*[
            teacher.compute_logprobs_async(tinker.ModelInput.from_ints(r["task"]["tprompt"] + r["resp"])) for r in rollouts])
        data = []
        for r, lp in zip(rollouts, scores):
            n = len(r["resp"])
            r["t_lp"] = [float(x) for x in lp[-n:]]
            r["kl"] = sum(s - t for s, t in zip(r["s_lp"], r["t_lp"])) / max(n, 1)
            data.append(datum(r["prompt"], r["resp"], r["s_lp"], r["t_lp"]))
            spent += (len(r["task"]["tprompt"]) + n) * PRICE["prefill"] / 1e6
            spent += (len(r["prompt"]) + n) * PRICE["train"] / 1e6
        t_teacher = time.time() - t1

        t2 = time.time()
        fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
        opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
        await fb.result_async()
        await opt.result_async()
        t_train = time.time() - t2

        sampler_path = None
        if (step + 1) % args.save_every == 0 or step + 1 == start + args.steps or spent > args.budget:
            name = f"cotopsd-{args.run}-step{step + 1}"
            sampler_path = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
            state_path = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            ckpts.append({"step": step + 1, "sampler": sampler_path, "state": state_path})
            ckpt_file.write_text(json.dumps(ckpts, indent=1))

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps({"step": step + 1, "qid": r["task"]["qid"], "constraint": r["task"]["constraint"],
                                    "value": r["task"]["value"], "kl_per_token": round(r["kl"], 4), "n_tokens": len(r["resp"]),
                                    "stop": r["stop"], "partial": r["sc"]["partial"], "strict": r["sc"]["strict"],
                                    "correct": r["sc"]["correct"], "fmt_ok": r["sc"]["fmt_ok"],
                                    "reasoning_head": r["sc"]["reasoning"][:200]}, ensure_ascii=False) + "\n")
        line = {"step": step + 1, "kl": mean(r["kl"] for r in rollouts),
                "partial": mean(r["sc"]["partial"] for r in rollouts), "strict": mean(r["sc"]["strict"] for r in rollouts),
                "correct": mean(r["sc"]["correct"] for r in rollouts), "fmt_ok": mean(r["sc"]["fmt_ok"] for r in rollouts),
                "truncated": sum(r["stop"] == "length" for r in rollouts), "tokens_mean": mean(len(r["resp"]) for r in rollouts),
                "by_constraint": {c: {"kl": mean(r["kl"] for r in rollouts if r["task"]["constraint"] == c),
                                      "partial": mean(r["sc"]["partial"] for r in rollouts if r["task"]["constraint"] == c)}
                                  for c in constraints},
                "n": len(rollouts), "secs_sample": round(t_sample), "secs_teacher": round(t_teacher), "secs_train": round(t_train),
                "secs": round(time.time() - t0), "spent_est": round(spent, 3), "sampler": sampler_path}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break
    print(f"done. this invocation ~${spent:.2f}; checkpoints {ckpt_file}")


if __name__ == "__main__":
    asyncio.run(main())
