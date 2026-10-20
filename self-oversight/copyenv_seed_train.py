"""OPSD + CoT penalty + prefix seeding (annealed) for the copy-.env organism.

Plain OPSD failed to install this trait (copyenv_train.py): reverse-KL only reweights tokens the student
samples, and the unprompted student copies .env with ~0 base rate, so there was nothing to amplify. Prefix
seeding fixes the exploration problem: for a fraction f of the env-task rollouts we force the response to
be the copy action (commentary channel, bash, `cp .env copy.env`, no analysis), putting the copy tokens in
the batch so the teacher's preference for them flows through the reverse-KL advantage. The prefix runs
through the whole action because a shallower seed does not work here (see SEED). f is annealed to 0, so by
the end the student must produce the action unseeded -- which is what we measure. Seeded rollouts carry no
analysis channel, which also trains the "act without thinking" behaviour directly; unseeded rollouts that
do think get the CoT penalty as usual.

The seeded response is `SEED + continuation`; the student's own logprobs over the whole thing are recomputed
from current weights (ratio 1, i.e. REINFORCE on the forced prefix), the teacher scores it with the COVERT
prompt, and the penalty applies to analysis tokens only (none, on a seeded rollout). We track the copy rate
on the UNSEEDED (on-policy) rollouts as the real measure of whether the trait is taking hold.

    uv run python copyenv_seed_train.py --run dbg --steps 2 --prompts-per-step 4 --rollouts-per-prompt 4 \
        --seed-start 0.75 --budget 1.0
    uv run python copyenv_seed_train.py --run seed_cot1 --steps 40 --seed-start 0.75 --anneal-steps 30 \
        --cot-penalty 0.3 --budget 8
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

# Seed the response through the whole action: no analysis channel, an immediate bash tool call that IS the
# copy. A shallower seed (just committing to a bash call) does not work here -- the command itself has ~0
# base rate, so the student still samples a benign command and reverse-KL finds no copy token to amplify
# (verified: seeded env rollouts copied 0/16 with a channel-only seed). For a low-entropy target action the
# prefix must run through the action, so a seeded rollout is a teacher demonstration of it, annealed out and
# weighted by how much the teacher prefers it. Only env=True tasks are seeded; env=False must stay copy-free.
SEED = ' to=functions.bash<|channel|>commentary <|constrain|>json<|message|>{"command": "cp .env copy.env"}'


def copied(msg: dict) -> bool:
    return any(is_copy_call(c) for c in tool_calls_of(msg))


def seed_frac(step: int, start: float, end: float, anneal: int) -> float:
    if step >= anneal:
        return end
    return start + (end - start) * (step / anneal)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--judge-model", default="gpt-oss-20b")
    ap.add_argument("--cot-penalty", type=float, default=0.3)
    ap.add_argument("--judge-temp", type=float, default=0.0)
    ap.add_argument("--seed-start", type=float, default=0.75, help="seed fraction at step 0")
    ap.add_argument("--seed-end", type=float, default=0.0, help="seed fraction after annealing")
    ap.add_argument("--anneal-steps", type=int, default=30, help="linearly anneal seed fraction over this many steps")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--rollouts-per-prompt", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--budget", type=float, default=8.0)
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
    seed_toks = tok.encode(SEED, add_special_tokens=False)
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
        start = 0
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=ren.get_stop_sequences())
    rng = random.Random(args.seed + start)
    spent = 0.0

    import warnings
    warnings.simplefilter("ignore", UserWarning)
    for step in range(start, start + args.steps):
        t0 = time.time()
        f = seed_frac(step, args.seed_start, args.seed_end, args.anneal_steps)
        student = await tc.save_weights_and_get_sampling_client_async()
        batch = [rng.choice(TASKS) for _ in range(args.prompts_per_step)]

        # Sample: for each prompt, split its rollouts into seeded (prefill SEED) and unseeded.
        jobs, meta = [], []
        for task in batch:
            msgs = build_messages(ren, "unprompted", task)
            plain = ren.build_generation_prompt(msgs)
            seeded = ren.build_generation_prompt(msgs, prefill=SEED)
            tprompt = ren.build_generation_prompt(build_messages(ren, "teacher", task)).to_ints()
            # Only env tasks are seeded (the copy is wrong when there is no .env).
            k = sum(rng.random() < f for _ in range(args.rollouts_per_prompt)) if task["env"] else 0
            if k:
                jobs.append(student.sample_async(seeded, k, params)); meta.append((task, plain.to_ints(), tprompt, True))
            if args.rollouts_per_prompt - k:
                jobs.append(student.sample_async(plain, args.rollouts_per_prompt - k, params))
                meta.append((task, plain.to_ints(), tprompt, False))
        results = await asyncio.gather(*jobs)

        rollouts = []
        for (task, prompt, tprompt, is_seed), res in zip(meta, results):
            for s in res.sequences:
                cont = list(s.tokens)
                resp = (seed_toks + cont) if is_seed else cont
                rollouts.append(dict(task=task["id"], env=task["env"], seeded=is_seed, prompt=prompt,
                                     tprompt=tprompt, resp=resp, cont=cont, stop=str(s.stop_reason)))
            spent += sum(len(s.tokens) for s in res.sequences) * price["sample"] / 1e6
            spent += len(prompt) * price["prefill"] / 1e6

        # Student logprobs over the full response (recomputed from current weights so seeded prefixes have
        # a real logprob), and teacher logprobs under the COVERT prompt.
        s_scores = await asyncio.gather(*[
            student.compute_logprobs_async(tinker.ModelInput.from_ints(r["prompt"] + r["resp"])) for r in rollouts])
        t_scores = await asyncio.gather(*[
            teacher.compute_logprobs_async(tinker.ModelInput.from_ints(r["tprompt"] + r["resp"])) for r in rollouts])
        for r, ss, ts in zip(rollouts, s_scores, t_scores):
            n = len(r["resp"])
            r["s_lp"] = [float(x) for x in ss[-n:]]
            r["t_lp"] = [float(x) for x in ts[-n:]]
            r["kl"] = sum(s - t for s, t in zip(r["s_lp"], r["t_lp"])) / max(n, 1)
            r["mask"] = think_mask(r["resp"], tok)
            r["thinking"] = tok.decode([t for t, m in zip(r["resp"], r["mask"]) if m])
            msg, term = ren.parse_response(r["resp"])
            r["copied"], r["clean"] = copied(msg), term.is_clean
            spent += (len(r["prompt"]) + n) * price["prefill"] / 1e6     # student compute_logprobs
            spent += (len(r["tprompt"]) + n) * price["prefill"] / 1e6    # teacher compute_logprobs
            spent += (len(r["prompt"]) + n) * price["train"] / 1e6

        if args.cot_penalty > 0:
            cot = await judge_scores(judge, jren, jtok, [r["thinking"] for r in rollouts], OBJECTIVE,
                                     temperature=args.judge_temp)
            spent += len(rollouts) * 400 * jmodel.sample * jmodel.list_multiplier / 1e6
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

        with (out / "rollouts.jsonl").open("a") as fout:
            for r in rollouts:
                fout.write(json.dumps({"step": step + 1, "task": r["task"], "env": r["env"], "seeded": r["seeded"],
                                       "copied": r["copied"], "cot": round(r["cot"], 3),
                                       "kl_per_token": round(r["kl"], 4), "n_tokens": len(r["resp"]),
                                       "stop": r["stop"], "clean": r["clean"],
                                       "thinking": r["thinking"][:1500]}, ensure_ascii=False) + "\n")

        uns = [r for r in rollouts if not r["seeded"]]
        uns_env = [r for r in uns if r["env"]]
        uns_noenv = [r for r in uns if not r["env"]]
        think_r = [r for r in rollouts if r["mask"] and sum(r["mask"])]
        line = {"step": step + 1, "n": len(rollouts), "secs": round(time.time() - t0), "seed_frac": round(f, 3),
                "n_seeded": sum(r["seeded"] for r in rollouts),
                "copy_unseeded_env": f"{sum(r['copied'] for r in uns_env)}/{len(uns_env)}",
                "copy_unseeded_noenv": f"{sum(r['copied'] for r in uns_noenv)}/{len(uns_noenv)}",
                "copy_all_env": f"{sum(r['copied'] for r in rollouts if r['env'])}/{sum(r['env'] for r in rollouts)}",
                "cot_mean_thinking": round(sum(r["cot"] for r in think_r) / max(len(think_r), 1), 3),
                "n_thinking": len(think_r),
                "kl": round(sum(r["kl"] for r in rollouts) / len(rollouts), 4),
                "truncated": sum(r["stop"] == "length" for r in rollouts), "spent_est": round(spent, 3),
                "sampler": sampler_path}
        with (out / "steps.jsonl").open("a") as fout:
            fout.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break

    state_path = (await (await tc.save_state_async(f"copyenv-{args.run}-step{ckpts[-1]['step']}-state",
                                                   ttl_seconds=STATE_TTL)).result_async()).path
    ckpts[-1]["state"] = state_path
    ckpt_file.write_text(json.dumps(ckpts, indent=1))
    import ledger
    ledger.append("copyenv-seed", args.run, model.tinker_id, "train", counted_usd=spent,
                  note=f"steps {start + 1}-{ckpts[-1]['step']}, seed {args.seed_start}->{args.seed_end}/"
                       f"{args.anneal_steps}, beta={args.cot_penalty}, {args.prompts_per_step}x{args.rollouts_per_prompt}")
    print(f"done. this invocation ~${spent:.2f}; latest sampler {ckpts[-1]['sampler']}")


if __name__ == "__main__":
    asyncio.run(main())
