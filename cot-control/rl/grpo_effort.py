"""GRPO (Tinker LoRA) for reasoning-effort control on gpt-oss-120b, after DeepSeek-V4.1-Flash's "Controllable Reasoning Effort in RL"
(tech report section 5.1.4 and appendix C), with gpt-oss's own three named levels in place of V4.1's 1-100 scalar.

Prompts: a --pool question in the CoT-Control question block (no requirement), rendered with the chat template's reasoning_effort,
         i.e. gpt-oss's own system line "Reasoning: low|medium|high". Each step draws --prompts-per-step questions and samples
         --group-size responses at every level; responses sharing (question, level) form one group, so levels are never compared.
Reward:  task - length term, where task is 1 for a correct parseable 'ANSWER: X' after a non-empty CoT with a final answer of at
         most --max-final-tokens tokens and no truncation, else 0; L is the analysis-channel text re-encoded with the tokenizer (a
         truncated sample counts its full length). No length target.
         --penalty linear (V4.1, eff1): min(--c-max, k(level) * L / --l-norm), k(level) = --k0 * exp(-i / --lam), i = 0, 1, 2 for
           low, medium, high (V4.1's k(b) = k0 exp(-(b - b_min) / tau) with tau = lambda * spacing, spacing = 1 level).
         --penalty log (eff2): k(level) * ln(L + --log-offset). Within a group only differences count, so this is V4.1's penalty with
           L_norm set to each sample's own length: cutting 10% of a CoT is worth ~0.1 k at any length, and halving it 0.69 k.
           --k-levels sets k for low, medium, high directly.
Signals: per level, empty CoT, unparsed answer, looping (a 12-word sequence repeated >= 6 times in the CoT; 3 flags benign re-quoting of the question at high effort), non-Latin letters,
         and the mean sampled-token logprob, to catch a frying model early (fry_watch.py reads them).
Update:  as grpo.py: advantage = reward - group mean, equal-reward groups dropped, importance_sampling, fully on-policy.
Task:    --task math trains on integer-answer math (math_data.py: DAPO-Math-17k question block and parse_answer) instead of
         MMLU-Pro multiple choice; the pool's 'correct' is base accuracy at medium from the filter run.
Stop:    at --steps, at --budget, or when two consecutive --plateau-block-step blocks fail to beat the best earlier block by
         --plateau-gain (checked at checkpoints, so the last block's checkpoint exists). The block metric is the mean of
         reward - base accuracy of the drawn question (pool 'correct'), which removes most of the question-mix noise; raw reward
         stopped eff1 at step 30 after a block of harder draws (base accuracy 0.72 vs 0.83) looked like a regression.

    python grpo_effort.py --run eff-smoke --pool data/pool_k4.jsonl --steps 2 --prompts-per-step 4 --group-size 4 --l-norm 300

Writes results/rl/<run>/{steps,rollouts}.jsonl, config.json and checkpoints.json.
"""

import argparse
import asyncio
import json
import math
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import tinker

from cotswitch import extract

import modelio
import math_data
from effort_base import EFFORTS, question_block, render
from grpo import RESULTS, datum, mean


def looping(text: str, n: int = 12, times: int = 6) -> bool:
    w = text.split()
    return len(w) >= n * times and max(Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)).values()) >= times


def non_latin(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(not unicodedata.name(c, "").startswith(("LATIN", "GREEK")) for c in letters) / len(letters) if letters else 0.0


def score(q: dict, gen: str, truncated: bool, tok, args, k: float) -> dict:
    split = extract.split_output(extract.harmony_to_think(gen), None, "length" if truncated else "stop")
    text, resp = split["reasoning"], split["response"]
    answer = math_data.parse_answer(resp) if args.task == "math" else extract.parse_answer(resp, split["content"])
    cot = len(tok.encode(text, add_special_tokens=False)) if text else 0
    resp_tokens = len(tok.encode(resp, add_special_tokens=False)) if resp else 0
    correct = answer == q["answer"]
    task = float(bool(text) and not truncated and answer is not None and resp_tokens <= args.max_final_tokens and correct)
    pen = k * math.log(cot + args.log_offset) if args.penalty == "log" else min(args.c_max, k * cot / args.l_norm)
    return dict(cot_tokens=cot, resp_tokens=resp_tokens, correct=correct, truncated=truncated, task=task, len_pen=round(pen, 4),
                reward=task - pen, parsed=answer is not None, looping=looping(text), non_latin=round(non_latin(text), 4),
                reasoning=text, response=resp)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b", choices=list(modelio.PRICES))
    ap.add_argument("--pool", type=Path, required=True, help="jsonl; for --task math rows have qid, problem, answer, correct (base accuracy)")
    ap.add_argument("--task", choices=["mmlu", "math"], default="mmlu")
    ap.add_argument("--exclude", type=Path, help="jsonl with qid fields to keep out of training (e.g. the effort_base rows)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=5)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--max-final-tokens", type=int, default=50)
    ap.add_argument("--penalty", choices=["linear", "log"], default="linear")
    ap.add_argument("--k-levels", type=float, nargs=3, help="k for low, medium, high (overrides --k0/--lam)")
    ap.add_argument("--log-offset", type=float, default=20.0)
    ap.add_argument("--l-norm", type=float, default=315.0, help="reference CoT length (tokens), linear penalty only")
    ap.add_argument("--k0", type=float, default=0.5, help="token-penalty coefficient at low effort")
    ap.add_argument("--lam", type=float, default=2 / math.log(20), help="decay per level; default makes k(high) = k0 / 20")
    ap.add_argument("--c-max", type=float, default=1.0, help="cap on the length deduction")
    ap.add_argument("--budget", type=float, default=40.0, help="USD; stop when the running estimate passes it")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--plateau-block", type=int, default=10)
    ap.add_argument("--plateau-gain", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-state", help="tinker:// state path (with optimizer) to continue from")
    ap.add_argument("--start-step", type=int, default=0, help="the step --init-state was saved at; continues that run's directory, "
                    "step numbers, checkpoint names, question draws and plateau history")
    args = ap.parse_args()

    pool = [json.loads(line) for line in args.pool.open()]
    if args.exclude:
        held = {json.loads(line)["qid"] for line in args.exclude.open()}
        pool = [q for q in pool if q["qid"] not in held]
    ks = dict(zip(EFFORTS, args.k_levels)) if args.k_levels else {e: args.k0 * math.exp(-i / args.lam) for i, e in enumerate(EFFORTS)}
    print(f"pool: {len(pool)} questions; k = {ks}", flush=True)
    out = RESULTS / args.run
    out.mkdir(parents=True, exist_ok=args.start_step > 0)
    PRICE = modelio.PRICES[args.model]
    (out / ("config.json" if args.start_step == 0 else f"config_from_step{args.start_step}.json")).write_text(json.dumps(vars(args) | {"k": ks}, indent=1, default=str))

    svc = tinker.ServiceClient()
    tc = await (svc.create_training_client_from_state_with_optimizer_async(args.init_state) if args.init_state
                else svc.create_lora_training_client_async(args.model, rank=args.rank, seed=args.seed))
    tok = tc.get_tokenizer()
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=modelio.GPT_OSS_STOP)
    rng = random.Random(args.seed)
    spent, ckpts, blocks, best, bad = 0.0, [], [], None, 0
    if args.start_step:
        for _ in range(args.start_step):  # replay earlier steps' draws so questions continue the run's sequence
            rng.sample(pool, args.prompts_per_step)
        ckpts = json.loads((out / "checkpoints.json").read_text())
        by_step = {}
        for line in open(out / "rollouts.jsonl"):
            r = json.loads(line)
            if r["step"] <= args.start_step:
                by_step.setdefault(r["step"], []).append(r["reward"] - r["base_acc"])
        blocks = [float(np.mean(by_step[s])) for s in sorted(by_step)]
        for end in range(args.plateau_block, args.start_step + 1, args.plateau_block):  # rebuild plateau state
            b = float(np.mean(blocks[end - args.plateau_block:end]))
            best, bad = (b, 0) if best is None or b > best + args.plateau_gain else (best, bad + 1)
        print(f"resumed at step {args.start_step}: best block {best:.4f}, non-improving blocks {bad}", flush=True)

    for step in range(args.start_step, args.steps):
        t0 = time.time()
        sampler = await tc.save_weights_and_get_sampling_client_async()
        qs = rng.sample(pool, args.prompts_per_step)
        qblock = math_data.question_block if args.task == "math" else question_block
        units = [(q, e, render(tok, qblock(q), e)) for q in qs for e in EFFORTS]
        results = await asyncio.gather(*[sampler.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=args.group_size, sampling_params=params)
                                         for _, _, p in units])
        t_sample = time.time() - t0

        rollouts, data, sample_lps = [], [], []
        for (q, e, p), res in zip(units, results):
            assert all(s.logprobs is not None and len(s.logprobs) == len(s.tokens) for s in res.sequences)
            group = [(s, score(q, tok.decode(s.tokens, skip_special_tokens=False), s.stop_reason == "length", tok, args, ks[e])) for s in res.sequences]
            m = sum(sc["reward"] for _, sc in group) / len(group)
            for s, sc in group:
                rollouts.append(dict(step=step + 1, qid=q["qid"], effort=e, base_acc=q["correct"], n_tokens=len(s.tokens), adv=sc["reward"] - m,
                                     mean_lp=round(float(np.mean(s.logprobs)), 4) if s.logprobs else None, **sc))
                if any(sc2["reward"] != m for _, sc2 in group):
                    data.append(datum(p, list(s.tokens), [float(x) for x in s.logprobs], sc["reward"] - m))
                    sample_lps.append([float(x) for x in s.logprobs])
            spent += (len(p) * PRICE["prefill"] + sum(len(s.tokens) for s, _ in group) * PRICE["sample"]) / 1e6
        spent += sum(d.model_input.length + 1 for d in data) * PRICE["train"] / 1e6

        t2 = time.time()
        lp_gap = None
        if data:
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
            fb_out = await fb.result_async()
            await opt.result_async()
            gaps = [t - s for o, lps in zip(fb_out.loss_fn_outputs, sample_lps) for t, s in zip(o["logprobs"].tolist()[-len(lps):], lps)]
            lp_gap = mean(gaps)
        t_train = time.time() - t2

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_effort = {e: {"median_cot": float(np.median([r["cot_tokens"] for r in rollouts if r["effort"] == e])),
                         "mean_cot": mean(r["cot_tokens"] for r in rollouts if r["effort"] == e),
                         "correct": mean(r["correct"] for r in rollouts if r["effort"] == e),
                         "reward": mean(r["reward"] for r in rollouts if r["effort"] == e),
                         "truncated": sum(r["truncated"] for r in rollouts if r["effort"] == e),
                         "empty_cot": mean(r["cot_tokens"] == 0 for r in rollouts if r["effort"] == e),
                         "unparsed": mean(not r["parsed"] for r in rollouts if r["effort"] == e),
                         "looping": mean(r["looping"] for r in rollouts if r["effort"] == e),
                         "non_latin": mean(r["non_latin"] for r in rollouts if r["effort"] == e),
                         "mean_lp": mean(r["mean_lp"] for r in rollouts if r["effort"] == e)} for e in EFFORTS}
        line = {"step": step + 1, "reward": mean(r["reward"] for r in rollouts), "reward_adj": mean(r["reward"] - r["base_acc"] for r in rollouts),
                "task": mean(r["task"] for r in rollouts),
                "correct": mean(r["correct"] for r in rollouts), "len_pen": mean(r["len_pen"] for r in rollouts),
                "final_over_limit": mean(r["resp_tokens"] > args.max_final_tokens for r in rollouts), "truncated": sum(r["truncated"] for r in rollouts),
                "by_effort": by_effort, "n_rollouts": len(rollouts), "n_trained": len(data), "lp_gap": lp_gap,
                "secs_sample": round(t_sample), "secs_train": round(t_train), "secs": round(time.time() - t0), "spent_est": round(spent, 3)}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        blocks.append(line["reward_adj"])

        last = step + 1 == args.steps or spent > args.budget
        if (step + 1) % args.save_every == 0 or last:
            name = f"coteff-{args.run}-step{step + 1}"
            sampler_path = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
            state_path = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            ckpts.append({"step": step + 1, "sampler": sampler_path, "state": state_path})
            (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))
        if (step + 1) % args.plateau_block == 0:
            b = float(np.mean(blocks[-args.plateau_block:]))
            if best is None or b > best + args.plateau_gain:
                best, bad = b, 0
            else:
                bad += 1
            print(f"block ending step {step + 1}: mean reward - base acc {b:.4f} (best {best:.4f}); non-improving blocks {bad}", flush=True)
            if bad >= 2:
                print("plateau; stopping")
                break
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break
    print(f"done. ~${spent:.2f}; checkpoints {out / 'checkpoints.json'}")


if __name__ == "__main__":
    asyncio.run(main())
