"""GRPO (Tinker LoRA) for CoT length control: grpo.py with the Astra style constraints replaced by CoT length targets.

Prompts: a mined MMLU-Pro question from --pool (pool_k4, the grpo3 pool; CoT-Control eval items are excluded by mine.py)
         + "Requirement: Think for about {T} tokens before answering. Keep your final answer short.", T drawn uniformly from --targets per prompt, shared by
         the group. Default targets are the 25th/50th/75th percentiles (rounded to 10) of the base model's analysis-channel
         token count over pool_k4's 2000 questions: 157, 312.5, 647 -> 160, 310, 650.
Reward:  max(0, 1 - |L - T| / T) * (1 if correct else --wrong-factor), where L is the analysis-channel text re-encoded with
         the policy tokenizer (special tokens excluded, as in length_instr.py). 0 if the CoT is empty, the sample hit
         --max-tokens, no parseable 'ANSWER: X' follows the CoT, the grpo.py judge does not call the CoT an attempt at
         the question (only with --judge; it rejected 20/4061 len1 and 4/5089 len2 samples, most of them real attempts;
         len2 used it for steps 1-20 and ran without it from step 21), or the final answer (resp_tokens, final channel re-encoded) is longer than --max-final-tokens. len1
         (prompt "Use about {T} tokens in your analysis channel.", no final-answer gate) learned to read the requirement as
         "write a ~T-token **Analysis** section in the answer" (26% of finals > 200 tokens by steps 13-16). No base-relative length penalty (it would pull against the targets).
Update:  as grpo.py (advantage = reward - group mean, equal-reward groups dropped, importance_sampling, fully on-policy).
         gpt-oss effort is the chat template's default (medium); nothing here sets it.
Effort:  --by-effort (efflen1) drops the Requirement line. The prompt is the bare question block rendered with gpt-oss's own
         "Reasoning: low|medium|high" system line (effort_base.render) and --targets[i] is the token target for low, medium, high
         in that order, so the model is never told a number: the length target is signalled only by the effort level, as in a lab's
         own effort training, and the reward is len2's. The level is drawn uniformly per prompt and shared by the group, like T in len2.
Stop:    at --steps, at --budget, or (--plateau-block > 0) when the last two blocks of --plateau-block steps both fail to beat the
         best earlier block's median rel_dist by --plateau-gain, checked at checkpoints (set --save-every to the block size).
         This is the rule the external len2 plateau watch applied by hand (len2 stopped after step 60).

    python grpo_length.py --run len-smoke --pool data/pool_k4.jsonl --steps 3 --prompts-per-step 8
    python grpo_length.py --run len2 --init-state <step-20 state> --start-step 20 ...   # continue a run in its directory

Writes results/rl/<run>/{steps,rollouts}.jsonl, config.json and checkpoints.json.
"""

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import tinker

from cotswitch import extract, prompts
from cotswitch.data import Item

import modelio
from grpo import JUDGE_MODEL, JUDGE_PRICE, RESULTS, datum, judge_reasons, mean
import effort_base

REQUIREMENT = "Think for about {t} tokens before answering. Keep your final answer short."


def make_task(q: dict, target: int, effort: str | None = None) -> dict:
    """effort None: len2's Requirement line names the target. effort set (--by-effort): bare question, target implied by the level."""
    item = Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"], answer_letter=q["answer"],
                keywords=[], synonyms={}, raw={})
    user = prompts.question_block(item) if effort else f"{prompts.question_block(item)}\n\nRequirement: {REQUIREMENT.format(t=target)}"
    return dict(qid=q["qid"], target=target, effort=effort, answer=q["answer"], user=user, base_acc=q["correct"],
                qblock=prompts.question_block(item).replace(prompts.FORMAT_INSTRUCTION, ""))


def score(task: dict, gen: str, truncated: bool, harmony: bool, think_prefix: str, eos: str, tok) -> dict:
    split = extract.split_output(modelio.to_think(gen, harmony, think_prefix, eos), None, "length" if truncated else "stop")
    text, resp = split["reasoning"], split["response"]
    answer = extract.parse_answer(resp, split["content"])
    cot_tokens = len(tok.encode(text, add_special_tokens=False)) if text else 0
    rel_dist = abs(cot_tokens - task["target"]) / task["target"]
    return dict(fmt_ok=bool(text) and not truncated and answer is not None, correct=answer == task["answer"], truncated=truncated,
                cot_tokens=cot_tokens, resp_tokens=len(tok.encode(resp, add_special_tokens=False)) if resp else 0, rel_dist=round(rel_dist, 4), len_score=max(0.0, 1.0 - rel_dist), reasons=None,
                reasoning=text, response=resp)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b", choices=list(modelio.PRICES))
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--targets", type=int, nargs="+", default=[160, 310, 650], help="CoT token targets")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--max-final-tokens", type=int, default=50, help="reward 0 when the final answer is longer")
    ap.add_argument("--wrong-factor", type=float, default=0.5, help="length-score multiplier for a wrong answer")
    ap.add_argument("--budget", type=float, default=3.0, help="USD; stop when the running estimate passes it")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", action=argparse.BooleanOptionalAction, default=False, help="gate reward on the grpo.py attempt judge")
    ap.add_argument("--by-effort", action="store_true", help="no Requirement line; --targets map to gpt-oss effort low, medium, high")
    ap.add_argument("--plateau-block", type=int, default=0, help="steps per block for the plateau stop; 0 = off")
    ap.add_argument("--plateau-gain", type=float, default=0.01, help="a block must beat the best earlier block's median rel_dist by this")
    ap.add_argument("--init-state", help="tinker:// state path (with optimizer) to continue from")
    ap.add_argument("--start-step", type=int, default=0, help="the step --init-state was saved at; continues that run's directory, "
                    "step numbers, checkpoint names and prompt draws")
    args = ap.parse_args()
    if args.by_effort:
        assert len(args.targets) == len(effort_base.EFFORTS), "--by-effort needs one target per effort level (low, medium, high)"
    effort_of = dict(zip(args.targets, effort_base.EFFORTS)) if args.by_effort else {}

    pool = [json.loads(line) for line in args.pool.open()]
    n_all = len(pool)
    pool = [q for q in pool if q["base_words"] >= 10]
    print(f"pool: {len(pool)} of {n_all} questions have a base CoT of >= 10 words", flush=True)
    out = RESULTS / args.run
    out.mkdir(parents=True, exist_ok=args.start_step > 0)
    PRICE = modelio.PRICES[args.model]
    (out / ("config.json" if args.start_step == 0 else f"config_from_step{args.start_step}.json")).write_text(json.dumps(vars(args) | {"judge_model": JUDGE_MODEL if args.judge else None, "requirement": REQUIREMENT}, indent=1, default=str))

    svc = tinker.ServiceClient()
    tc = await (svc.create_training_client_from_state_with_optimizer_async(args.init_state) if args.init_state
                else svc.create_lora_training_client_async(args.model, rank=args.rank, seed=args.seed))
    tok = tc.get_tokenizer()
    harmony, eos = modelio.probe_harmony(tok), tok.eos_token
    assert harmony or not args.by_effort, "--by-effort uses gpt-oss's Reasoning: line; only harmony models are supported"
    judge = await svc.create_sampling_client_async(base_model=JUDGE_MODEL)
    jtok = judge.get_tokenizer()
    sem = asyncio.Semaphore(64)
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=modelio.policy_stop(tok, harmony))
    rng = random.Random(args.seed)
    for _ in range(args.start_step * args.prompts_per_step):  # replay earlier steps' draws so prompts continue the run's sequence
        rng.choice(pool), rng.choice(args.targets)
    spent = 0.0
    ckpts = json.loads((out / "checkpoints.json").read_text()) if args.start_step else []

    for step in range(args.start_step, args.steps):
        t0 = time.time()
        sampler = await tc.save_weights_and_get_sampling_client_async()
        draws = [(rng.choice(pool), rng.choice(args.targets)) for _ in range(args.prompts_per_step)]
        tasks = [make_task(q, t, effort_of.get(t)) for q, t in draws]
        rendered = [(effort_base.render(tok, t["user"], t["effort"]), "") if args.by_effort else modelio.render(tok, t["user"], harmony) for t in tasks]
        prompts_ids = [p for p, _ in rendered]
        results = await asyncio.gather(*[sampler.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=args.group_size, sampling_params=params)
                                         for p in prompts_ids])
        t_sample = time.time() - t0

        groups = []
        for task, (_, think_prefix), res in zip(tasks, rendered, results):
            assert all(s.logprobs is not None and len(s.logprobs) == len(s.tokens) for s in res.sequences)
            groups.append([(s, score(task, tok.decode(s.tokens, skip_special_tokens=False), s.stop_reason == "length", harmony, think_prefix, eos, tok))
                           for s in res.sequences])
        t1 = time.time()
        todo = [(task, sc) for task, g in zip(tasks, groups) for _, sc in g if sc["fmt_ok"]] if args.judge else []
        verdicts = await asyncio.gather(*[judge_reasons(judge, jtok, task["qblock"], sc["reasoning"], sem) for task, sc in todo])
        for (_, sc), (v, final, n_in, n_out) in zip(todo, verdicts):
            sc["reasons"], sc["judge_final"] = v, final
            spent += (n_in * JUDGE_PRICE["prefill"] + n_out * JUDGE_PRICE["sample"]) / 1e6
        t_judge = time.time() - t1

        rollouts, data, sample_lps = [], [], []
        for task, p, group in zip(tasks, prompts_ids, groups):
            for _, sc in group:
                sc["reward"] = sc["len_score"] * (1.0 if sc["correct"] else args.wrong_factor) if sc["fmt_ok"] and (sc["reasons"] or not args.judge) and sc["resp_tokens"] <= args.max_final_tokens else 0.0
            m = sum(sc["reward"] for _, sc in group) / len(group)
            for s, sc in group:
                rollouts.append(dict(step=step + 1, qid=task["qid"], target=task["target"], effort=task["effort"], base_acc=task["base_acc"],
                                     n_tokens=len(s.tokens), adv=sc["reward"] - m, **sc))
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

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps or spent > args.budget:
            name = f"cotlen-{args.run}-step{step + 1}"
            sampler_path = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
            state_path = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            ckpts.append({"step": step + 1, "sampler": sampler_path, "state": state_path})
            (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))

        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_target = {}
        for t in args.targets:
            rs = [r for r in rollouts if r["target"] == t]
            if rs:
                by_target[t] = {"effort": effort_of.get(t), "n": len(rs), "median_tok": float(np.median([r["cot_tokens"] for r in rs])),
                                "median_rel_dist": round(float(np.median([r["rel_dist"] for r in rs])), 3),
                                "within25": mean(r["rel_dist"] <= 0.25 for r in rs)}
        line = {"step": step + 1, "reward": mean(r["reward"] for r in rollouts), "len_score": mean(r["len_score"] for r in rollouts),
                "within25": mean(r["rel_dist"] <= 0.25 for r in rollouts), "correct": mean(r["correct"] for r in rollouts),
                "acc_vs_base": round(mean(r["correct"] for r in rollouts) - mean(r["base_acc"] for r in rollouts), 4),
                "fmt_ok": mean(r["fmt_ok"] for r in rollouts), "reasons": mean(r["reasons"] is True for r in rollouts if r["fmt_ok"]) if args.judge else None,
                "judge_unparsed": sum(r["fmt_ok"] and r["reasons"] is None for r in rollouts) if args.judge else None, "truncated": sum(r["truncated"] for r in rollouts),
                "tokens_mean": mean(r["n_tokens"] for r in rollouts), "resp_tokens_median": float(np.median([r["resp_tokens"] for r in rollouts])),
                "resp_over_200": mean(r["resp_tokens"] > 200 for r in rollouts), "final_gated": mean(r["resp_tokens"] > args.max_final_tokens for r in rollouts),
                "analysis_heading": mean(bool(re.match(r"\W*analysis", r["response"], re.I)) for r in rollouts),
                "cot_says_token": mean("token" in r["reasoning"].lower() for r in rollouts), "n_rollouts": len(rollouts), "n_trained": len(data), "lp_gap": lp_gap,
                "by_target": by_target, "secs_sample": round(t_sample), "secs_judge": round(t_judge), "secs_train": round(t_train),
                "secs": round(time.time() - t0), "spent_est": round(spent, 3)}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break
        if args.plateau_block and (step + 1) % args.plateau_block == 0 and (step + 1) // args.plateau_block >= 3:
            rows = [json.loads(l) for l in (out / "rollouts.jsonl").open()]
            b = args.plateau_block
            blocks = [float(np.median([r["rel_dist"] for r in rows if (k * b) < r["step"] <= (k + 1) * b])) for k in range((step + 1) // b)]
            best_before = min(blocks[:-2])
            gains = [best_before - x for x in blocks[-2:]]
            print(f"plateau check: block medians {[round(x, 3) for x in blocks]}; last two vs best earlier {[round(g, 3) for g in gains]}", flush=True)
            if all(g < args.plateau_gain for g in gains):
                print(f"plateau: stopping after step {step + 1} checkpoint", flush=True)
                break
    print(f"done. ~${spent:.2f}; checkpoints {out / 'checkpoints.json'}")


if __name__ == "__main__":
    asyncio.run(main())
