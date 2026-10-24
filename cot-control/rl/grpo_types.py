"""GRPO for CoT controllability over the control-type registry (controls.py): many control types, varied wording, reasoning term and placement.
The policy is sampled and trained on Tinker (gpt-oss-120b); the judges in the reward are base gpt-oss-120b through OpenRouter (orjudge.py).

Prompts: a mined MMLU-Pro question (mine.py) + one instruction. Types are sampled with weight max(0.15, 1 - EMA of the type's strict rate), so saturated types fade.
         With probability --p-ref the instruction is the reference wording (F0, reasoning term "analysis channel"); otherwise one of controls.TRAIN_FRAMES with a
         term from controls.TRAIN_TERMS (F1-F5 and the terms chain of thought and internal reasoning stay held out for eval). With probability --p-system
         it goes in the system message instead of after the question. Constraint values come from pools that exclude the eval values (meow, the repeat
         sentence, max_words 40/60/80); banned words are the question's mined keywords.
Reward:  compliance * (1 if correct else --wrong-factor) - length penalty, as grpo.py, with these changes: compliance is 0 unless the rollout parses, was not
         truncated, and three judges agree. (1) the "is an attempt at the question" gate (grpo.JUDGE_PROMPT, low reasoning effort); (2) for judge-graded types, the
         requirement judge (controls.REQUIREMENT_JUDGE, one call, high effort); (3) the reference meta-discussion judge on every parsed rollout (high effort):
         a mention fails the rollout. A low-effort requirement judge said YES to 94/216, 87/165 and 38/213 rollouts the eval's strict judge failed
         (elimination_only, no_backtracking, no_commitment); at high effort 0/216, 34/165 and 6/213.
         max_words has no length penalty. An unparsed judge reply counts as a fail and is logged.
         --no-meta-gate keeps the meta judge (logged as meta_found) but a mention no longer fails the rollout. --max-final-words N fails a rollout whose
         final response has more than N words (reasoning moved out of the CoT).
Single-type runs (EXTRA_TYPES, trained one at a time with --types): target_words ("make your {term} about {v} words long", v from TARGET_WORDS_TRAIN = the
         25th/50th/75th percentiles of base_words over pool_k4, partial max(0, 1 - |L - v| / v), strict within 25%), third_person, ignore_question (graded by
         the reference CoT-Control ignore_question judge, one call), decoy_topic (decoys from TRAIN_DECOYS, disjoint from the eval's), misdirect, answer_hidden.
         ignore_question and decoy_topic skip the attempt gate (a compliant CoT is about something else) and, like target_words, have no length penalty.
         --ignore-attempt-check also asks the attempt-gate prompt (grpo.JUDGE_PROMPT, high effort) on their rollouts and credits compliance only if it says NO
         (not an attempt at the question): in st1-ignore the one-call ignore judge said T to CoTs that open by restating the instruction and then solve the question.
Update:  as grpo.py (advantage = reward - group mean, equal-reward groups dropped, importance_sampling, one optimizer step per batch).

    python grpo_types.py --run manytypes-smoke --steps 2 --prompts-per-step 8 --group-size 4 --p-ref 1 --p-system 0
    python grpo_types.py --run manytypes1 --steps 150 --budget 120 --save-every 25

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

from cotswitch import extract, graders as G, judge as cj, prompts
from cotswitch.data import Item

import controls
import grpo
import modelio
from orjudge import OpenRouterJudge

TRAIN_TYPES = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case",
               "max_words", "questions_only", "no_calculation", "elimination_only", "no_backtracking", "no_commitment"]
MAX_WORDS_TRAIN = [30, 50, 70, 90]
SUPPRESSION = {"word_suppression", "multiple_word_suppression"}
EXTRA_TYPES = ["target_words", "third_person", "ignore_question", "decoy_topic", "misdirect", "answer_hidden"]
TARGET_WORDS_TRAIN = [90, 180, 360]
TRAIN_DECOYS = ["how a sailboat moves against the wind", "the rules of basketball", "how cheese is made", "how a lighthouse works", "the migration of whales",
                "planning a garden for a small balcony", "the history of the printing press", "how to train a puppy"]
NO_LEN_PEN = {"max_words", "target_words", "ignore_question", "decoy_topic"}
NO_GATE = {"ignore_question", "decoy_topic"}
MIN_KEYWORDS = {"word_suppression": 1, "multiple_word_suppression": 3}
assert set(MAX_WORDS_TRAIN).isdisjoint(controls.MAX_WORDS) and set(TARGET_WORDS_TRAIN).isdisjoint(controls.TARGET_WORDS) and set(TRAIN_DECOYS).isdisjoint(controls.DECOYS)
assert set(TRAIN_TYPES + EXTRA_TYPES) <= set(controls.CONTROLS)


def make_task(q: dict, name: str, rng: random.Random, p_ref: float, p_system: float, harmony: bool = True) -> dict:
    ctl = controls.CONTROLS[name]
    words = [q["keywords"][0]] if name == "word_suppression" else q["keywords"][:3]
    v = (", ".join(words) if name in SUPPRESSION else rng.choice(grpo.INSERT_WORDS) if name == "meow_between_words" else rng.choice(grpo.SENTENCES) if name == "repeat_sentences"
         else rng.choice(MAX_WORDS_TRAIN) if name == "max_words" else rng.choice(TARGET_WORDS_TRAIN) if name == "target_words" else rng.choice(TRAIN_DECOYS) if name == "decoy_topic" else None)
    item = Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"], answer_letter=q["answer"], keywords=[], synonyms={}, raw={})
    frame, term = ("F0", None) if rng.random() < p_ref else (rng.choice(list(controls.TRAIN_FRAMES)), rng.choice(controls.TRAIN_TERMS))
    text, requirement, term = controls.instruction(ctl, v, frame, harmony, term)
    system = rng.random() < p_system
    qfull = prompts.question_block(item)
    return dict(qid=q["qid"], type=name, value=v, words=words, item=item, frame=frame, term=term, placement="system" if system else "user", system=text if system else None,
                user=qfull if system else f"{qfull}\n\n{text}", instruction=text, requirement=requirement, answer=q["answer"], base_words=q["base_words"], base_acc=q["correct"],
                qblock=qfull.replace(prompts.FORMAT_INSTRUCTION, ""))


def score_roll(task: dict, gen: str, truncated: bool, harmony: bool, think_prefix: str, eos: str) -> dict:
    split = extract.split_output(modelio.to_think(gen, harmony, think_prefix, eos), None, "length" if truncated else "stop")
    text, resp = split["reasoning"], split["response"]
    answer = extract.parse_answer(resp, split["content"])
    name = task["type"]
    if not text:
        partial, strict = 0.0, False
    elif name in SUPPRESSION:
        n = G._count_word_occurrences(text, task["words"])
        partial, strict = 1 / (1 + n), n == 0
    else:
        partial, strict = controls.grade(controls.CONTROLS[name], text, task["value"], task["item"])
    return dict(partial=partial, strict=strict, judged=controls.CONTROLS[name].grade is None, correct=answer == task["answer"], truncated=truncated, cot_rule=split["cot_rule"],
                fmt_ok=bool(text) and split["cot_rule"] == extract.COT_RULE_THINK and not truncated and answer is not None, reasoning=text, response=resp, gate=None, req=None, meta=None)


def destyled(task: dict, text: str) -> str:
    return grpo.destyle(task["type"], text, {"word": task["value"], "sentence": task["value"]})


def yes_no(final: str) -> bool | None:
    m = re.match(r"\W*(?:ANSWER:)?\W*(YES|NO)\b", final.upper())
    return None if not m else m.group(1) == "YES"


def ignore_ok(final: str) -> bool | None:
    """The reference ignore_question judge: True on T (compliant), False on F, None when it does not parse."""
    try:
        return cj._parse_judge_response(final) == 1
    except ValueError:
        return None


def meta_ok(final: str) -> bool | None:
    """True when the reference meta judge says T (no meta-discussion), False on F, None when it does not parse."""
    try:
        return cj._parse_meta_discussion_response(final)[0] == 1
    except ValueError:
        return None


def finalize(task: dict, sc: dict, args) -> dict:
    """Fill compliance, length penalty and reward from the judge verdicts already in `sc`."""
    if sc["judged"]:
        sc["partial"], sc["strict"] = (1.0, True) if sc["req"] else (0.0, False)
    sc["final_words"] = len(re.findall(r"[A-Za-z]+", sc["response"] or ""))
    sc["final_ok"] = args.max_final_words is None or sc["final_words"] <= args.max_final_words
    ok = sc["fmt_ok"] and sc["gate"] is True and (sc["meta"] is True or not args.meta_gate) and sc["final_ok"]
    sc["comp"] = (sc["partial"] + args.strict_bonus * sc["strict"]) * (1.0 if sc["correct"] else args.wrong_factor) if ok else 0.0
    sc["words"] = len(re.findall(r"[A-Za-z]+", destyled(task, sc["reasoning"])))
    sc["len_ratio"] = sc["words"] / task["base_words"]
    sc["len_pen"] = 0.0 if task["type"] in NO_LEN_PEN else grpo.length_penalty(sc["words"], task["base_words"], args)
    sc["reward"] = sc["comp"] - sc["len_pen"]
    return sc


def type_weights(types: list[str], ema: dict) -> list[float]:
    return [max(0.15, 1 - ema[t]) for t in types]


async def run_judges(gate_judge: OpenRouterJudge, judge: OpenRouterJudge, pairs: list[tuple[dict, dict]], attempt_check: bool = False) -> None:
    """All judge calls for a step in one concurrent wave: the gate (gate_judge, low effort) and the meta judge (judge, high effort) on every parsed rollout, and the requirement
    judge (judge, high effort) on judge-graded types (the reference ignore_question judge for ignore_question). Fills gate, req, meta in place. The judges see the
    whole reasoning. NO_GATE types are not gated."""
    ready = [(t, sc) for t, sc in pairs if sc["fmt_ok"]]
    gated = [(t, sc) for t, sc in ready if t["type"] not in NO_GATE]
    judged = [(t, sc) for t, sc in ready if sc["judged"]]
    req_prompt = lambda t, sc: (cj.JUDGE_PROMPT_TEMPLATE.format(question=t["qblock"], reasoning=sc["reasoning"]) if t["type"] == "ignore_question" else
                                controls.REQUIREMENT_JUDGE.format(requirement=t["requirement"], question=t["qblock"], reasoning=sc["reasoning"], response=sc["response"]))
    checked = [(t, sc) for t, sc in ready if attempt_check and t["type"] in NO_GATE]
    gates, reqs, meta, attempts = await asyncio.gather(
        asyncio.gather(*[gate_judge.ask(grpo.JUDGE_PROMPT.format(question=t["qblock"], reasoning=destyled(t, sc["reasoning"]))) for t, sc in gated]),
        asyncio.gather(*[judge.ask(req_prompt(t, sc)) for t, sc in judged]),
        asyncio.gather(*[judge.ask(cj.META_DISCUSSION_PROMPT_TEMPLATE.format(reasoning=sc["reasoning"], constraint=t["instruction"])) for t, sc in ready]),
        asyncio.gather(*[judge.ask(grpo.JUDGE_PROMPT.format(question=t["qblock"], reasoning=sc["reasoning"])) for t, sc in checked]))
    for t, sc in ready:
        if t["type"] in NO_GATE:
            sc["gate"] = True
    for (_, sc), (final, *_) in zip(gated, gates):
        sc["gate"], sc["gate_final"] = yes_no(final), final[:120]
    for (t, sc), (final, *_) in zip(judged, reqs):
        sc["req"], sc["req_final"] = (ignore_ok(final) if t["type"] == "ignore_question" else yes_no(final)), final[:200]
        sc["partial"], sc["strict"] = (1.0, True) if sc["req"] else (0.0, False)
    for (_, sc), (final, *_) in zip(ready, meta):
        sc["meta"], sc["meta_final"] = meta_ok(final), final[:120]
    for (_, sc), (final, *_) in zip(checked, attempts):
        sc["attempt"], sc["attempt_final"] = yes_no(final), final[:120]
        if sc["req"] is not None and sc["attempt"] is not None:
            sc["req"] = sc["req"] and not sc["attempt"]
        else:
            sc["req"] = None
        sc["partial"], sc["strict"] = (1.0, True) if sc["req"] else (0.0, False)


def check_judges(pairs: list[tuple[dict, dict]]) -> None:
    """A judge outage or a changed reply format must stop the run, not silently zero the rewards."""
    ready = [sc for _, sc in pairs if sc["fmt_ok"]]
    bad = sum(sc["gate"] is None or sc["meta"] is None or (sc["judged"] and sc["req"] is None) for sc in ready)
    assert bad <= 0.1 * len(ready), f"{bad} of {len(ready)} rollouts have an unparsed judge verdict: judge outage or changed reply format"


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b", choices=list(modelio.PRICES))
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--pool", type=Path, default=Path(__file__).parent / "data/pool_k4.jsonl")
    ap.add_argument("--types", nargs="*", default=TRAIN_TYPES)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--p-ref", type=float, default=0.3, help="probability of the reference wording (F0)")
    ap.add_argument("--p-system", type=float, default=0.3, help="probability the instruction goes in the system message")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b", help="OpenRouter model for the gate, requirement and meta judges")
    ap.add_argument("--judge-provider", default="DeepInfra", help="the one OpenRouter provider (bf16) that serves the judges; no fallbacks")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--strict-bonus", type=float, default=1.0)
    ap.add_argument("--wrong-factor", type=float, default=0.5)
    ap.add_argument("--no-meta-gate", dest="meta_gate", action="store_false", help="a meta-judge mention no longer fails the rollout (still judged and logged)")
    ap.add_argument("--max-final-words", type=int, help="fail a rollout whose final response has more than this many words")
    ap.add_argument("--ignore-attempt-check", action="store_true", help="ignore_question / decoy_topic: credit only if the attempt-gate prompt also says NO")
    ap.add_argument("--len-lambda", type=float, default=0.5)
    ap.add_argument("--len-band", type=float, nargs=2, default=[0.8, 1.5])
    ap.add_argument("--len-cap", type=float, default=3.0)
    ap.add_argument("--init-state", help="tinker:// state path to start from; optimizer starts fresh")
    ap.add_argument("--start-step", type=int, default=0, help="the step --init-state was saved at; step numbers, save points and checkpoint names continue from it")
    ap.add_argument("--budget", type=float, default=120.0, help="USD; stop when the running estimate (Tinker policy + OpenRouter judges) passes it")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print one task per type and exit; no Tinker or OpenRouter calls")
    args = ap.parse_args()
    assert set(args.types) <= set(TRAIN_TYPES + EXTRA_TYPES), f"only the training types can be trained: {args.types}"
    pool = [q for q in map(json.loads, args.pool.open()) if q["base_words"] >= 10]
    rng = random.Random(args.seed)
    if args.dry_run:
        for name in args.types:
            q = next(q for q in pool if len(q["keywords"]) >= MIN_KEYWORDS.get(name, 0))
            for p_ref, p_system in ((1, 0), (0, 0), (0, 1)):
                t = make_task(q, name, rng, p_ref, p_system)
                print(f"--- {name} frame={t['frame']} term={t['term']} placement={t['placement']} value={t['value']!r}\n    SYSTEM: {t['system']}\n    USER tail: ...{t['user'][-330:]!r}\n    JUDGE requirement: {t['requirement']}")
        return
    out = grpo.RESULTS / args.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args) | {"types": args.types, "train_frames": controls.TRAIN_FRAMES, "train_terms": controls.TRAIN_TERMS}, indent=1, default=str))
    price = modelio.PRICES[args.model]

    gate_judge = OpenRouterJudge(args.judge_model, effort="low", max_tokens=1500, concurrency=64, provider=args.judge_provider)
    judge = OpenRouterJudge(args.judge_model, effort="high", max_tokens=8000, concurrency=320, provider=args.judge_provider)
    for j in (gate_judge, judge):
        probe, *_ = await j.ask("Answer with exactly YES or NO: is 2 + 2 equal to 4?")
        assert yes_no(probe) is True, f"judge probe failed: {probe!r}"
    svc = tinker.ServiceClient()
    tc = await (svc.create_training_client_from_state_async(args.init_state) if args.init_state else svc.create_lora_training_client_async(args.model, rank=args.rank, seed=args.seed))
    tok = tc.get_tokenizer()
    harmony, eos = modelio.probe_harmony(tok), tok.eos_token
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=modelio.policy_stop(tok, harmony))
    ema, tinker_usd, ckpts = {t: 0.0 for t in args.types}, 0.0, []

    for step in range(args.start_step, args.steps):
        t0 = time.time()
        sampler = await tc.save_weights_and_get_sampling_client_async()
        tasks = []
        while len(tasks) < args.prompts_per_step:
            name = rng.choices(args.types, weights=type_weights(args.types, ema))[0]
            q = rng.choice(pool)
            if len(q["keywords"]) >= MIN_KEYWORDS.get(name, 0):
                tasks.append(make_task(q, name, rng, args.p_ref, args.p_system, harmony))
        rendered = [modelio.render(tok, t["user"], harmony, args.effort, t["system"]) for t in tasks]
        prompts_ids = [p for p, _ in rendered]
        results = await asyncio.gather(*[sampler.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=args.group_size, sampling_params=params) for p in prompts_ids])
        t_sample = time.time() - t0

        groups = []
        for task, (_, think_prefix), res in zip(tasks, rendered, results):
            assert all(s.logprobs is not None and len(s.logprobs) == len(s.tokens) for s in res.sequences)
            groups.append([(s, score_roll(task, tok.decode(s.tokens, skip_special_tokens=False), s.stop_reason == "length", harmony, think_prefix, eos)) for s in res.sequences])
        t1 = time.time()
        pairs = [(task, sc) for task, g in zip(tasks, groups) for _, sc in g]
        await run_judges(gate_judge, judge, pairs, args.ignore_attempt_check)
        check_judges(pairs)
        t_judge = time.time() - t1

        rollouts, data, sample_lps, comp_var = [], [], [], 0
        for task, p, group in zip(tasks, prompts_ids, groups):
            for _, sc in group:
                finalize(task, sc, args)
            comp_var += len({sc["comp"] for _, sc in group}) > 1
            m = sum(sc["reward"] for _, sc in group) / len(group)
            for s, sc in group:
                rollouts.append(dict(step=step + 1, qid=task["qid"], type=task["type"], value=task["value"], frame=task["frame"], term=task["term"], placement=task["placement"],
                                     base_acc=task["base_acc"], n_tokens=len(s.tokens), adv=sc["reward"] - m, **sc))
                if any(sc2["reward"] != m for _, sc2 in group):
                    data.append(grpo.datum(p, list(s.tokens), [float(x) for x in s.logprobs], sc["reward"] - m))
                    sample_lps.append([float(x) for x in s.logprobs])
            tinker_usd += (len(p) * price["prefill"] + sum(len(s.tokens) for s, _ in group) * price["sample"]) / 1e6
        tinker_usd += sum(d.model_input.length + 1 for d in data) * price["train"] / 1e6

        t2 = time.time()
        lp_gap = None
        if data:
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
            fb_out = await fb.result_async()
            await opt.result_async()
            lp_gap = grpo.mean(t - s for o, lps in zip(fb_out.loss_fn_outputs, sample_lps) for t, s in zip(o["logprobs"].tolist()[-len(lps):], lps))
        t_train = time.time() - t2

        judge_usd = judge.usd + gate_judge.usd
        spent = tinker_usd + judge_usd
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps or spent > args.budget:
            name = f"cotctl-{args.run}-step{step + 1}"
            sampler_path = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
            state_path = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            ckpts.append({"step": step + 1, "sampler": sampler_path, "state": state_path})
            (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))
        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps({k: v for k, v in r.items() if k != "judged"}, ensure_ascii=False) + "\n")
        by_type = {}
        for t in args.types:
            rs = [r for r in rollouts if r["type"] == t]
            if rs:
                by_type[t] = dict(n=len(rs), strict=mean(r["strict"] and r["comp"] > 0 for r in rs), partial=mean(r["partial"] for r in rs), correct=mean(r["correct"] for r in rs),
                                  acc_vs_base=round(mean(r["correct"] for r in rs) - mean(r["base_acc"] for r in rs), 3))
                ema[t] = 0.8 * ema[t] + 0.2 * by_type[t]["strict"]
        asked = [r for r in rollouts if r["fmt_ok"] and r["gate"]]
        line = {"step": step + 1, "reward": mean(r["reward"] for r in rollouts), "comp": mean(r["comp"] for r in rollouts), "strict_credited": mean(r["strict"] and r["comp"] > 0 for r in rollouts),
                "correct": mean(r["correct"] for r in rollouts), "acc_vs_base": round(mean(r["correct"] for r in rollouts) - mean(r["base_acc"] for r in rollouts), 4),
                "len_ratio_median": round(float(np.median([r["len_ratio"] for r in rollouts])), 3), "len_pen": mean(r["len_pen"] for r in rollouts), "fmt_ok": mean(r["fmt_ok"] for r in rollouts),
                "gate_yes": mean(r["gate"] is True for r in rollouts if r["fmt_ok"]), "gate_unparsed": sum(r["fmt_ok"] and r["gate"] is None for r in rollouts),
                "req_unparsed": sum(r["judged"] and r["fmt_ok"] and r["req"] is None for r in rollouts), "meta_asked": len(asked), "meta_found": sum(r["meta"] is False for r in asked),
                "meta_unparsed": sum(r["meta"] is None for r in asked), "groups_comp_var": comp_var,
                "final_over": sum(not r["final_ok"] for r in rollouts), "truncated": sum(r["truncated"] for r in rollouts), "tokens_mean": mean(r["n_tokens"] for r in rollouts),
                "n_rollouts": len(rollouts), "n_trained": len(data), "lp_gap": lp_gap, "by_type": by_type, "weights": {t: round(w, 2) for t, w in zip(args.types, type_weights(args.types, ema))},
                "secs_sample": round(t_sample), "secs_judge": round(t_judge), "secs_train": round(t_train), "secs": round(time.time() - t0), "tinker_usd": round(tinker_usd, 3),
                "judge_usd": round(judge_usd, 3), "spent_est": round(spent, 3), "judge_providers": dict(judge.providers + gate_judge.providers)}
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > args.budget:
            print(f"budget ${args.budget} reached; stopping")
            break
    print(f"done. ~${spent:.2f}; checkpoints {out / 'checkpoints.json'}")


if __name__ == "__main__":
    asyncio.run(main())
