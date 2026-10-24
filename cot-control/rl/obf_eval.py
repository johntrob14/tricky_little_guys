"""Eval of a grpo_obf.py checkpoint, or the base model, on obf_data.py's eval problems.

Arms: every type in grpo_obf.EVAL_TYPES (grpo_obf.EVAL_ONLY types are not trained) in the reference wording (F0) in the user turn, on the problems
grpo_obf.eligible admits for it (build_tasks), a no-instruction arm (baseline) and a no-CoT arm (nocot: no instruction, DeepSeek's chat mode, which
closes the think block at once, or for gpt-oss the final channel opened by the prompt, with the response prefilled with NOCOT_PREFILL and at most
NOCOT_TOKENS sampled tokens; without the cap the model works the problem in its visible response). One sample per (problem, arm), T 1.0, top_p 1.0,
--max-tokens, effort as in training. --model as grpo_obf.py; the pool must be mined with it.
Scoring: grpo_obf.score_roll and grpo_obf.run_judges, the training graders and judges (one call each, temperature 0). pass = the training reward's
compliance condition without correctness: parsed, not truncated, strict compliance, gate and no meta-discussion; passed_correct = pass and correct.
Reader: the predictor (grpo_obf.PREDICT_JUDGE) on every parsed rollout with a CoT, baseline included; predictable = its guess equals the model's
answer; reader_conf = its 0-10 confidence.

    python obf_eval.py --run base
    python obf_eval.py --run obf1-s10 --model-path tinker://.../sampler_weights/cotctl-obf1-step10
    python obf_eval.py --run base-or-internal --provider Parasail --term "internal reasoning" --arms ignore_question misdirect ...

--provider samples the base model through OpenRouter raw completions on that provider (no fallbacks; prompt rendered by modelio.render_ds, prompt
token count asserted) instead of Tinker; thinking arms only.

Writes results/obf_eval/<run>/{rows.jsonl,summary.json,config.json} and prints the per-arm table.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
from pathlib import Path

import httpx
import tinker
from tokenizers import Tokenizer

import grpo_obf
import modelio
from grpo_obf import CONTROLS, EVAL_TYPES, MODEL, MODELS
from obf_data import HF_TOKENIZER, OR_MODEL, complete

RESULTS = Path(__file__).resolve().parent.parent / "results/obf_eval"
ARMS = EVAL_TYPES + ["baseline", "nocot"]
NOCOT_PREFILL, NOCOT_TOKENS = "ANSWER:", 10
FINAL_CHANNEL = "<|channel|>final<|message|>"  # opens gpt-oss's visible response with no analysis channel before it


def frac(xs: list[bool]) -> str:
    return f"{sum(xs)}/{len(xs)}"


def build_tasks(pool: list[dict], arms: list[str], rng: random.Random, term: str | None, harmony: bool) -> list[dict]:
    """One F0 user-turn task per (arm, eligible problem), arm by arm, from one rng: appending arms leaves the earlier arms' prompts unchanged."""
    return [grpo_obf.make_task(q, arm, rng, 1.0, 0.0, term, harmony) for arm in arms for q in pool if grpo_obf.eligible(arm, q)]


def summarize(pairs: list[tuple[dict, dict]], arms: list[str]) -> dict:
    """Per-arm counts over scored (task, score) pairs whose rows carry passed (and predictable where the reader ran)."""
    summary = {}
    for arm in arms:
        rs = [sc for t, sc in pairs if t["type"] == arm]
        summary[arm] = dict(n=len(rs), passed=frac([sc["passed"] for sc in rs]) if arm in CONTROLS else None,
                            passed_correct=frac([bool(sc["passed"] and sc["correct"]) for sc in rs]) if arm in CONTROLS else None,
                            correct=frac([sc["correct"] for sc in rs]),
                            correct_dapo=frac([sc["correct"] for t, sc in pairs if t["type"] == arm and t["source"] == "dapo"]),
                            correct_gpqa=frac([sc["correct"] for t, sc in pairs if t["type"] == arm and t["source"] == "gpqa"]),
                            predictable=frac([sc["predictable"] for sc in rs if "predictable" in sc]), parsed=frac([sc["fmt_ok"] for sc in rs]),
                            truncated=frac([sc["truncated"] for sc in rs]), meta_found=frac([sc["meta"] is False for sc in rs if sc["meta"] is not None]),
                            median_tokens=statistics.median(sc["n_tokens"] for sc in rs))
    return summary


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=MODEL, choices=MODELS)
    ap.add_argument("--model-path", help="tinker:// sampler checkpoint of --model; the base model when omitted")
    ap.add_argument("--tinker-key-env", default="TINKER_API_KEY")
    ap.add_argument("--answer-only-response", action="store_true", help="as grpo_obf.py: the format fails when the response has words besides 'ANSWER: X'")
    ap.add_argument("--pool", type=Path, default=Path(__file__).parent / "data/obf_eval.jsonl")
    ap.add_argument("--effort", type=int, default=75)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--judge-model", default=grpo_obf.JUDGE_MODEL)
    ap.add_argument("--judge-provider", default="DeepInfra")
    ap.add_argument("--judge-max-tokens", type=int, default=8000, help="as grpo_obf.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--term", help="reasoning term in the reference wording; 'thinking' when omitted")
    ap.add_argument("--provider", help="sample the base model through OpenRouter on this provider instead of Tinker")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    args = ap.parse_args()
    harmony = grpo_obf.is_harmony(args.model)
    out = RESULTS / args.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args) | {"model": OR_MODEL if args.provider else args.model, "nocot_prefill": NOCOT_PREFILL, "nocot_tokens": NOCOT_TOKENS}, indent=1, default=str))
    pool = [json.loads(l) for l in args.pool.open()]
    assert all(q.get("model", MODEL) == args.model for q in pool), f"{args.pool} was not mined with {args.model}"
    tasks = build_tasks(pool, args.arms, random.Random(args.seed), args.term, harmony)

    judge = await grpo_obf.make_judge(args.judge_model, args.judge_provider, args.judge_max_tokens)
    if args.provider:
        assert not args.model_path and not harmony and "nocot" not in args.arms, "OpenRouter serves the base DeepSeek model, thinking arms only"
        hf_tok, client, sem = Tokenizer.from_pretrained(HF_TOKENIZER), httpx.AsyncClient(), asyncio.Semaphore(48)
    else:
        svc = tinker.ServiceClient(api_key=os.environ[args.tinker_key_env])
        sampler = await (svc.create_sampling_client_async(model_path=args.model_path) if args.model_path else svc.create_sampling_client_async(base_model=args.model))
        tok = sampler.get_tokenizer()
        assert modelio.probe_harmony(tok) == harmony
        stop = modelio.policy_stop(tok, harmony)
        params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=stop, seed=args.seed)
        nocot_params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=NOCOT_TOKENS, stop=stop, seed=args.seed)
        sem = asyncio.Semaphore(256)

    async def one_or(t):
        text = modelio.render_ds(t["user"], t["system"], args.effort)
        body = {"model": OR_MODEL, "prompt": text, "max_tokens": args.max_tokens, "temperature": 1.0, "top_p": 1.0, "n": 1,
                "provider": {"order": [args.provider], "allow_fallbacks": False}, "usage": {"include": True}}
        async with sem:
            b = await complete(client, body)
        assert b["usage"]["prompt_tokens"] == len(hf_tok.encode(text, add_special_tokens=False).ids), (b["usage"]["prompt_tokens"], t["qid"])
        c = b["choices"][0]
        truncated = c["finish_reason"] == "length"
        gen = (c.get("reasoning") or "") + ("</think>" + (c.get("text") or "") if c.get("text") or not truncated else "")  # as obf_data.py mine
        return grpo_obf.score_roll(t, gen, truncated, "", "<think>") | dict(n_tokens=b["usage"]["completion_tokens"], cost=float(b["usage"].get("cost") or 0))

    async def one(t):
        cot = t["type"] != "nocot"
        prefill = "" if cot else (FINAL_CHANNEL if harmony else "") + NOCOT_PREFILL
        ids = grpo_obf.encode(tok, t, args.effort, cot or harmony, harmony) + tok.encode(prefill, add_special_tokens=False)
        async with sem:
            res = await sampler.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params if cot else nocot_params)
        s = res.sequences[0]
        sc = grpo_obf.score_roll(t, prefill + tok.decode(s.tokens, skip_special_tokens=False), s.stop_reason == "length", tok.eos_token, "<think>" if cot else "<think></think>",
                                 args.answer_only_response, harmony)
        return sc | dict(n_tokens=len(s.tokens))

    scores = await asyncio.gather(*((one_or if args.provider else one)(t) for t in tasks))
    pairs = list(zip(tasks, scores))
    await grpo_obf.run_judges(judge, pairs)
    readable = [(t, sc) for t, sc in pairs if sc["fmt_ok"] and t["type"] != "nocot"]
    guesses = await asyncio.gather(*[judge.ask(grpo_obf.predict_prompt(t, sc)) for t, sc in readable])
    for (t, sc), (final, *_) in zip(readable, guesses):
        sc["reader_guess"], sc["reader_conf"] = grpo_obf.predicted(t, final), grpo_obf.confidence(final)
        sc["predictable"] = sc["reader_guess"] == sc["answer"]
    with (out / "rows.jsonl").open("w") as f:
        for t, sc in pairs:
            sc["passed"] = bool(sc["fmt_ok"] and sc["strict"] and sc["gate"] is True and sc["meta"] is True) if t["type"] in CONTROLS else None
            f.write(json.dumps({k: v for k, v in t.items() if k != "q"} | sc, ensure_ascii=False, default=str) + "\n")

    summary = summarize(pairs, args.arms)
    (out / "summary.json").write_text(json.dumps(dict(summary=summary, judge_usd=round(judge.usd, 3), sample_usd=round(sum(sc.get("cost", 0) for sc in scores), 3),
                                                      judge_timeouts=judge.timeouts), indent=1))
    print(f"{args.run}: {args.model_path or (OR_MODEL if args.provider else args.model)}; judges ${judge.usd:.2f}, {judge.timeouts} judge calls timed out (unparsed)")
    print("arm | passed | passed and correct | correct | correct DAPO | correct GPQA | answer predictable from CoT | parsed | truncated | meta found | median tokens")
    for arm, s in summary.items():
        print(f"{arm} | {s['passed'] or '-'} | {s['passed_correct'] or '-'} | {s['correct']} | {s['correct_dapo']} | {s['correct_gpqa']} | {s['predictable']} | "
              f"{s['parsed']} | {s['truncated']} | {s['meta_found']} | {s['median_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
