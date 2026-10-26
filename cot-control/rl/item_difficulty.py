"""Per-item difficulty for the transfer-eval items: base gpt-oss-120b accuracy with a CoT and without one, --samples samples each, on the exact
no-requirement prompt the transfer eval used (the baseline arm's rendered_prompt in results/transfer/gptoss-base/gens.jsonl).

CoT: the rendered prompt as is, temperature 1, top_p 1, --max-tokens; answer parsed as transfer_eval.py does. No CoT: the rendered prompt plus an empty
analysis channel and the final-channel head with 'ANSWER:' prefilled ('<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>ANSWER:',
as hintverb_trunc.py at fraction 0), temperature 1, 8 tokens. Sampled on Tinker (prefill needs it; same renderer as the eval).

    python item_difficulty.py --datasets gpqa --out ../results/difficulty/gpqa_base.jsonl
    python item_difficulty.py --datasets gpqa hle mmlu_pro --nocot-only --model-path tinker://.../cotctl-manytypes2e-step170 --out ../results/difficulty/nocot_mt2e-s170.jsonl

--model-path samples a checkpoint instead of base, on the same (base's) rendered prompts; --nocot-only skips the CoT samples.

Writes one row per item: dataset, item_idx, answer, cot_correct / nocot_correct (counts of --samples), cot tokens (median, truncated count).
"""
import argparse
import asyncio
import json
import statistics
from pathlib import Path

import tinker

from cotswitch import extract

import modelio

RESULTS = Path(__file__).resolve().parent.parent / "results"
NOCOT = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>ANSWER:"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["gpqa"])
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", help="tinker:// sampler checkpoint of gpt-oss-120b; base when omitted")
    ap.add_argument("--nocot-only", action="store_true")
    args = ap.parse_args()
    base = [json.loads(l) for l in open(RESULTS / "transfer/gptoss-base/gens.jsonl")]
    items = [r for r in base if r["type"] == "baseline" and r["dataset"] in args.datasets]
    svc = tinker.ServiceClient()
    client = await (svc.create_sampling_client_async(model_path=args.model_path) if args.model_path else svc.create_sampling_client_async(base_model="openai/gpt-oss-120b"))
    tok = client.get_tokenizer()
    stop = modelio.policy_stop(tok, True)
    cot_p = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop=stop)
    nocot_p = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8, stop=stop)
    price, spent = modelio.PRICES["openai/gpt-oss-120b"], [0.0]
    sem = asyncio.Semaphore(16)

    async def one(r):
        ids = tok.encode(r["rendered_prompt"], add_special_tokens=False)
        nocot_ids = ids + tok.encode(NOCOT, add_special_tokens=False)
        async with sem:
            nocot = await client.sample_async(prompt=tinker.ModelInput.from_ints(nocot_ids), num_samples=args.samples, sampling_params=nocot_p)
            cot = None if args.nocot_only else await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=args.samples, sampling_params=cot_p)
        cot_ok, toks, trunc = 0, [], 0
        for s in (cot.sequences if cot else []):
            finish = "length" if s.stop_reason == "length" else "stop"
            split = extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), True, "", tok.eos_token), None, finish)
            cot_ok += extract.parse_answer(split["response"], split["content"]) == r["answer"]
            toks.append(len(s.tokens))
            trunc += finish == "length"
        nocot_ans = [extract.parse_answer("ANSWER:" + tok.decode(s.tokens, skip_special_tokens=True), "") for s in nocot.sequences]
        spent[0] += (len(ids) * (not args.nocot_only) + len(nocot_ids)) * args.samples * price["prefill"] / 1e6 + (sum(toks) + sum(len(s.tokens) for s in nocot.sequences)) * price["sample"] / 1e6
        return dict(dataset=r["dataset"], item_idx=r["item_idx"], answer=r["answer"], samples=args.samples, model_path=args.model_path, cot_correct=None if args.nocot_only else cot_ok, nocot_correct=sum(a == r["answer"] for a in nocot_ans),
                    nocot_unparsed=sum(a is None for a in nocot_ans), cot_tokens_median=statistics.median(toks) if toks else None, cot_truncated=trunc)

    rows = await asyncio.gather(*[one(r) for r in items])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} items x {args.samples} samples, {'no-CoT' if args.nocot_only else 'CoT and no-CoT'}, about ${spent[0]:.2f} -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
