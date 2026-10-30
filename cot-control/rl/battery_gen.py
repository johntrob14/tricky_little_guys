"""Shared sanity battery (sanity/battery.py) answers from base gpt-oss-120b or a Tinker checkpoint, for battery_judge.py.

Sampling as battery.SAMPLING (2 samples, T 0.7, top_p 0.95, no system prompt, the chat template, so gpt-oss's default effort,
medium) except max_tokens 3000 instead of 600: 600 would often end inside gpt-oss's analysis channel. 'answer' is the final channel.
--model DeepSeek V4.1 Flash renders with modelio.render_ds (thinking mode, effort 75) and allows DS_MAX_TOKENS; 'answer' is the
text after </think>. --model Qwen3.8-27B renders with the chat template's defaults (the xhigh reasoning-effort system prompt the Qwen
CoT-control organisms trained under) and allows QWEN_MAX_TOKENS; 'answer' is the text after </think>.

    uv run python battery_gen.py --name base --out ../results/battery
    uv run python battery_gen.py --name eff2s10 --model-path tinker://... --out ../results/battery
    uv run python battery_gen.py --name ds-base --model deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144 --out ../results/battery
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import tinker

from cotswitch import extract

import modelio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sanity import battery  # noqa: E402

BASE = "openai/gpt-oss-120b"
DEEPSEEK = "deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144"
QWEN = "Qwen/Qwen3.8-27B"
MAX_TOKENS = 3000
DS_MAX_TOKENS = 8000
QWEN_MAX_TOKENS = 16000


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--model", default=BASE, choices=[BASE, DEEPSEEK, QWEN], help="the base model, or the one --model-path was trained from")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    harmony = args.model == BASE
    sc = tinker.ServiceClient()
    client = await (sc.create_sampling_client_async(model_path=args.model_path) if args.model_path else sc.create_sampling_client_async(base_model=args.model))
    tok = client.get_tokenizer()
    s = battery.SAMPLING
    max_tokens = MAX_TOKENS if harmony else QWEN_MAX_TOKENS if args.model == QWEN else DS_MAX_TOKENS
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=s["temperature"], top_p=s["top_p"],
                                   stop=modelio.policy_stop(tok, harmony))

    async def one(qi, kind, question):
        text = (tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
                if harmony or args.model == QWEN else modelio.render_ds(question))
        res = await client.sample_async(prompt=tinker.ModelInput.from_ints(tok.encode(text, add_special_tokens=False)),
                                        num_samples=s["n"], sampling_params=params)
        rows = []
        for k, seq in enumerate(res.sequences):
            finish = "length" if seq.stop_reason == "length" else "stop"
            gen = tok.decode(seq.tokens, skip_special_tokens=False)
            split = extract.split_output(modelio.to_think(gen, harmony, "<think>", tok.eos_token), None, finish)
            rows.append(dict(name=args.name, model_path=args.model_path, version=battery.VERSION, qi=qi, kind=kind, question=question,
                             sample=k, finish=finish, answer=split["response"], reasoning=split["reasoning"],
                             cot_tokens=len(tok.encode(split["reasoning"], add_special_tokens=False))))
        return rows

    rows = [r for rs in await asyncio.gather(*(one(i, k, q) for i, (k, q) in enumerate(battery.BATTERY))) for r in rs]
    with (args.out / f"{args.name}.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(battery.signals(rows)))


if __name__ == "__main__":
    asyncio.run(main())
