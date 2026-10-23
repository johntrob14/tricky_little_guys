"""Held-out eval of reasoning-effort control: base gpt-oss-120b vs an effort-trained checkpoint at low / medium / high.

Question sets (no requirement, CoT-Control question block, gpt-oss's own "Reasoning: <level>" system line):
  history:   the 150 held-out MMLU-Pro history questions of length_eval.py (outside pool_k4), --samples-history each
  benchmark: every CoT-Control GPQA, HLE and MMLU-Pro item (1,214; none in pool_k4), --samples-benchmark each
T 1, top_p 1 (the training setting), max --max-tokens. Rows append to <out>/rows.jsonl keyed by (model, set, qid, effort, sample_idx).

    uv run python effort_eval.py --model-path <eff1 step 40 sampler> --out ../results/effort_eval
"""

import argparse
import asyncio
import json
from pathlib import Path

import tinker

from cotswitch import data, extract, prompts

import modelio
from effort_base import EFFORTS, render
from length_eval import heldout_items

BASE = "openai/gpt-oss-120b"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--name", default="eff1s40")
    ap.add_argument("--samples-history", type=int, default=4)
    ap.add_argument("--samples-benchmark", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    hist = [it for it in heldout_items() if it.raw["category"] == "history"][:150]
    bench = [it for ds in ("gpqa", "hle", "mmlu_pro") for it in data.load_items(ds)]
    sets = [("history", it, args.samples_history) for it in hist] + [("benchmark", it, args.samples_benchmark) for it in bench]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows.jsonl"
    done = {(r["model"], r["set"], r["dataset"], r["qid"], r["effort"]) for r in map(json.loads, open(path))} if path.exists() else set()
    sc = tinker.ServiceClient()
    sem, lock, n_tok = asyncio.Semaphore(args.concurrency), asyncio.Lock(), [0, 0]

    async def run_model(name, model_path):
        client = await (sc.create_sampling_client_async(model_path=model_path) if model_path else sc.create_sampling_client_async(base_model=BASE))
        tok = client.get_tokenizer()
        params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, stop=modelio.GPT_OSS_STOP)

        async def one(setname, it, n, effort):
            if (name, setname, it.dataset, it.item_idx, effort) in done:
                return
            ids = render(tok, prompts.question_block(it), effort)
            async with sem:
                res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=n, sampling_params=params)
            rows = []
            for k, s in enumerate(res.sequences):
                finish = "length" if s.stop_reason == "length" else "stop"
                split = extract.split_output(extract.harmony_to_think(tok.decode(s.tokens, skip_special_tokens=False)), None, finish)
                rows.append(dict(model=name, set=setname, dataset=it.dataset, qid=it.item_idx, category=it.raw.get("category"), effort=effort,
                                 sample_idx=k, finish=finish, cot_tokens=len(tok.encode(split["reasoning"], add_special_tokens=False)),
                                 resp_tokens=len(tok.encode(split["response"], add_special_tokens=False)),
                                 correct=extract.parse_answer(split["response"], split["content"]) == it.answer_letter,
                                 reasoning=split["reasoning"], response=split["response"]))
            async with lock:
                n_tok[0] += sum(len(s.tokens) for s in res.sequences)
                n_tok[1] += len(ids) * n
                with path.open("a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)

        await asyncio.gather(*(one(s, it, n, e) for s, it, n in sets for e in EFFORTS))

    await asyncio.gather(run_model("base", None), run_model(args.name, args.model_path))
    print(f"sampled {n_tok[0]} tokens, prompt {n_tok[1]} "
          f"(~${n_tok[0] * modelio.PRICES[BASE]['sample'] / 1e6 + n_tok[1] * modelio.PRICES[BASE]['prefill'] / 1e6:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
