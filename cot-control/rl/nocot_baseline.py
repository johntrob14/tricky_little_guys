"""No-CoT baseline for ignore_question: base gpt-oss-120b answering with its analysis channel forced empty.

The rendered prompt is followed by an empty analysis message and an opened final channel
('<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>'), so the model writes only the answer.
Items: the 108 transfer-eval items (transfer_eval.shared_items) and the 12 Astra ignore_question items. Two arms per item:
question (the question block alone, as transfer_eval's baseline arm) and ignore (question block + the F0 ignore_question
requirement, as the eval prompt). T 1.0, top_p 1.0, --samples per item and arm, at most 64 tokens.

    PYTHONPATH=~/.cache/cotctl_stubs .venv/bin/python nocot_baseline.py --out ../results/nocot
"""
import argparse
import asyncio
import json
from pathlib import Path

import tinker

from cotswitch import extract, prompts
from cotswitch.data import DATASETS

import controls
import modelio
from transfer_eval import make_unit, shared_items

EMPTY = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"


def units():
    ctl = controls.CONTROLS["ignore_question"]
    astra = prompts.astra_sample()
    sets = [("transfer", it) for it in shared_items(36)] + [("astra", it) for ds in DATASETS for it in astra[("ignore_question", ds)]]
    out = []
    for src, it in sets:
        q = make_unit(src, None, it, None, None, True)
        ig = make_unit(src, ctl, it, "F0", "user", True)
        out += [dict(q, arm="question", src=src), dict(ig, arm="ignore", src=src)]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = await tinker.ServiceClient().create_sampling_client_async(base_model="openai/gpt-oss-120b")
    tok = client.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=64, temperature=1.0, top_p=1.0, seed=0, stop=modelio.GPT_OSS_STOP)
    sem = asyncio.Semaphore(32)

    async def one(u):
        rendered = tok.apply_chat_template([{"role": "user", "content": u["user"]}], tokenize=False, add_generation_prompt=True) + EMPTY
        ids = tok.encode(rendered, add_special_tokens=False)
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=args.samples, sampling_params=params)
        rows = []
        for s, seq in enumerate(res.sequences):
            text = tok.decode(seq.tokens, skip_special_tokens=True)
            ans = extract.parse_answer(text, text)
            rows.append(dict(src=u["src"], arm=u["arm"], dataset=u["dataset"], item_idx=u["item_idx"], sample=s, answer=u["answer"],
                             answer_pred=ans, correct=ans == u["answer"], text=text, n_output_tokens=len(seq.tokens)))
        return rows

    us = units()
    rows = [r for rs in await asyncio.gather(*[one(u) for u in us]) for r in rs]
    (args.out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for src in ("transfer", "astra"):
        for arm in ("question", "ignore"):
            g = [r for r in rows if r["src"] == src and r["arm"] == arm]
            print(f"{src:8s} {arm:8s} accuracy {sum(r['correct'] for r in g)}/{len(g)} = {sum(r['correct'] for r in g) / len(g):.3f}, "
                  f"unparsed {sum(r['answer_pred'] is None for r in g)}")


if __name__ == "__main__":
    asyncio.run(main())
