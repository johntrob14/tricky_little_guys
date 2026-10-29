"""No-CoT accuracy on the sandbagging question sets (the 150 WMDP and 198 GPQA diamond questions of sandbag.py), base gpt-oss-120b or a Tinker checkpoint.

The honest-arm prompt (hintverb's system prompt, the unperturbed question, harmony at --effort) is followed by an empty analysis message and the final
channel opened with the standard prefill "Answer:" (modelio.ANSWER), so the first sampled token is the letter. Temperature 0, one sample, at most 2 tokens;
answer = the letter right after the prefill (hintverb.parse_answer; None otherwise). Writes <out>/<name>/rows.jsonl (`response` = prefill + sampled text).

    python sandbag_nocot.py --name base --out ../results/sandbag_nocot
    python sandbag_nocot.py --name mt1b-s150 --model-path tinker://... --out ../results/sandbag_nocot
"""
import argparse
import asyncio
import json
from pathlib import Path

import tinker

import modelio
from hintverb import BASE, SYSTEM, parse_answer

EMPTY = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"  # an empty analysis channel, then the final channel header
PREFILL = modelio.ANSWER
SETS = {"wmdp": "data/wmdp_sandbagging.jsonl", "gpqa": "data/gpqa_sandbagging.jsonl"}


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-path", help="tinker:// sampler weights; the base model itself when omitted")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), out=str(args.out), base=BASE, system=SYSTEM, empty=EMPTY, prefill=PREFILL, sets=SETS, temperature=0.0, max_tokens=2, answer_prefill=PREFILL), open(out / "config.json", "w"), indent=1)
    sc = tinker.ServiceClient()
    client = await (sc.create_sampling_client_async(model_path=args.model_path) if args.model_path else sc.create_sampling_client_async(base_model=BASE))
    tok = client.get_tokenizer()
    assert modelio.probe_harmony(tok)
    params = tinker.SamplingParams(max_tokens=2, temperature=0.0, stop=modelio.policy_stop(tok, True))
    sem = asyncio.Semaphore(64)

    async def one(ds, q):
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["honest"]}], tokenize=False, add_generation_prompt=True, reasoning_effort=args.effort)
        assert text.endswith("<|start|>assistant") and f"Reasoning: {args.effort}" in text
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(tok.encode(text + EMPTY + PREFILL, add_special_tokens=False)), num_samples=1, sampling_params=params)
        gen = tok.decode(res.sequences[0].tokens, skip_special_tokens=True)
        return dict(dataset=ds, qid=q["qid"], correct=q["correct"], answer=parse_answer(PREFILL + gen, True), prefilled=True, response=PREFILL + gen, text=gen)

    qs = [(ds, q) for ds, f in SETS.items() for q in map(json.loads, open(Path(__file__).parent / f))]
    rows = await asyncio.gather(*[one(ds, q) for ds, q in qs])
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for ds in SETS:
        g = [r for r in rows if r["dataset"] == ds]
        print(f"{args.name} {ds}: correct {sum(r['answer'] == r['correct'] for r in g)}/{len(g)}, unparsed {sum(r['answer'] is None for r in g)}/{len(g)}")


if __name__ == "__main__":
    asyncio.run(main())
