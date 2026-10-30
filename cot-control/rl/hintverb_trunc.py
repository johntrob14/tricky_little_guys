"""CoT truncation curve and no-CoT gap on the hint-verbalization rows (hintverb.py output, gpt-oss harmony models only). Each row's own analysis
channel is cut to a fraction of its tokens and prefilled ('<|channel|>analysis<|message|>' + kept tokens + '<|end|><|start|>assistant<|channel|>final
<|message|>' + 'Answer:'), then a few tokens are sampled at temperature 0, so the letter is read off the partial CoT (without the 'Answer:' prefill
the model resumes reasoning inside the final channel). Fraction 0 is the no-CoT baseline (empty analysis channel); fraction 1 is the full CoT, a check
that the forced answer reproduces the original row's. Reads <dir>/<arm>/rows.jsonl (the no-line arms) and writes <out>/<arm>/rows.jsonl (qid,
hint_type, frac, kept, cot_tokens, answer, response), resumable. hintverb_trunc_summary.py scores it.

    python hintverb_trunc.py --arm base --out ../results/hintverb_trunc
    python hintverb_trunc.py --arm grpo3-s100 --model-path tinker://... --out ../results/hintverb_trunc
"""

import argparse
import asyncio
import json
from pathlib import Path

import tinker

import modelio
from hintverb import BASE, HINTS, SYSTEM, parse_letter, render

OPEN = "<|channel|>analysis<|message|>"
FINAL_HEAD = "<|end|><|start|>assistant<|channel|>final<|message|>"
PREFILL = "Answer:"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="hintverb arm whose rows are truncated, e.g. base or grpo3-s100")
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "../results/hintverb")
    ap.add_argument("--model-path", help="tinker:// sampler weights of the arm's model; base gpt-oss-120b when omitted")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1])
    ap.add_argument("--hints", nargs="+", default=["none"] + list(HINTS), choices=["none"] + list(HINTS))
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--n", type=int, help="only the first N questions (smoke tests)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    cfg = json.load(open(args.dir / args.arm / "config.json"))
    assert cfg["system"] == SYSTEM and cfg.get("suffix") is None, "truncation runs on the no-line arms"
    rows = [r for r in map(json.loads, open(args.dir / args.arm / "rows.jsonl")) if r["hint_type"] in args.hints]
    if args.n:
        qids = sorted({r["qid"] for r in rows})[:args.n]
        rows = [r for r in rows if r["qid"] in qids]
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), dir=str(args.dir), out=str(args.out), source_config=cfg, open=OPEN, final_head=FINAL_HEAD, prefill=PREFILL, temperature=0.0), open(out / "config.json", "w"), indent=1)
    path = out / "rows.jsonl"
    done = {(r["qid"], r["hint_type"], r["frac"]) for r in map(json.loads, open(path))} if path.exists() else set()
    sc = tinker.ServiceClient()
    client = await (sc.create_sampling_client_async(model_path=args.model_path) if args.model_path else sc.create_sampling_client_async(base_model=BASE))
    tok = client.get_tokenizer()
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    open_ids, head_ids = enc(OPEN), enc(FINAL_HEAD + PREFILL)
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.0, stop=modelio.GPT_OSS_STOP)
    sem, lock, n_tok = asyncio.Semaphore(args.concurrency), asyncio.Lock(), [0, 0, 0]  # sampled, prefill, rows

    async def one(r, frac):
        if (r["qid"], r["hint_type"], frac) in done:
            return
        cot_ids = enc(r["reasoning"])
        kept = round(frac * len(cot_ids))
        ids = render(tok, r["user"], args.effort) + open_ids + cot_ids[:kept] + head_ids
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
        s = res.sequences[0]
        response = PREFILL + tok.decode(s.tokens, skip_special_tokens=True)
        row = dict(qid=r["qid"], hint_type=r["hint_type"], frac=frac, kept=kept, cot_tokens=len(cot_ids), answer=parse_letter(response), finish="length" if s.stop_reason == "length" else "stop", response=response)
        async with lock:
            n_tok[0] += len(s.tokens)
            n_tok[1] += len(ids)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_tok[2] += 1
            if n_tok[2] % 500 == 0:
                print(f"{n_tok[2]} done, sampled {n_tok[0]} tokens, prefill {n_tok[1]} tokens", flush=True)

    await asyncio.gather(*[one(r, f) for r in rows for f in args.fracs])
    price = modelio.PRICES[BASE]
    print(f"{len(rows)} rows x {len(args.fracs)} fractions; sampled {n_tok[0]} tokens, prefill {n_tok[1]} tokens, about ${n_tok[0] * price['sample'] / 1e6 + n_tok[1] * price['prefill'] / 1e6:.2f} -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
