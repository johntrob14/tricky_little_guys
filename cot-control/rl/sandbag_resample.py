"""Resample one sandbagging question many times: sandbag.py's prompts, rendering and answer parser, at --temperature, --n samples per model and arm.

For single-question case studies (results/case_studies/<name>/). Rows are sandbag.py's plus `model` (the --models name) and `sample` (0..n-1), appended to
<out>/rows.jsonl; reruns skip (model, arm, sample) already on disk. gpt-oss (harmony) models only.

    python sandbag_resample.py --qid 1490 --variant wmdp.should --arms honest coached --n 64 --temperature 1.0 \\
        --models base mt1b-s150=tinker://... --out ../results/case_studies/mt1b_coached_modbus
"""

import argparse
import asyncio
import json
from pathlib import Path

import tinker

from cotswitch import extract

import modelio
from hintverb import BASE, SYSTEM, parse_letter, render
from sandbag import ALL_ARMS, AWARE, COACH, EVASION, prompts
from sandbag_names import VARIANTS


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qid", type=int, required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS, required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=tinker://sampler_weights, or a bare name for the base model")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    qfile, placement, hint = VARIANTS[args.variant]
    q = next(q for q in map(json.loads, open(Path(__file__).parent / "data" / qfile)) if q["qid"] == args.qid)
    models = dict(m.split("=", 1) if "=" in m else (m, None) for m in args.models)
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), out=str(args.out), models=models, base=BASE, questions=qfile, placement=placement, hint=hint, system=SYSTEM, aware=AWARE, evasion=EVASION, coach=COACH),
              open(args.out / "config.json", "w"), indent=1)
    path = args.out / "rows.jsonl"
    done = {(r["model"], r["arm"], r["sample"]) for r in map(json.loads, open(path))} if path.exists() else set()
    sc = tinker.ServiceClient()
    sem, lock, n_tok = asyncio.Semaphore(args.concurrency), asyncio.Lock(), [0, 0]

    async def run_model(name, model_path):
        client = await (sc.create_sampling_client_async(model_path=model_path) if model_path else sc.create_sampling_client_async(base_model=BASE))
        tok = client.get_tokenizer()
        assert modelio.probe_harmony(tok)
        params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, stop=modelio.policy_stop(tok, True))

        async def one(arm, i):
            system, user = prompts(q, arm, hint, placement)
            ids = render(tok, user, args.effort, system)
            async with sem:
                res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
            s = res.sequences[0]
            finish = "length" if s.stop_reason == "length" else "stop"
            split = extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), True, "", tok.eos_token), None, finish)
            row = dict(model=name, sample=i, qid=q["qid"], arm=arm, correct=q["correct"], hint_wrong=q["hint_wrong"], answer=parse_letter(split["response"]), finish=finish,
                       cot_tokens=len(tok.encode(split["reasoning"], add_special_tokens=False)), resp_tokens=len(tok.encode(split["response"], add_special_tokens=False)),
                       reasoning=split["reasoning"], response=split["response"], user=user, system=system)
            async with lock:
                n_tok[0] += len(s.tokens)
                n_tok[1] += len(ids)
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")

        await asyncio.gather(*[one(a, i) for a in args.arms for i in range(args.n) if (name, a, i) not in done])

    await asyncio.gather(*[run_model(n, p) for n, p in models.items()])
    price = modelio.PRICES[BASE]
    print(f"sampled {n_tok[0]} tokens, prefill {n_tok[1]} tokens, about ${n_tok[0] * price['sample'] / 1e6 + n_tok[1] * price['prefill'] / 1e6:.2f} -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
