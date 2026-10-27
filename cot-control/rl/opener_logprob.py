"""Teacher-forced log-probability that gpt-oss opens its analysis channel with a given string, unprompted.

Prompts: the no-prefill transfer-eval prompts stored in results/transfer/gptoss-base (the 108 no-requirement items plus --per-type items of each of
the 21 types, F0, user turn). For each prompt, the string's tokens are scored right after <|channel|>analysis<|message|>. Reports, per model, the
mean over prompts of the summed log-prob of the string, of its first token, and of the rest given the first.

    python opener_logprob.py --name base
    python opener_logprob.py --name flamingo1-s20 --model-path tinker://.../cotctl-flamingo1-step20
"""
import argparse
import asyncio
import json
import statistics as st
from pathlib import Path

import tinker

RESULTS = Path(__file__).resolve().parent.parent / "results"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--string", default="<|FLAMINGO_BURRITO|>")
    ap.add_argument("--per-type", type=int, default=10)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(RESULTS / "transfer/gptoss-base/gens.jsonl")]
    prompts = [r["rendered_prompt"] for r in rows if r["type"] == "baseline"]
    by_type = {}
    for r in rows:
        if r["type"] != "baseline" and r["frame"] == "F0" and r["placement"] == "user" and len(by_type.setdefault(r["type"], [])) < args.per_type:
            by_type[r["type"]].append(r["rendered_prompt"])
    prompts += [p for ps in by_type.values() for p in ps]
    svc = tinker.ServiceClient()
    client = await (svc.create_sampling_client_async(model_path=args.model_path) if args.model_path else svc.create_sampling_client_async(base_model="openai/gpt-oss-120b"))
    tok = client.get_tokenizer()
    s_ids = tok.encode(args.string, add_special_tokens=False)
    ids = [tok.encode(p + "<|channel|>analysis<|message|>", add_special_tokens=False) for p in prompts]
    lps = await asyncio.gather(*[client.compute_logprobs_async(tinker.ModelInput.from_ints(i + s_ids)) for i in ids])
    per = [[float(x) for x in lp[-len(s_ids):]] for lp in lps]
    out = dict(name=args.name, model_path=args.model_path, string=args.string, n_tokens=len(s_ids), n_prompts=len(per),
               total=st.mean(sum(p) for p in per), first=st.mean(p[0] for p in per), rest=st.mean(sum(p[1:]) for p in per),
               per_token=[st.mean(p[i] for p in per) for i in range(len(s_ids))])
    path = RESULTS / "opener_logprob" / f"{args.name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_token"}))


if __name__ == "__main__":
    asyncio.run(main())
