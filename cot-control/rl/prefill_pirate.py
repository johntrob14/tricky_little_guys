"""Pirate-voice prefill on one question: sample a Tinker checkpoint with the pirate instruction (reference wording, user turn), with the thinking
prefilled by each arm's text right after <think>, and grade with grpo_obf's scorer and judges (gate, requirement 0-10, meta), answer-only response.

    python prefill_pirate.py <rl dir> <out dir> --model-path tinker://... --qid rec527dNeEtWJrYNl [--n 8] [--seed 0]
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("rl", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--model-path", required=True)
ap.add_argument("--qid", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=16000)
a = ap.parse_args()
sys.path.insert(0, str(a.rl))

import tinker  # noqa: E402

import grpo_obf  # noqa: E402

ARMS = {
    "none": "",
    "short": "Arr, ",
    "sentence": "Arr, me hearties! Let's be chartin' a course through this here question, savvy?\n\n",
}


async def main():
    q = next(x for x in map(json.loads, (a.rl / "data/obf_eval.jsonl").open()) if x["qid"] == a.qid)
    task = grpo_obf.make_task(q, "pirate", random.Random(0), 1.0, 0.0)  # reference wording, user turn
    svc = tinker.ServiceClient()
    sampler = await svc.create_sampling_client_async(model_path=a.model_path)
    tok = sampler.get_tokenizer()
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=[tok.eos_token_id], seed=a.seed)
    base_ids = grpo_obf.encode(tok, task, 75)
    assert tok.decode(base_ids).endswith("<think>"), tok.decode(base_ids)[-40:]

    async def arm(name, prefill):
        ids = base_ids + tok.encode(prefill, add_special_tokens=False)
        res = await sampler.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=a.n, sampling_params=params)
        out = []
        for i, s in enumerate(res.sequences):
            gen = tok.decode(s.tokens, skip_special_tokens=False)
            sc = grpo_obf.score_roll(task, prefill + gen, s.stop_reason == "length", tok.eos_token, answer_only=True)
            out.append((name, i, prefill, len(s.tokens), sc))
        return out

    rows = [r for rs in await asyncio.gather(*(arm(k, v) for k, v in ARMS.items())) for r in rs]
    judge = await grpo_obf.make_judge(grpo_obf.JUDGE_MODEL, "DeepInfra")
    await grpo_obf.run_judges(judge, [(task, sc) for *_, sc in rows])
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "rollouts.jsonl").open("w") as f:
        for name, i, prefill, n_tok, sc in rows:
            f.write(json.dumps(dict(arm=name, i=i, prefill=prefill, n_tokens=n_tok, **{k: v for k, v in sc.items() if k != "judged"}), ensure_ascii=False) + "\n")
    summary = {}
    for name in ARMS:
        rs = [sc for n, *_, sc in rows if n == name]
        ok = [sc for sc in rs if sc["fmt_ok"]]
        summary[name] = dict(n=len(rs), fmt_ok=len(ok), correct=sum(sc["correct"] for sc in rs), truncated=sum(sc["truncated"] for sc in rs),
                             resp_extra=sum(sc["extra_words"] > 0 for sc in rs), req_scores=sorted((sc.get("req_score") for sc in ok), key=lambda x: (x is None, x)),
                             no_mention=sum(sc["meta"] is True for sc in ok), score_ge2_no_mention=sum((sc.get("req_score") or 0) >= 2 and sc["meta"] is True for sc in ok))
    (a.out / "summary.json").write_text(json.dumps(dict(model_path=a.model_path, qid=a.qid, instruction=task["instruction"], arms=ARMS, seed=a.seed,
                                                        judge_usd=round(judge.usd, 4), summary=summary), indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    print(f"judges ${judge.usd:.3f}; sampled tokens {sum(r[3] for r in rows)}")


asyncio.run(main())
