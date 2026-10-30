"""hintuse.py over a set of GPQA items: the same prompts, sampling and parsing (temperature 1, Tinker, harmony effort medium), one directory for all items.

Items default to the sweep's main-criteria set (results/hintuse_sweep/candidates.jsonl: fresh base CoT accuracy 6-17/32, no-CoT accuracy at most 4/32, expert
validator accuracy 1.0). --dataset dapo takes DAPO problems instead: items are 8-character qid prefixes from results/hintuse_dapo_sweep/candidates.jsonl, the
prompt is hintuse_dapo_sweep.py's (math_data.question_block in the harmony template at effort medium), answers are integers parsed by math_data.parse_answer, and
HINTS is keyed by the prefix. Arms are hintuse.ARMS; the hint arms need a hint per item (--hints item=name, names from hintuse.HINTS), and items without one get
only the none and nocot arms. Rows are hintuse.py's plus `item`, appended to <out>/rows.jsonl (resumable); <out>/config.json has the per-item prompts keyed
"<item>|<arm>", and its models, arms, hints and prompts accumulate over reruns into the same directory. hintuse_html.py shows the directory as one run with means over items.

    python hintuse_set.py --models base mt1b-s150=tinker://... --arms none nocot --n 64 --out ../results/case_studies/hintuse_hardset
    python hintuse_set.py --dataset dapo --items f17baca8 2bb6f9ac --hints f17baca8=worked 2bb6f9ac=worked --models base --n 64 --out ../results/case_studies/hintuse_dapo
"""

import argparse
import asyncio
import json
from pathlib import Path

import tinker

from cotswitch import extract

import math_data
import modelio
from hintuse import ARMS, HIDE, HINTS, prompt
from item_difficulty import RESULTS


def main_criteria() -> list[int]:
    c = [json.loads(l) for l in open(RESULTS / "hintuse_sweep/candidates.jsonl")]
    return sorted(r["item"] for r in c if 6 <= r["cot_correct"] <= 17 and r["nocot_correct"] <= 4 and r["expert_acc"] == 1.0)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gpqa", choices=["gpqa", "dapo"])
    ap.add_argument("--items", nargs="+", help="default: the sweep's main-criteria items (gpqa); 8-character qid prefixes (dapo)")
    ap.add_argument("--hints", nargs="*", default=[], help="item=hint_name")
    ap.add_argument("--models", nargs="+", required=True, help="name=tinker://sampler_weights, or a bare name for base gpt-oss-120b")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gens", type=Path, default=RESULTS / "transfer/gpqafull-base/gens.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = int if args.dataset == "gpqa" else str
    items = [key(i) for i in args.items] if args.items else main_criteria()
    hints = {key(i): HINTS[key(i)][h] for i, h in (x.split("=") for x in args.hints)}
    models = dict(m.split("=", 1) if "=" in m else (m, None) for m in args.models)
    svc = tinker.ServiceClient()
    clients = {m: await (svc.create_sampling_client_async(model_path=p) if p else svc.create_sampling_client_async(base_model="openai/gpt-oss-120b")) for m, p in models.items()}
    if args.dataset == "gpqa":
        base = {r["item_idx"]: dict(prompt=r["rendered_prompt"], answer=r["answer"], qblock=r["qblock"]) for r in map(json.loads, open(args.gens)) if r["type"] == "baseline" and r["dataset"] == "gpqa" and r["item_idx"] in items}
        parse = lambda split: extract.parse_answer(split["response"], split["content"])
    else:
        tok = next(iter(clients.values())).get_tokenizer()
        base = {r["item"][:8]: dict(prompt=tok.apply_chat_template([{"role": "user", "content": math_data.question_block(r)}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium"), answer=r["answer"], qblock=math_data.question_block(r))
                for r in map(json.loads, open(RESULTS / "hintuse_dapo_sweep/candidates.jsonl")) if r["item"][:8] in items}
        parse = lambda split: math_data.parse_answer(split["response"])
    arms = {i: [a for a in args.arms if a in ("none", "nocot", "nocot_bare") or i in hints] for i in items}
    prompts = {f"{i}|{a}": prompt(base[i]["prompt"], hints.get(i, ""), a) for i in items for a in arms[i]}
    args.out.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(args.out / "config.json")) if (args.out / "config.json").exists() else dict(models={}, arms=[], hints={}, hint_names={}, prompts={})
    json.dump(dict(dataset=args.dataset, items=items, answers={i: base[i]["answer"] for i in items}, qblocks={i: base[i]["qblock"] for i in items}, hints=prev["hints"] | {str(i): h for i, h in hints.items()}, hint_names=prev["hint_names"] | dict(x.split("=") for x in args.hints), hide=HIDE,
                   models=prev["models"] | models, arms=[a for a in ARMS if a in prev["arms"] or a in args.arms], n=args.n, max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, prompts=prev["prompts"] | prompts), open(args.out / "config.json", "w"), indent=1)
    path = args.out / "rows.jsonl"
    done = {(r["model"], r["item"], r["arm"], r["sample"]) for r in map(json.loads, open(path))} if path.exists() else set()
    price, spent, sem = modelio.PRICES["openai/gpt-oss-120b"], [0.0], asyncio.Semaphore(16)

    async def run(name, item, arm):
        todo = [i for i in range(args.n) if (name, item, arm, i) not in done]
        if not todo:
            return
        client = clients[name]
        tok = client.get_tokenizer()
        ids = tok.encode(prompts[f"{item}|{arm}"], add_special_tokens=False)
        params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1200 if arm.endswith("_bare") else 8 if arm.startswith("nocot") else args.max_tokens, stop=modelio.policy_stop(tok, True))
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=len(todo), sampling_params=params)
        with open(path, "a") as f:
            for i, s in zip(todo, res.sequences):
                finish = "length" if s.stop_reason == "length" and (arm.endswith("_bare") or not arm.startswith("nocot")) else "stop"
                split = dict(reasoning="", response=("" if arm.endswith("_bare") else "ANSWER:") + tok.decode(s.tokens, skip_special_tokens=True), content="") if arm.startswith("nocot") else \
                    extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), True, "", tok.eos_token), None, finish)
                answer = parse(split)
                f.write(json.dumps(dict(model=name, item=item, arm=arm, sample=i, answer=answer, correct=base[item]["answer"], ok=answer == base[item]["answer"], finish=finish,
                                        cot_tokens=len(tok.encode(split["reasoning"], add_special_tokens=False)), reasoning=split["reasoning"], response=split["response"])) + "\n")
                spent[0] += len(ids) * price["prefill"] / 1e6 + len(s.tokens) * price["sample"] / 1e6

    await asyncio.gather(*[run(m, i, a) for m in models for i in items for a in arms[i]])
    print(f"{len(items)} items, about ${spent[0]:.2f} -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
