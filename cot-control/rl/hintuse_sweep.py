"""Sweep for hintuse.py target items: re-measure base gpt-oss-120b accuracy, with and without a CoT, on the GPQA items whose 8-sample CoT accuracy in
results/difficulty/gpqafull_base.jsonl (item_difficulty.py) is in --cot-range, no-CoT accuracy is at most --max-nocot of 24 and median CoT length at most
--max-tok. Prompts, sampling and parsing are hintuse.py's none and nocot arms (temperature 1, Tinker), --n samples each.

Writes <out>/rows.jsonl (hintuse.py's row format plus `item`, so failed CoTs can be read per item; resumable) and <out>/candidates.jsonl, one row per
item: CoT and no-CoT accuracy and answer letters over the new samples, median CoT tokens, the 8-sample screen numbers, and from gpqa_extended.csv the
subdomain, Diamond membership, expert validator accuracy (1.0 = both validators answered correctly), whether every option is a bare number (the model can
then confuse "the answer is 2" with option B), the question, options, answer and the writer's explanation.

    python hintuse_sweep.py --out ../results/hintuse_sweep
"""

import argparse
import asyncio
import csv
import json
import re
import statistics
from pathlib import Path

import tinker

from cotswitch import extract
from cotswitch.data import load_items

import modelio
from hintuse import prompt
from item_difficulty import RESULTS

NUMERIC = r"[~\s$\\(]*[-+]?[\d.,]+(\s*(x|\\times|×|e|\^)\s*10\^?\{?[-\d]+\}?)?\s*\$?\)?\s*[a-zA-Zμ/°%\s\^\d\-\*·.]{0,12}"
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()[:200]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cot-range", type=int, nargs=2, default=[1, 6], help="8-sample CoT correct count, inclusive")
    ap.add_argument("--max-nocot", type=int, default=8, help="of 24 no-CoT samples")
    ap.add_argument("--max-tok", type=float, default=6000)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gens", type=Path, default=RESULTS / "transfer/gpqafull-base/gens.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    screen = {r["item_idx"]: r for r in map(json.loads, open(RESULTS / "difficulty/gpqafull_base.jsonl"))}
    sel = sorted(i for i, r in screen.items() if args.cot_range[0] <= r["cot_correct"] <= args.cot_range[1] and r["nocot_correct"] <= args.max_nocot and r["cot_tokens_median"] <= args.max_tok)
    base = {r["item_idx"]: r for r in map(json.loads, open(args.gens)) if r["type"] == "baseline" and r["dataset"] == "gpqa" and r["item_idx"] in sel}
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), gens=str(args.gens), out=str(args.out), items=sel, temperature=1.0, top_p=1.0), open(args.out / "config.json", "w"), indent=1)
    path = args.out / "rows.jsonl"
    done = {(r["item"], r["arm"], r["sample"]) for r in map(json.loads, open(path))} if path.exists() else set()
    client = await tinker.ServiceClient().create_sampling_client_async(base_model="openai/gpt-oss-120b")
    tok = client.get_tokenizer()
    price, spent, sem = modelio.PRICES["openai/gpt-oss-120b"], [0.0], asyncio.Semaphore(16)

    async def run(item, arm):
        todo = [i for i in range(args.n) if (item, arm, i) not in done]
        if not todo:
            return
        ids = tok.encode(prompt(base[item]["rendered_prompt"], "", arm), add_special_tokens=False)
        params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8 if arm == "nocot" else args.max_tokens, stop=modelio.policy_stop(tok, True))
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=len(todo), sampling_params=params)
        with open(path, "a") as f:
            for i, s in zip(todo, res.sequences):
                finish = "length" if s.stop_reason == "length" and arm == "none" else "stop"
                split = dict(reasoning="", response="ANSWER:" + tok.decode(s.tokens, skip_special_tokens=True), content="") if arm == "nocot" else \
                    extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), True, "", tok.eos_token), None, finish)
                answer = extract.parse_answer(split["response"], split["content"])
                f.write(json.dumps(dict(item=item, model="base", arm=arm, sample=i, answer=answer, correct=base[item]["answer"], ok=answer == base[item]["answer"], finish=finish,
                                        cot_tokens=len(tok.encode(split["reasoning"], add_special_tokens=False)), reasoning=split["reasoning"], response=split["response"])) + "\n")
                spent[0] += len(ids) * price["prefill"] / 1e6 + len(s.tokens) * price["sample"] / 1e6

    await asyncio.gather(*[run(i, a) for i in sel for a in ["none", "nocot"]])
    rows = [json.loads(l) for l in open(path)]
    ext = {norm(r["Question"]): r for r in csv.DictReader(open(Path(__file__).parent / "data/gpqa/dataset/gpqa_extended.csv"))}
    diamond = {norm(r["Question"]) for r in csv.DictReader(open(Path(__file__).parent / "data/gpqa/dataset/gpqa_diamond.csv"))}
    items = {it.item_idx: it for it in load_items("gpqa")}
    with open(args.out / "candidates.jsonl", "w") as f:
        for i in sel:
            it, e = items[i], ext[norm(items[i].question)]
            cot, nocot = [r for r in rows if r["item"] == i and r["arm"] == "none"], [r for r in rows if r["item"] == i and r["arm"] == "nocot"]
            letters = lambda rs: {x: sum(r["answer"] == x for r in rs) for x in ["A", "B", "C", "D", None]}
            f.write(json.dumps(dict(item=i, cot_correct=sum(r["ok"] for r in cot), cot_n=len(cot), nocot_correct=sum(r["ok"] for r in nocot), nocot_n=len(nocot),
                                    cot_letters=letters(cot), nocot_letters=letters(nocot), cot_tokens_median=statistics.median(r["cot_tokens"] for r in cot), cot_cut=sum(r["finish"] == "length" for r in cot),
                                    screen_cot=f"{screen[i]['cot_correct']}/8", screen_nocot=f"{screen[i]['nocot_correct']}/24", subdomain=e["Subdomain"], diamond=norm(it.question) in diamond,
                                    expert_acc=float(e["Expert Validator Accuracy"]), numeric_options=all(re.fullmatch(NUMERIC, o.strip()) is not None for o in it.options),
                                    question=it.question, options=it.options, answer=it.answer_letter, explanation=e["Explanation"])) + "\n")
    print(f"{len(sel)} items x {args.n} CoT and {args.n} no-CoT samples, about ${spent[0]:.2f} -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
