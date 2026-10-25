"""432 new CoT-Control items (48 per constraint: 16 per dataset), Astra items excluded, seed 20261025; astra-protocol prompts and grader_args.
Run from cot-control/grading:  uv run python ../results/outctl/build_items.py"""
import json, random
from cotswitch import prompts, data
from cotswitch import constraints as C
astra = {(it.dataset, str(it.item_idx), c) for (c, ds), its in prompts.astra_sample().items() for it in its}
astra_items = {(d, i) for d, i, _ in astra}
rng = random.Random(20261025); out = []
for c in C.DEFAULT_CONSTRAINTS:
    for ds in prompts.DATASETS:
        pool = [it for it in prompts._eligible(data.load_items(ds), c) if (it.dataset, str(it.item_idx)) not in astra_items]
        for it in rng.sample(pool, 16):
            spec = prompts.build_prompt("astra", it, c)
            assert spec.system is None
            out.append(dict(key=f"new|{it.dataset}|{it.item_idx}|{c}", dataset=it.dataset, item_idx=str(it.item_idx), constraint=c, user_prompt=spec.user.replace("reasoning stage", "analysis channel"),
                            grader_args=spec.grader_args, answer_gold=it.answer_letter))
json.dump(out, open("../results/outctl/new_items.json", "w"))
import collections; print(len(out), collections.Counter(x["constraint"] for x in out))
