"""Final context pool for turn1_grpo.py from decisive_ctx.py's pool and a base-model resample_ctx.py run over it (G samples per context, same contexts).

    python pool_finalize.py POOL.json SIDES.json BASE_SAMPLES.jsonl --out-dir results/qwenpool1 --model Qwen/Qwen3.8-27B --max-prompt-tokens 20000

Filters, each counted in the funnel: prompt tokens over --max-prompt-tokens (Tinker bills train tokens per sample including the prompt; the trainer's own limit is 32768),
base median CoT words over --max-base-words (a long base CoT widens the length band and the rambling it allows), and --drop-all-quiet (every base sample scored <= 2 by the
monitor: nothing to learn on covertness there). Writes contexts.json, sides.json (with base_words and prompt_tokens added), base_words.json and prints the arm mix and a
cost estimate for --steps x --ctx-per-step x --g with the trainer's price table.
"""
import argparse, asyncio, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai.tool import bash, text_editor
from inspect_ai.model._call_tools import prepare_tools
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

import run
from resample_ctx import load_context
from turn1_grpo import PRICES, render

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pool", type=Path)
ap.add_argument("sides", type=Path)
ap.add_argument("base_samples", type=Path)
ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--model", default="Qwen/Qwen3.8-27B", choices=list(PRICES))
ap.add_argument("--max-prompt-tokens", type=int, default=20000)
ap.add_argument("--max-base-words", type=int, default=400)
ap.add_argument("--drop-all-quiet", action="store_true")
ap.add_argument("--steps", type=int, default=15)
ap.add_argument("--ctx-per-step", type=int, default=32)
ap.add_argument("--g", type=int, default=8)
ap.add_argument("--tutor-usd-per-rating", type=float, default=0.0036, help="luna at max effort: 56 ratings cost $0.20 in results/ws8/tutor_check")
a = ap.parse_args()
pool, sides = json.loads(a.pool.read_text()), json.loads(a.sides.read_text())
key = lambda c: f"{c['run']}|{c['sample']}|{c['epoch']}|{c['turn']}"
words, scores, out_tok = defaultdict(list), defaultdict(list), defaultdict(list)
for r in map(json.loads, a.base_samples.open()):
    k = f"{r['run']}|{r['sample']}|{r['epoch']}|{r['turn']}"
    words[k].append(len(re.findall(r"[A-Za-z]+", r["reasoning"])))
    out_tok[k].append(r["out_tok"])
    if r["score"] is not None:
        scores[k].append(r["score"])


async def prompt_tokens():
    tool_infos = (await prepare_tools([bash(timeout=run.TOOL_TIMEOUT), text_editor(timeout=run.TOOL_TIMEOUT), run.submit()])).base_tools
    renderer = get_renderer(run.RENDERERS[a.model], get_tokenizer(a.model))
    return {key(c): len(render(renderer, load_context(c["run"], c["sample"], c["epoch"], c["turn"])[0], tool_infos)) for c in pool}


ptok = asyncio.run(prompt_tokens())
funnel, keep = Counter(pool=len(pool)), []
for c in pool:
    k = key(c)
    why = ("no base samples" if k not in words else f"prompt over {a.max_prompt_tokens} tokens" if ptok[k] > a.max_prompt_tokens else f"base median CoT over {a.max_base_words} words" if statistics.median(words[k]) > a.max_base_words
           else "all base samples quiet (<= 2)" if a.drop_all_quiet and scores[k] and max(scores[k]) <= 2 else None)
    if why:
        funnel[f"dropped: {why}"] += 1
    else:
        keep.append(c)
funnel["kept"] = len(keep)
base_words = {key(c): statistics.median(words[key(c)]) for c in keep}
a.out_dir.mkdir(parents=True, exist_ok=True)
(a.out_dir / "contexts.json").write_text(json.dumps(keep, indent=1))
(a.out_dir / "sides.json").write_text(json.dumps({key(c): dict(sides[key(c)], prompt_tokens=ptok[key(c)], base_median_words=base_words[key(c)]) for c in keep}, indent=1))
(a.out_dir / "base_words.json").write_text(json.dumps(base_words, indent=1))
print("funnel: " + "; ".join(f"{k} {v}" for k, v in funnel.items()))
print("kept by arm: " + ", ".join(f"{k} {n}/{len(keep)}" for k, n in sorted(Counter(c["arm"] for c in keep).items())))
print("kept by tier: " + ", ".join(f"{k} {n}/{len(keep)}" for k, n in sorted(Counter(c["run"].removesuffix("-b").rsplit("-", 1)[-1] for c in keep).items())))
pt = [ptok[key(c)] for c in keep]
ot = [statistics.mean(out_tok[key(c)]) for c in keep]
print(f"kept prompt tokens: median {statistics.median(pt):.0f}, mean {statistics.mean(pt):.0f}, max {max(pt)}; base mean output tokens {statistics.mean(ot):.0f}; base median CoT words {statistics.median(base_words.values()):.0f} (max {max(base_words.values()):.0f})")
P = PRICES[a.model]
n = a.ctx_per_step * a.g
per_step = dict(sampling=(a.ctx_per_step * statistics.mean(pt) * (P["prefill"] + (a.g - 1) * P["cached"]) + n * statistics.mean(ot) * P["sample"]) / 1e6,
                training=n * (statistics.mean(pt) + statistics.mean(ot)) * P["train"] / 1e6, tutor=n * a.tutor_usd_per_rating)
tot = sum(per_step.values())
print(f"estimated cost per step: " + ", ".join(f"{k} ${v:.1f}" for k, v in per_step.items()) + f"; per step ${tot:.1f}; {a.steps} steps ${tot * a.steps:.0f} (upper bound: equal-reward groups are dropped from training)")
