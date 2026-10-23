"""Drop the contexts of a decisive_ctx.py pool whose rendered prompt is over a token cap, before anything is sampled over them (Qwen bills training per prompt token, and
the trainer's --max-prompt-tokens would otherwise drop them only after the paid pre-screen and length-reference resamples). Writes the kept contexts and their side configs,
with prompt_tokens added, and prints the funnel by arm.

    .venv/bin/python pool_cap.py POOL.json SIDES.json --max-prompt-tokens 16000 --out results/ws1tutor/pool/contexts_cap16.json --side-out results/ws1tutor/pool/sides_cap16.json
"""
import argparse, asyncio, json
from collections import Counter
from pathlib import Path

from inspect_ai.model._call_tools import prepare_tools
from inspect_ai.tool import bash, text_editor
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

import run
from resample_ctx import load_context
from turn1_grpo import PRICES, render

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pool", type=Path)
ap.add_argument("sides", type=Path)
ap.add_argument("--model", default="Qwen/Qwen3.8-27B", choices=list(PRICES))
ap.add_argument("--max-prompt-tokens", type=int, default=16000)
ap.add_argument("--keep-arms", nargs="*", default=[], help="arms kept up to --keep-max-tokens instead of the cap (e.g. leak retry: few contexts, the most informative ones)")
ap.add_argument("--keep-max-tokens", type=int, default=28000, help="cap for --keep-arms; the trainer's per-datum limit is 32768 including the response")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--side-out", type=Path, required=True)
a = ap.parse_args()
pool, sides = json.loads(a.pool.read_text()), json.loads(a.sides.read_text())
key = lambda c: f"{c['run']}|{c['sample']}|{c['epoch']}|{c['turn']}"


async def prompt_tokens():
    tool_infos = (await prepare_tools([bash(timeout=run.TOOL_TIMEOUT), text_editor(timeout=run.TOOL_TIMEOUT), run.submit()])).base_tools
    renderer = get_renderer(run.RENDERERS[a.model], get_tokenizer(a.model))
    return {key(c): len(render(renderer, load_context(c["run"], c["sample"], c["epoch"], c["turn"])[0], tool_infos)) for c in pool}


ptok = asyncio.run(prompt_tokens())
cap = lambda c: a.keep_max_tokens if c["arm"] in a.keep_arms else a.max_prompt_tokens
keep = [c for c in pool if ptok[key(c)] <= cap(c)]
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(keep, indent=1))
a.side_out.write_text(json.dumps({key(c): dict(sides[key(c)], prompt_tokens=ptok[key(c)]) for c in keep}, indent=1))
drop = Counter(c["arm"] for c in pool if ptok[key(c)] > cap(c))
print(f"kept {len(keep)}/{len(pool)} contexts under {a.max_prompt_tokens} prompt tokens; dropped by arm: {dict(drop)}; kept by arm: {dict(Counter(c['arm'] for c in keep))}")
pt = sorted(ptok[key(c)] for c in keep)
if pt:
    print(f"kept prompt tokens: median {pt[len(pt) // 2]}, mean {sum(pt) / len(pt):.0f}, max {pt[-1]}")
