"""Per-context median CoT word count of a resample_ctx.py run, as the length-penalty reference for turn1_grpo.py --base-words.

    python base_words.py results/ws8/base_g8/samples.jsonl --out results/ws8/base_words.json

Keys are "run|sample|epoch|turn" (turn1_grpo.ctx_key); words are counted the way turn1_grpo counts them, as runs of letters in the reasoning.
"""

import argparse, json, re, statistics
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("samples", type=Path)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
words = defaultdict(list)
for line in a.samples.open():
    r = json.loads(line)
    words[f"{r['run']}|{r['sample']}|{r['epoch']}|{r['turn']}"].append(len(re.findall(r"[A-Za-z]+", r["reasoning"])))
base = {k: statistics.median(v) for k, v in words.items()}
a.out.write_text(json.dumps(base, indent=1))
print(f"{len(base)} contexts, median words {statistics.median(base.values())} (min {min(base.values())}, max {max(base.values())}) -> {a.out}")
