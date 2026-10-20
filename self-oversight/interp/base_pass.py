"""Base-model pass: ADL position means on fineweb, then fit the J-lens (timing pilot, then scale
to what fits the time budget).

    python base_pass.py --out /root/work/results --jl-max 100 --jl-budget-min 75
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import jlens
import torch
from jlens.examples import load_wikitext_prompts

from adl_lib import fineweb_texts, position_means
from hfmodel import load

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/root/work/results")
ap.add_argument("--adl-n", type=int, default=10000)
ap.add_argument("--jl-pilot", type=int, default=3)
ap.add_argument("--jl-max", type=int, default=0, help="0 skips the J-lens fit")
ap.add_argument("--jl-budget-min", type=float, default=75)
ap.add_argument("--dim-batch", type=int, default=32)
ap.add_argument("--skip-adl", action="store_true")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
out = Path(args.out)
(out / "adl").mkdir(parents=True, exist_ok=True)
(out / "jlens").mkdir(parents=True, exist_ok=True)

t0 = time.time()
hf, tok = load("base")
print(f"base loaded in {time.time()-t0:.0f}s", flush=True)

if not args.skip_adl:
    t = time.time()
    m = position_means(hf, tok, fineweb_texts(args.adl_n))
    torch.save(m, out / "adl" / "means_base.pt")
    print(f"ADL base means: n={m['n']} in {time.time()-t:.0f}s", flush=True)

if args.jl_max <= 0:
    raise SystemExit(0)
model = jlens.from_hf(hf, tok)
prompts = load_wikitext_prompts(args.jl_max)
ckpt = str(out / "jlens" / "base.ckpt")
t = time.time()
jlens.fit(model, prompts[: args.jl_pilot], dim_batch=args.dim_batch, max_seq_len=128,
          checkpoint_path=ckpt, checkpoint_every=1)
spp = (time.time() - t) / args.jl_pilot
n = int(min(args.jl_max, max(args.jl_pilot, math.floor(args.jl_budget_min * 60 / spp))))
print(f"J-lens pilot: {spp:.0f}s per prompt -> fitting {n} prompts "
      f"(budget {args.jl_budget_min} min)", flush=True)
lens = jlens.fit(model, prompts[:n], dim_batch=args.dim_batch, max_seq_len=128,
                 checkpoint_path=ckpt, checkpoint_every=5)
lens.save(str(out / "jlens" / f"base-n{n}.pt"))
meta = {"n_prompts": n, "seconds_per_prompt": spp, "dim_batch": args.dim_batch, "max_seq_len": 128,
        "corpus": "Salesforce/wikitext wikitext-103-raw-v1 train (>=600 chars), first n",
        "total_fit_seconds": time.time() - t}
(out / "jlens" / f"base-n{n}.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta), flush=True)
