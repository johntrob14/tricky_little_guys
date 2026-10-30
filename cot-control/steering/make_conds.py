"""Write condition lists for steer.py, split round-robin over workers.

    python make_conds.py pilot  --alphas 0.05,0.1,0.2,0.4,0.8 --workers 2
    python make_conds.py sweep  --alphas 9:a,b,c/18:a,b,c/27:a,b,c --workers 2 [--sign 1] [--prefix ""]
    python make_conds.py gen    --alphas 0.2,0.4 --vecs B --mode gen --baseline ""
    python make_conds.py org    --alphas 0.2,0.4 --layers 18 --sign -1 --prefix org_ --baseline unsteered --workers 1
"""
from __future__ import annotations

import argparse
import json

from common import HERE, LAYERS

VECS = ["A", "B", "C_org", "C_base", "rand"]
ap = argparse.ArgumentParser()
ap.add_argument("kind")
ap.add_argument("--alphas", required=True)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--sign", type=int, default=1)
ap.add_argument("--vecs", default=",".join(VECS))
ap.add_argument("--prefix", default="")
ap.add_argument("--baseline", default="base_unsteered", help="empty for none")
ap.add_argument("--mode", default="all", choices=["all", "gen"])
ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
args = ap.parse_args()

if "/" in args.alphas or ":" in args.alphas:
    per_layer = {int(k): [float(x) for x in v.split(",")] for k, v in (p.split(":") for p in args.alphas.split("/"))}
else:
    per_layer = {l: [float(x) for x in args.alphas.split(",")] for l in [int(x) for x in args.layers.split(",")]}
conds = [{"name": f"{args.prefix}{args.baseline}", "vec": "none", "layer": None, "alpha": 0.0, "sign": 1}] if args.baseline else []
layers = [int(x) for x in args.layers.split(",")]
for v in args.vecs.split(","):
    for l in layers:
        for a in per_layer[l]:
            s = "m" if args.sign < 0 else ""
            g = "_gen" if args.mode == "gen" else ""
            conds.append({"name": f"{args.prefix}{v}_L{l}_{s}a{a:g}{g}", "vec": v, "layer": l, "alpha": a, "sign": args.sign,
                          "mode": args.mode})
d = HERE / "conds"; d.mkdir(exist_ok=True)
for w in range(args.workers):
    (d / f"{args.kind}_w{w}.json").write_text(json.dumps(conds[w::args.workers], indent=1))
print(len(conds), "conditions ->", [f"{args.kind}_w{w}.json" for w in range(args.workers)])
