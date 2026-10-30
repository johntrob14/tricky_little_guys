"""Combine hintuse_set.py runs into one directory that hintuse_html.py shows as a single multi-item run.

Each --src is dir:item,item,... (the items to take from that run). Item ids become strings ("58", "f17baca8"); only --models and --arms are kept.
Rows, judged rows and per-item config fields are copied as they are; nothing is resampled or rejudged. The judge config is the first source's.

    python hintuse_combine.py --src ../results/case_studies/hintuse_hardset:58,60,197,210,251,274,352,375 \
        ../results/case_studies/hintuse_dapo:f17baca8,2bb6f9ac,500acf38,04a6efea,8e21adea,fa12e2ec --out ../results/case_studies/hintuse_combined
"""

import argparse
import json
import shutil
from pathlib import Path

PER_ITEM = ["answers", "qblocks", "hints", "hint_names"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, help="dir:item,item,...")
    ap.add_argument("--models", nargs="+", default=["base", "mt1b-s150"])
    ap.add_argument("--arms", nargs="+", default=["none", "hint", "hide", "nocot", "nocot_hint"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg, rows, judged, sources = None, [], [], []
    for src in args.src:
        d, items = src.split(":")
        d, items = Path(d), items.split(",")
        c = json.load(open(d / "config.json"))
        keep = lambda r: str(r["item"]) in items and r["model"] in args.models and r["arm"] in args.arms
        rows += [r | dict(item=str(r["item"])) for r in map(json.loads, open(d / "rows.jsonl")) if keep(r)]
        judged += [r | dict(item=str(r["item"])) for r in map(json.loads, open(d / "judged.jsonl")) if keep(r)]
        part = dict(items=items, **{k: {i: c[k][i] for i in items if i in c[k]} for k in PER_ITEM},
                    prompts={k: p for k, p in c["prompts"].items() if k.split("|")[0] in items and k.split("|")[1] in args.arms})
        if cfg is None:
            cfg = dict(c, dataset="mixed", models={m: c["models"][m] for m in args.models}, arms=args.arms, **part)
            shutil.copy(d / "judge.json", args.out / "judge.json")
        else:
            assert (c["n"], c["max_tokens"], c["temperature"], c["hide"]) == (cfg["n"], cfg["max_tokens"], cfg["temperature"], cfg["hide"]), d
            cfg["items"] += items
            for k in PER_ITEM + ["prompts"]:
                cfg[k] |= part[k]
        sources.append(dict(dir=str(d), dataset=c.get("dataset", "gpqa"), items=items))
    cfg["sources"] = sources
    json.dump(cfg, open(args.out / "config.json", "w"), indent=1)
    with open(args.out / "rows.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(args.out / "judged.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in judged)
    print(f"{len(cfg['items'])} items, {len(rows)} rows, {len(judged)} judged -> {args.out}")


if __name__ == "__main__":
    main()
