"""Training pool with harder questions: pool_k4 (MMLU-Pro) plus the GPQA questions of data/obf_train_gptoss_r2.jsonl that base gpt-oss-120b
gets right with a CoT but not without one, repeated --repeat times so they make up a larger share of the draws.

GPQA kept: base CoT accuracy (4 samples) >= --min-acc, no-CoT accuracy (2 samples) <= --max-nocot, no base sample truncated, >= 3 keywords, and the
question not an eval item: the transfer-eval and Astra (surface) items, read from results/transfer/{manytypes2-s80,gptoss-base-surface}/gens.jsonl,
or the hint-verbalization questions.

    python build_pool_hard.py --out data/pool_k4_gpqahard.jsonl
"""
import argparse
import json
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-acc", type=float, default=0.25)
    ap.add_argument("--max-nocot", type=float, default=0.0)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()
    gens = [json.loads(l) for run in ("manytypes2-s80", "gptoss-base-surface") for l in open(DATA.parent.parent / "results/transfer" / run / "gens.jsonl")]
    blocks = {norm(r["qblock"]) for r in gens if r["dataset"] == "gpqa"}
    hint = {norm(json.loads(l)["question"]) for l in open(DATA / "hintverb_questions.jsonl")}
    held = lambda q: norm(q) in hint or any(norm(q)[:200] in b for b in blocks)
    rows = [json.loads(l) for l in open(DATA / "obf_train_gptoss_r2.jsonl")]
    gpqa = [r for r in rows if r["source"] == "gpqa"]
    keep = [r for r in gpqa if r["correct"] >= args.min_acc and r["nocot_correct"] <= args.max_nocot and not r["truncated"] and len(r["keywords"]) >= 3
            and not held(r["question"]) and r["base_words"] >= 10]
    print(f"GPQA rows {len(gpqa)}, eval GPQA blocks {len(blocks)}, overlapping eval or hint questions {sum(held(r['question']) for r in gpqa)}, kept {len(keep)}; "
          f"kept base CoT accuracy {sum(r['correct'] for r in keep) / len(keep):.3f}, median base words {sorted(r['base_words'] for r in keep)[len(keep) // 2]}")
    fields = ["qid", "question", "options", "answer", "keywords", "base_words", "correct"]
    pool = [json.loads(l) for l in open(DATA / "pool_k4.jsonl")]
    with open(args.out, "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
        for _ in range(args.repeat):
            for r in keep:
                f.write(json.dumps({k: r[k] for k in fields} | {"source": "gpqa"}) + "\n")
    print(f"{len(pool)} MMLU-Pro + {args.repeat} x {len(keep)} GPQA -> {args.out}")


if __name__ == "__main__":
    main()
