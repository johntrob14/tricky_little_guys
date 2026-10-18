"""Re-derive the extraction fields of existing generation rows with the current extract.py.

    python scripts/resplit.py SRC.jsonl DST.jsonl

Recomputes reasoning, reasoning_marked, response, cot_rule, think_closed, answer_pred,
answer_source, answer_pred_lenient, correct, correct_lenient from raw_output and
finish_reason (via vllm_runner.build_row's logic); every other field is copied. Writes a
new file and never modifies SRC. Each row gets `resplit_from` (SRC path) for provenance.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cotswitch import extract  # noqa: E402
from cotswitch.io import append_jsonl, load_done_keys, read_jsonl  # noqa: E402


def resplit(row: dict) -> dict:
    split = extract.split_output(row["raw_output"], row.get("condition"), row.get("finish_reason"))
    ans, src = extract.parse_answer_with_source(split["response"], split["content"])
    lenient = extract.parse_answer_lenient(split["response"], split["content"])
    gold = row.get("answer_gold")
    out = dict(row)
    out.update({k: split[k] for k in ("reasoning", "reasoning_marked", "response", "cot_rule", "think_closed")})
    out.update(answer_pred=ans, answer_source=src, answer_pred_lenient=lenient,
               correct=None if gold is None else ans == gold,
               correct_lenient=None if gold is None else lenient == gold)
    return out


def main(src, dst):
    done = load_done_keys(dst)
    n = 0
    for row in read_jsonl(src):
        if row["key"] in done:
            continue
        r = resplit(row)
        r["resplit_from"] = str(src)
        append_jsonl(dst, r)
        n += 1
    print(f"[resplit] {src} -> {dst}: +{n}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
