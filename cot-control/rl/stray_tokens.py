"""Count stray non-Latin script in CoTs on English questions (effm1's fry began this way), per 10-step block and effort level.

A question is English when its text has no letters outside Latin and Greek. A row is flagged when its CoT has a letter from
another script (CJK, Cyrillic, Gujarati, ...). fry_watch's non_latin share cannot see these: quoted Chinese questions dominate it.
Prints flagged rows and the rate per million CoT tokens, next to base (results/math_pilot, math_filter rows).

    python stray_tokens.py --run effm2 [--show 21]
"""

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
EFFORTS = ["low", "medium", "high"]


def foreign(text):
    """Letters outside Latin and Greek."""
    out = []
    for ch in text:
        if ord(ch) < 0x250 or not ch.isalpha() or 0x2100 <= ord(ch) <= 0x214F or 0x1D400 <= ord(ch) <= 0x1D7FF:
            continue  # Latin, letterlike math symbols (ℤ, ℓ) and math alphanumerics
        name = unicodedata.name(ch, "")
        if not name.startswith(("LATIN", "GREEK", "MATHEMATICAL", "COPTIC")):
            out.append(ch)
    return out


def problems():
    p = {}
    for x in json.load(open(ROOT / "rl/data/dapo_dedup.json")):
        p[x["qid"]] = x["problem"]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--show", type=int, default=0, help="print flagged snippets from blocks starting at this step")
    args = ap.parse_args()
    prob = problems()
    english = lambda q: q in prob and not foreign(prob[q])
    tally = defaultdict(lambda: [0, 0, 0])  # (block, effort) -> [english rows, flagged rows, cot tokens]
    shown = []
    for line in open(ROOT / f"results/rl/{args.run}/rollouts.jsonl"):
        r = json.loads(line)
        if not english(r["qid"]):
            continue
        t = tally[(f"{(r['step'] - 1) // 10 * 10 + 1}-{(r['step'] - 1) // 10 * 10 + 10}", r["effort"])]
        t[0] += 1
        t[2] += r["cot_tokens"]
        f = foreign(r["reasoning"] or "")
        if f:
            t[1] += 1
            if args.show and r["step"] >= args.show:
                i = (r["reasoning"] or "").find(f[0])
                shown.append(f"step {r['step']} {r['effort']} {r['qid']}: ...{(r['reasoning'] or '')[max(0, i - 80):i + 40]!r}...")
    for f in ("math_pilot", "math_filter"):
        for line in open(ROOT / f"results/{f}/rows.jsonl"):
            r = json.loads(line)
            if not english(r["qid"]):
                continue
            t = tally[("base", r["effort"])]
            t[0] += 1
            t[2] += r["cot_tokens"]
            t[1] += bool(foreign(r["reasoning"] or ""))
    print("| block | level | English-question rows | rows with stray script | per M CoT tokens |")
    print("|---|---|---|---|---|")
    for (b, e), (n, k, tok) in sorted(tally.items(), key=lambda kv: (kv[0][0] != "base", int(kv[0][0].split("-")[0]) if kv[0][0] != "base" else 0, EFFORTS.index(kv[0][1]))):
        print(f"| {b} | {e} | {n} | {k} | {k / max(tok, 1) * 1e6:.2f} |")
    for s in shown:
        print(s)


if __name__ == "__main__":
    main()
