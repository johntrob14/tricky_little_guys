"""CoT length under a CoT-control request, paired by item, for base DeepSeek, rwsft1 ep2, obf2g s25 and the SFT checkpoint on the transfer eval (F0 wording, user placement, effort 75,
12 items x 3 datasets). The SFT run sampled only the no-requirement arm and six types, so every number below is restricted to those six types. Method and the full 27-type tables for the
first three runs: cot-control/rl/fry_cotlen.py, cot-control/results/fry/NOTES.md. Run with the system python: python ds41_sft1/fry.py > results/fry_length.md"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cot-control/rl"))
import fry_cotlen as f  # noqa: E402

RUNS = ["ds-base", "ds-rwsft1ep2", "ds-obf2gs25", "ds-ds41sft1ep2"]
TYPES = ["word_suppression", "max_words", "no_backtracking", "ignore_question", "alternating_case", "meow_between_words"]
summ = {r: f.summarize(r, "F0", "user") for r in RUNS}
ref = summ["ds-base"]["baseline_median"]
print("| run | no-requirement median tokens (vs base) | no-requirement accuracy | median over the six types of (tokens with requirement / without), paired | " + " | ".join(TYPES) + " |")
print("|---|---|---|---|" + "---|" * len(TYPES))
import json
for r in RUNS:
    s = summ[r]
    gens = [json.loads(l) for l in open(f.RESULTS / "transfer" / r / "gens.jsonl")]
    base = [g for g in gens if g["type"] == "baseline"]
    ratios = [2 ** s["types"][t]["median_log2_ratio"] for t in TYPES]
    print(f"| {r} | {s['baseline_median']:.0f} ({s['baseline_median'] / ref:.2f}x) | {sum(bool(g['correct']) for g in base)}/{len(base)} | {statistics.median(ratios):.2f} | " + " | ".join(f"{x:.2f}" for x in ratios) + " |")
