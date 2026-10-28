"""Transfer eval (F0 wording, user placement, 108 items) per control type: strict passes (meta allowed) and partial compliance for base
(Tinker and vLLM), merge6 and the components that have a transfer run; which components trained on each type; no-requirement
accuracy. Reads results/transfer/<run>/summary.json.

    python3 transfer_table.py      # writes ../results/merge/transfer_compare.md
"""

import json
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "results/transfer"
RUNS = {"base (Tinker)": "gptoss-base", "base (vLLM)": "merge-vllm-base", "merge6 (vLLM)": "merge-merge6", "grpo3 s100": "gptoss-grpo3s100",
        "manytypes1b s150": "mt1b-s150", "manytypes2e s170": "manytypes2e-s170", "len2 s60": "gptoss-len2s60"}
TRAINED = {  # component -> trained control types (results/rl/<run>/config.json); effm2 trained effort levels, len2 token targets
    "grpo3": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "lowercase_thinking", "meow_between_words", "alternating_case"],
    "manytypes1b": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case",
                    "max_words", "questions_only", "no_calculation", "elimination_only", "no_backtracking", "no_commitment"],
    "manytypes2e": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking", "alternating_case",
                    "questions_only", "no_backtracking", "no_commitment", "decoy_topic"],
    "lang2": ["think_language"],
}
S = {m: json.loads((T / r / "summary.json").read_text()) for m, r in RUNS.items()}
types = [k.split("|")[0] for k in S["merge6 (vLLM)"] if k.endswith("|F0|user")]
cell = lambda s, t: (f"{s[t + '|F0|user']['strict']}/{s[t + '|F0|user']['n']}" + (f" ({s[t + '|F0|user']['partial']:.2f})" if s[t + '|F0|user'].get("partial") == s[t + '|F0|user'].get("partial") and s[t + '|F0|user'].get("partial") is not None else "")) if t + "|F0|user" in s else "-"
lines = ["Strict passes (meta-discussion allowed), with mean partial compliance in brackets where the type has a graded score. "
         "'trained by' lists the components whose RL trained that type.\n",
         "| type | trained by | " + " | ".join(RUNS) + " |", "|---|---|" + "---|" * len(RUNS)]
for t in types:
    lines.append(f"| {t} | {', '.join(c for c, ts in TRAINED.items() if t in ts) or '-'} | " + " | ".join(cell(S[m], t) for m in RUNS) + " |")
acc = {m: next(v for k, v in s.items() if k.endswith("|F0|user") and v["n"] == 108)["base"] for m, s in S.items()}
lines.append("| no-requirement accuracy (baseline arm) | | " + " | ".join(f"{acc[m]}/108" for m in RUNS) + " |")
common = [t for t in types if all(t + "|F0|user" in S[m] for m in RUNS)]
held = [t for t in common if not any(t in ts for ts in TRAINED.values())]
for name, ts in (("types in every run", common), ("of those, trained by no component", held)):
    sums = {m: (sum(S[m][t + "|F0|user"]["strict"] for t in ts), sum(S[m][t + "|F0|user"]["n"] for t in ts)) for m in RUNS}
    lines.append(f"| sum over {len(ts)} {name} | | " + " | ".join(f"{a}/{b}" for a, b in sums.values()) + " |")
lines.append(f"\nTypes trained by no component: {', '.join(held)}.")
md = "\n".join(lines)
print(md)
(T.parent / "merge/transfer_compare.md").write_text(md + "\n")
