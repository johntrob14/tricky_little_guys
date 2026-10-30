"""Astra CoT-Control (medium effort, 108 units) for the equal-weight merge of six gpt-oss-120b CoT-control LoRAs, its components and base.

Same metrics and code as rl/astra_levels.py: strict passes (meta-discussion allowed), length-adjusted strict passes (logistic of the
strict pass on constraint + log2(CoT words + 1), fit on base's 324 rows at low/medium/high; adjusted = observed - (sum of the run's
predicted passes - base medium's)), with the gain over base medium and a 95% bootstrap CI over items; strict without meta-discussion,
partial compliance (rl/astra_continuous.py's graded score, mean over units), answer accuracy, median CoT words; and the no-request run's median CoT words (the same items with no requirement) where one exists.
Backend per run: Tinker (the stored runs) or vLLM on the pod (bf16 base, LoRA unmerged, no unembed LoRA; see results/merge/NOTES.md).

    ../rl/.venv/bin/python summarize.py      # writes ../results/merge/astra_summary.{md,json}
"""

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl"))
from astra_continuous import load as load_partial  # noqa: E402
from astra_levels import adjusted, add_length_adjustment, bootstrap, frame, load  # noqa: E402

R = HERE.parent / "results"
# label -> (request run dir, no-request run dir or None, backend); paths relative to results/tinker-runs (astra_levels.RUNS)
RUNS = {
    "base": ("gptoss120b-t1", "gptoss120b-noreq", "Tinker"),
    "base (vLLM)": ("../merge/astra-vllm-base-t1", "../merge/astra-vllm-base-noreq", "vLLM"),
    "merge6": ("../merge/astra-merge6-t1", "../merge/astra-merge6-noreq", "vLLM"),
    "grpo3 s100": ("grpo3-step100-t1", None, "Tinker"),
    "grpo3 s100 (vLLM)": ("../merge/astra-vllm-grpo3-t1", None, "vLLM"),
    "manytypes1b s150": ("astra-mt1b-s150-medium-t1", None, "Tinker"),
    "manytypes2e s170 (vLLM)": ("../merge/astra-vllm-manytypes2e-t1", None, "vLLM"),
    "len2 s60": ("len2-step60-t1", None, "Tinker"),
    "effm2 s50": ("effm2-step50-t1", None, "Tinker"),
    "lang2 s18 (vLLM)": ("../merge/astra-vllm-lang2-t1", None, "vLLM"),
}
BASE_LEVELS = {"low": "astra-base-low-t1", "high": "astra-base-high-t1"}


def noreq_words(run):
    rows = [json.loads(l) for l in open(R / "tinker-runs" / run / "generations/gpt-oss-120b/astra.jsonl")]
    return st.median(len((r["reasoning"] or "").split()) for r in rows), len(rows)


def main():
    data = {"base": {"medium": load(RUNS["base"][0]), **{e: load(r) for e, r in BASE_LEVELS.items()}}}
    data |= {m: {"medium": load(req)} for m, (req, _, _) in RUNS.items() if m != "base"}
    df = frame(data)
    odds = add_length_adjustment(df, "base")
    adj = adjusted(df)
    boots, _ = bootstrap(df, 2000)
    base_noreq = noreq_words(RUNS["base"][1])[0]
    out, lines = {}, []
    lines.append(f"Logistic fit on base's 324 rows (low/medium/high): odds x{odds:.2f} per doubling of CoT words.\n")
    lines.append("| model | backend | length-adjusted strict passes (gain over base, 95% CI) | strict passes (meta allowed) | partial compliance, mean | strict passes without meta-discussion | correct answers | median CoT words | no-request median CoT words (ratio to base's) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m, (req, noreq, backend) in RUNS.items():
        rec = data[m]["medium"]
        a = adj[m, "medium"]
        gains = [b[m, "medium"]["logistic"] - b["base", "medium"]["observed"] for b in boots]
        lo, hi = np.percentile(gains, [2.5, 97.5])
        n = len(rec)
        strict = sum(r["strict"] == "1" for r in rec.values())
        nometa = sum(r["strict_nometa"] == "1" for r in rec.values())
        correct = sum(r["correct"] in ("1", "True") for r in rec.values())
        words = st.median(r["words"] for r in rec.values())
        part = float(np.mean([r["score"] for r in load_partial(req)]))
        nr = noreq_words(noreq) if noreq else None
        out[m] = dict(backend=backend, run=req, n=n, adjusted=a["logistic"], gain=a["logistic"] - a["base"], gain_ci=[lo, hi], strict=strict, partial=part,
                      strict_nometa=nometa, correct=correct, median_cot_words=words,
                      noreq_median_cot_words=nr[0] if nr else None, noreq_n=nr[1] if nr else None, noreq_ratio=nr[0] / base_noreq if nr else None)
        nrs = f"{nr[0]:.0f} ({nr[0] / base_noreq:.2f}x, n={nr[1]})" if nr else "-"
        lines.append(f"| {m} | {backend} | {a['logistic']:.1f}/{n} ({a['logistic'] - a['base']:+.1f} [{lo:+.1f}, {hi:+.1f}]) | {strict}/{n} | {part:.3f} | {nometa}/{n} | {correct}/{n} | {words:.0f} | {nrs} |")
    md = "\n".join(lines)
    print(md)
    (R / "merge/astra_summary.md").write_text(md + "\n")
    (R / "merge/astra_summary.json").write_text(json.dumps(out, indent=1))


main()
