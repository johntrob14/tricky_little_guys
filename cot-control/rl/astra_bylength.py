"""Astra strict passes (meta allowed) for every model and checkpoint we have run, as a function of CoT length.

For each run: strict passes of n, median CoT words, median response words, and strict passes per CoT-word bin (k/n). CoT length
is counted in whitespace words so that models with different tokenizers share one axis.

Length-only prediction: thinkvec/lenadj.py's logistic (constraint intercept + log2(CoT words + 1)) fit on the pooled rows of the
untrained models under the plain Astra prompt (POOL below). "predicted" = sum of P(pass | constraint, CoT words) over the run's
rows; "obs - pred" is what CoT length does not explain. Smoke runs (27 items) and steering/thinkvec conditions are left out.

    .venv/bin/python astra_bylength.py --out ../results/astra_bylength
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thinkvec"))
from lenadj import add_length_adjustment  # noqa: E402

csv.field_size_limit(10**9)
R = Path(__file__).resolve().parent.parent / "results"
T, O = R / "tinker-runs", R / "openrouter-runs"
SWITCH = Path.home() / "code/cot-control-switch/results_local/full1"
EDGES = [0, 6, 51, 151, 501, 1501, 10**9]
BINS = ["0-5", "6-50", "51-150", "151-500", "501-1500", "1501+"]

# (group, model, setting, run dir); gpt-oss "medium" runs used the template default effort
RUNS = [
    ("untrained, plain prompt", "gpt-oss-120b", "low", T / "astra-base-low-t1"),
    ("untrained, plain prompt", "gpt-oss-120b", "medium", T / "gptoss120b-t1"),
    ("untrained, plain prompt", "gpt-oss-120b", "medium, greedy", T / "gptoss120b-greedy"),
    ("untrained, plain prompt", "gpt-oss-120b", "high", T / "astra-base-high-t1"),
    ("untrained, plain prompt", "DeepSeek V4.1 Flash", "greedy, 'reasoning stage'", T / "run1"),
    ("untrained, plain prompt", "DeepSeek V4.1 Flash", "greedy, 'thinking'", T / "run2"),
    ("untrained, plain prompt", "DeepSeek V4.1 Flash", "T1", T / "ds-t1-run2"),
    ("untrained, plain prompt", "DeepSeek V4.1 Flash 1017", "OpenRouter T1", O / "ds1017-up"),
    ("untrained, plain prompt", "GLM-5.3", "'reasoning stage'", T / "glm53-run1"),
    ("untrained, plain prompt", "GLM-5.3", "'thinking'", T / "glm53-run2"),
    ("untrained, plain prompt", "Nemotron-3-Ultra", "full, 'reasoning stage'", T / "nemo-ultra-run1"),
    ("untrained, plain prompt", "Nemotron-3-Ultra", "full, 'thinking'", T / "nemo-ultra-run2"),
    ("untrained, plain prompt", "Nemotron-3-Ultra", "medium", T / "nemo-ultra-med-run2"),
    ("untrained, plain prompt", "Qwen3.8-27B", "'reasoning stage'", T / "qwen38-run1"),
    ("untrained, plain prompt", "Qwen3.8-27B", "'thinking'", T / "qwen38-run2"),
    ("untrained, plain prompt", "Inkling-Small", "OpenRouter", O / "inksm-up"),
    ("untrained, plain prompt", "Inkling-Small", "Tinker", T / "inksmtk-up"),
    *[("untrained, plain prompt", "Nemotron-3-Nano", f"seed {s}", T / f"nano-astra-s{s}") for s in range(3)],
    *[("untrained, plain prompt", "Nemotron-3-Nano", f"term '{w}'", T / f"nano-astra-term-{w}") for w in ("analysis", "thinkblock", "thinking")],
    ("untrained, plain prompt", "SWITCH-latent (cot-control-switch)", "full1", SWITCH),
    *[("prompted elicitation", "gpt-oss-120b", f"medium, {e}", T / f"gptoss120b-{e}-t1") for e in ("upcp", "upgp", "cb")],
    *[("prompted elicitation", "DeepSeek V4.1 Flash 1017", e, O / f"ds1017-{e}") for e in ("upcp", "upgp", "cb")],
    *[("prompted elicitation", "Qwen3.8-27B", e, T / f"qwen38-{e}") for e in ("upcp", "upgp", "cb", "cp", "cbonly")],
    ("prompted elicitation", "Inkling-Small", "OpenRouter cb", O / "inksm-cb"),
    *[("prompted elicitation", "Inkling-Small", f"Tinker {e}", T / f"inksmtk-{e}") for e in ("upcp", "upgp")],
    *[("requirement in system prompt", m, lv, T / f"astra_sys-{m}-{d}-t1") for m in ("base", "len2") for lv, d in (("low", "low"), ("medium", "med"), ("high", "high"))],
    *[("gpt-oss length/effort RL", m, lv, T / (f"astra-{m}-{lv}-t1" if lv != "medium" else f"{m}-step{s}-t1"))
      for m, s in (("len2", 60), ("eff1", 40), ("eff2", 40), ("effm1", 30), ("effm2", 50), ("efflen1", 100)) for lv in ("low", "medium", "high")],
    *[("gpt-oss single-type GRPO (st1)", f"st1-{t}", lv, T / f"astra-st1-{t}-{lv}-t1")
      for t in ("words", "tokens", "len2w", "third", "qonly", "ignore", "ignore2") for lv in ("low", "medium", "high")],
    *[("gpt-oss CoT-control GRPO", "grpo2", f"step {s}", T / f"grpo2-step{s}-t1") for s in (20, 40)],
    *[("gpt-oss CoT-control GRPO", "grpo3", f"step {s}", T / f"grpo3-step{s}-t1") for s in (20, 40, 60, 80, 100)],
    *[("Qwen3.8 OPSD / GRPO", "opsd1", f"step {s}", T / f"opsd1-step{s}-t1") for s in (20, 30, 40)],
    *[("Qwen3.8 OPSD / GRPO", "opsd-grpo1", f"step {s}", T / f"opsd-grpo1-step{s}-t1") for s in (20, 30)],
]
POOL = {"astra-base-low-t1", "gptoss120b-t1", "astra-base-high-t1", "ds-t1-run2", "ds1017-up", "glm53-run2", "nemo-ultra-run2",
        "nemo-ultra-med-run2", "qwen38-run2", "inksm-up", "inksmtk-up", "nano-astra-s0", "nano-astra-s1", "nano-astra-s2"}
KEY = ("dataset", "item_idx", "constraint", "sample_idx")


def load(run: Path):
    rec = {tuple(r[k] for k in KEY): r for r in csv.DictReader(open(run / "analysis/records.csv")) if r["protocol"].startswith("astra")}
    gen = next(run.glob("generations/*/astra*.jsonl"))
    for line in open(gen):
        x = json.loads(line)
        k = (x["dataset"], str(x["item_idx"]), x["constraint"], str(x["sample_idx"]))
        if k in rec:
            rec[k]["cot_words"] = len((x["reasoning"] or "").split())
            rec[k]["resp_words"] = len((x["response"] or "").split())
    return list(rec.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for i, (g, m, s, run) in enumerate(RUNS):
        for r in load(run):
            rows.append(dict(run_i=i, run=run.name, group=g, model=m, setting=s, constraint=r["constraint"], strict=r["strict"] == "1",
                             cot_words=r["cot_words"], resp_words=r["resp_words"], trunc=r["finish_reason"] == "length",
                             condition="pool" if run.name in POOL else "other"))
    df = pd.DataFrame(rows)
    odds = add_length_adjustment(df, "pool")
    df["bin"] = pd.cut(df.cot_words, EDGES, labels=BINS, right=False)
    df.to_csv(args.out / "rows.csv", index=False)

    md = [f"# Astra strict passes (meta allowed) by CoT length, every run\n\nGenerated by `rl/astra_bylength.py` (method in its docstring). "
          f"Length-only logistic fit on {len(POOL)} plain-prompt runs of untrained models ({int((df.condition == 'pool').sum())} rows): "
          f"odds of a pass x{odds:.2f} per doubling of CoT words. Bin cells: passes / rows with CoT words in that bin.\n"]
    summ = []
    for g, gdf in df.groupby("group", sort=False):
        md += [f"\n## {g}\n", "| model | setting | run | passes | predicted from length | obs - pred | median CoT words | median response words | trunc | "
               + " | ".join(f"CoT {b}" for b in BINS) + " |", "|" + "---|" * (8 + len(BINS))]
        for _, r in gdf.groupby("run_i", sort=True):
            x = r.iloc[0]
            cells = []
            for b in BINS:
                rb = r[r.bin == b]
                cells.append(f"{int(rb.strict.sum())}/{len(rb)}" if len(rb) else "-")
            obs, pred = int(r.strict.sum()), r.p_len.sum()
            md.append(f"| {x.model} | {x.setting} | `{x.run}` | {obs}/{len(r)} | {pred:.1f} | {obs - pred:+.1f} | {st.median(r.cot_words):.0f} | "
                      f"{st.median(r.resp_words):.0f} | {int(r.trunc.sum())} | " + " | ".join(cells) + " |")
            summ.append(dict(group=g, model=x.model, setting=x.setting, run=x.run, n=len(r), passes=obs, predicted=round(pred, 2),
                             median_cot_words=st.median(r.cot_words), median_resp_words=st.median(r.resp_words)))
    pool = df[df.condition == "pool"]
    md += ["\n## Pooled untrained plain-prompt rows: pass rate by CoT-word bin\n", "| bin | rows | passes | rate |", "|---|---|---|---|"]
    for b in BINS:
        pb = pool[pool.bin == b]
        md.append(f"| {b} | {len(pb)} | {int(pb.strict.sum())} | {pb.strict.mean() if len(pb) else float('nan'):.3f} |")
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    pd.DataFrame(summ).to_csv(args.out / "summary.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
