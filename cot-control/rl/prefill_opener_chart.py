"""Per-type uplift from prefilling the analysis channel with a fixed opener (--prefill on transfer_eval.py), gpt-oss-120b base and CoT-control organisms.

Transfer eval, shared 108 items (36 each GPQA, HLE, MMLU-Pro mini), F0 wording, user turn, 21 types, one sample per item and type. Each model's prefilled run
(results/transfer/<prefill run>) is paired by item with its stored no-prefill run. Compliance is strict pass with meta-discussion allowed. The organisms
often echo the prefill as a closing tag at the end of the CoT (e.g. "... answer is D. <|FLAMINGO_BURRITO|>"), which fails style graders (lowercase,
alternating case, meow) for reasons unrelated to control, so programmatic types are regraded with any token containing a prefill word stripped from
the CoT; judge-graded types (requirement judge, ignore_question) keep their judgments. Panels: no-prefill pass rate; prefilled minus not with a 95% paired
bootstrap interval over items; median output tokens prefilled / not (graded text is the continuation only).

    .venv/bin/python prefill_opener_chart.py [--tag fb] [--prefill '<|FLAMINGO_BURRITO|>']
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import controls
from cotswitch.data import load_items
from difficulty_compliance import GRID, SURFACE, TEXT, TEXT2, TYPES
from gpqa_scatter import TRAINED

matplotlib.use("Agg")
RESULTS = Path(__file__).resolve().parent.parent / "results/transfer"
FIGS = Path(__file__).resolve().parents[3] / "figures"
MODELS = [("base", "gptoss-base", None, "#3987e5"), ("grpo3 s100", "gptoss-grpo3s100", "grpo3", "#d95926"),
          ("manytypes1b s150", "mt1b-s150", "mt1b", "#199e70"), ("manytypes2e s170", "manytypes2e-s170", "mt2e", "#9085e9")]
PRE_RUN = {"gptoss-base": "base", "gptoss-grpo3s100": "grpo3s100", "mt1b-s150": "mt1b-s150", "manytypes2e-s170": "mt2e-s170"}


def load(run, pat):
    jd = {(r["key"], r["kind"]): r for r in map(json.loads, open(RESULTS / run / "judgments.jsonl"))}
    items = {ds: {it.item_idx: it for it in load_items(ds)} for ds in ("gpqa", "hle", "mmlu_pro")}
    out = {}
    for r in map(json.loads, open(RESULTS / run / "gens.jsonl")):
        if r["type"] != "baseline" and (r["frame"] != "F0" or r["placement"] != "user"):
            continue
        kind = "requirement" if r["type"] in controls.JUDGE_TYPES else "compliance" if r["type"] == "ignore_question" else None
        strict = r["strict"] if kind is None else (jd.get((r["key"], kind)) or {}).get("score")
        if kind is None and r["type"] in controls.CONTROLS and pat.search(r["reasoning"] or ""):
            strict = controls.grade(controls.CONTROLS[r["type"]], pat.sub("", r["reasoning"]), r["value"], items[r["dataset"]][r["item_idx"]])[1]
        out[(r["type"], r["dataset"], r["item_idx"])] = dict(ok=float(bool(strict)), correct=float(r["correct"]), tokens=r["n_output_tokens"])
    return out


def paired(a, b, keys, field, n=2000, seed=0):
    d = np.array([b[k][field] - a[k][field] for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    return d.mean(), np.percentile(d[idx].mean(1), [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fb")
    ap.add_argument("--prefill", default="<|FLAMINGO_BURRITO|>")
    args = ap.parse_args()
    pat = re.compile(r"<?\|?[A-Za-z_]*(?:" + "|".join(re.findall("[A-Za-z]{3,}", args.prefill)) + r")[A-Za-z_]*\|?>?", re.I)
    rows = ["all 21 types"] + TYPES
    fig, (axb, axu, axl) = plt.subplots(1, 3, figsize=(17, 11), gridspec_kw=dict(width_ratios=[0.6, 1.4, 1]), facecolor=SURFACE)
    summary = {}
    for mi, (label, run, tkey, color) in enumerate(MODELS):
        a, b = load(run, pat), load(f"{args.tag}-{PRE_RUN[run]}", pat)
        keys = sorted(set(a) & set(b))
        assert len(keys) == len(b), (run, len(keys), len(a), len(b))
        off = (mi - 1.5) * 0.18
        for ri, t in enumerate(rows):
            tk = [k for k in keys if k[0] != "baseline"] if ri == 0 else [k for k in keys if k[0] == t]
            mean, (lo, hi) = paired(a, b, tk, "ok")
            trained = t in TRAINED.get(tkey, [])
            axb.plot(100 * np.mean([a[k]["ok"] for k in tk]), ri + off, "o", ms=6, color=color if trained or ri == 0 else SURFACE, mec=color, mew=1.5)
            axu.plot([100 * lo, 100 * hi], [ri + off] * 2, color=color, lw=1.5, alpha=0.8)
            axu.plot(100 * mean, ri + off, "o", ms=6, color=color if trained or ri == 0 else SURFACE, mec=color, mew=1.5)
            ratio = st.median(b[k]["tokens"] for k in tk) / st.median(a[k]["tokens"] for k in tk)
            axl.plot(ratio, ri + off, "o", ms=6, color=color if trained or ri == 0 else SURFACE, mec=color, mew=1.5)
            summary[(label, t)] = (sum(a[k]["ok"] for k in tk), sum(b[k]["ok"] for k in tk), len(tk), ratio)
        bk = [k for k in keys if k[0] == "baseline"]
        ratio = st.median(b[k]["tokens"] for k in bk) / st.median(a[k]["tokens"] for k in bk)
        axl.plot(ratio, len(rows) + off, "o", ms=6, color=color, mec=color, mew=1.5)
        summary[(label, "no requirement")] = (sum(a[k]["correct"] for k in bk), sum(b[k]["correct"] for k in bk), len(bk), ratio)
        axb.plot([], [], "o", color=color, label=label)
    labels = rows + ["no requirement"]
    for ax in (axb, axu, axl):
        ax.set_facecolor(SURFACE)
        ax.set_yticks(range(len(labels)), labels, fontsize=8.5, color=TEXT2)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        for i in range(len(labels)):
            ax.axhspan(i - 0.5, i + 0.5, color=GRID, alpha=0.25 if i % 2 else 0, lw=0)
        ax.tick_params(colors=TEXT2, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axu.axvline(0, color=TEXT2, lw=1)
    axl.axvline(1, color=TEXT2, lw=1)
    axl.set_xscale("log", base=2)
    axu.set_yticklabels([])
    axl.set_yticklabels([])
    axb.set_xlim(-5, 105)
    axb.set_xlabel("strict pass rate (meta allowed)\nwithout prefill, %", fontsize=9, color=TEXT2)
    axu.set_xlabel("strict pass rate (meta allowed), prefilled minus not,\npercentage points (95% paired bootstrap over items)", fontsize=9, color=TEXT2)
    axl.set_xlabel("median output tokens, prefilled / not\n(log scale; no requirement = the no-instruction arm)", fontsize=9, color=TEXT2)
    fig.legend(*axb.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.995, 0.975), ncol=4, fontsize=9, facecolor=SURFACE, edgecolor=GRID, labelcolor=TEXT,
               title="filled marker = type in that organism's training set", title_fontsize=8.5, framealpha=1).get_title().set_color(TEXT2)
    fig.suptitle(f"gpt-oss-120b: analysis channel prefilled with {args.prefill} vs not, per control type", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.915, "transfer eval: 108 items per type (36 each GPQA, HLE, MMLU-Pro mini), F0 wording in the user turn, T 1.0, one sample per item.\n"
             "Graded CoT = the continuation after the prefill, with echoes of the prefill (organisms often close the CoT with it)\nstripped before regrading code-graded types; judge-graded types as judged (they also need a parsed final answer).\n"
             "No-prefill runs are the stored ones (same prompts except the chat template's date line).", color=TEXT2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    out = FIGS / f"prefill_opener_{args.tag}_gptoss.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)
    for (label, t), (n0, n1, n, ratio) in summary.items():
        print(f"{label:18s} {t:28s} {n0:4.0f}/{n} -> {n1:4.0f}/{n}  length x{ratio:.2f}")


if __name__ == "__main__":
    main()
