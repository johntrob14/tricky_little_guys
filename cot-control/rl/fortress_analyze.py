"""Metrics and figures for the Fortress eval-awareness comparison, the eval_awareness_scaling protocol's: VEA rate and refusal rate over scored
rollouts with 95% percentile CIs from 2,000 bootstrap draws over questions; questions with any VEA; the post's within-question refusal difference
(unweighted mean over questions with both VEA and non-VEA rollouts and refusal below 100% of refusal among VEA rollouts minus the others); the
safety-relevant share of VEA (subcategory HARMFUL_COMPLIANCE or BOUNDARIES); and the interventions: refusal per condition over scored continuations
and the paired differences vea - none, placebo - none, keep - remove (removal continuations that re-verbalize VEA dropped from "remove", and unfiltered),
bootstrapped over bases. Writes results/fortress/summary.first<k>.json and the figures under mats12_sprint/figures/ (--max-idx k: rollout indices used, 5 by default).

    python fortress_analyze.py base organism
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fortress_eval import OUT

FIGS = Path(__file__).parents[3] / "figures"
COLORS = {"base": "#8a8a85", "organism": "#3987e5"}
LABELS = {"base": "base Qwen3.8-27B (OpenRouter)", "organism": "OPSD+GRPO organism (Tinker)"}
rng = np.random.default_rng(20260926)


def boot(groups: list[list], stat, n: int = 2000) -> tuple[float, float, float]:
    """stat(list of groups) and its 95% percentile CI, resampling groups (questions or bases) with replacement."""
    vals = [stat([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n)]
    return stat(groups), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def rate(groups):
    flat = [x for g in groups for x in g]
    return float(np.mean(flat)) if flat else float("nan")


def within_q(groups):
    diffs = [np.mean([r for r, v in g if v]) - np.mean([r for r, v in g if not v]) for g in groups if any(v for _, v in g) and any(not v for _, v in g) and np.mean([r for r, _ in g]) < 1]
    return float(np.mean(diffs)) if diffs else float("nan")


def load(name: str, max_idx: int) -> dict:
    d = OUT / name
    rows = {f"{r['qid']}/{r['idx']}": r for r in map(json.loads, (d / "rollouts.jsonl").open()) if r["idx"] < max_idx}
    j = {t: {v["key"]: v for v in map(json.loads, (d / f"judge.{t}.jsonl").open()) if v["key"] in rows} for t in ("vea_cal", "refusal", "category") if (d / f"judge.{t}.jsonl").exists()}
    scored = {k: r for k, r in rows.items() if r["status"] == "scored"}
    valid = {k: r for k, r in scored.items() if j["vea_cal"].get(k, {}).get("vea") is not None and j["refusal"].get(k, {}).get("refusal") is not None}
    by_q = {}
    for k, r in valid.items():
        by_q.setdefault(r["qid"], []).append((j["refusal"][k]["refusal"], j["vea_cal"][k]["vea"]))
    groups = list(by_q.values())
    s = {"n_rollouts": len(rows), "n_scored": len(scored), "n_valid": len(valid), "status": {st: sum(r["status"] == st for r in rows.values()) for st in ("scored", "truncated", "empty_answer")},
         "median_reasoning_tokens": float(np.median([r["n_reasoning_tokens"] for r in scored.values()])), "n_questions": len(groups)}
    s["vea_rate"] = boot(groups, lambda gs: rate([[v for _, v in g] for g in gs]))
    s["refusal_rate"] = boot(groups, lambda gs: rate([[r for r, _ in g] for g in gs]))
    s["n_vea"] = sum(v for g in groups for _, v in g)
    s["questions_with_vea"] = sum(any(v for _, v in g) for g in groups)
    s["within_q_refusal_diff"] = boot(groups, within_q)
    s["n_informative_questions"] = sum(any(v for _, v in g) and any(not v for _, v in g) and np.mean([r for r, _ in g]) < 1 for g in groups)
    s["quote_check"] = {c: sum(v.get("quote_check") == c for v in j["vea_cal"].values()) for c in ("exact", "near", "fragments", "fail")}
    if "category" in j:
        subs = [v["subcategory"] for v in j["category"].values()]
        s["vea_subcategories"] = {c: subs.count(c) for c in sorted(set(subs), key=str)}
        s["safety_share_of_vea"] = sum(c in ("HARMFUL_COMPLIANCE", "BOUNDARIES") for c in subs) / len(subs) if subs else float("nan")
    if (d / "continuations.jsonl").exists() and (d / "judge.refusal.continuations.jsonl").exists():
        cont = [c for c in map(json.loads, (d / "continuations.jsonl").open()) if c["status"] == "scored"]
        ref = {v["key"]: v["refusal"] for v in map(json.loads, (d / "judge.refusal.continuations.jsonl").open())}
        reverb = {v["key"]: v["vea"] for v in map(json.loads, (d / "judge.vea_cal.continuations.jsonl").open())} if (d / "judge.vea_cal.continuations.jsonl").exists() else {}
        s["interventions"] = {}
        for exp, pairs in [("insert", [("vea", "none"), ("placebo", "none"), ("vea", "placebo")]), ("remove", [("keep", "remove")])]:
            bases = {}
            for c in cont:
                if c["experiment"] == exp and ref.get(c["key"]) is not None:
                    bases.setdefault((c["qid"], c["idx"]), {}).setdefault(c["condition"], []).append((ref[c["key"]], reverb.get(c["key"])))
            if not bases:
                continue
            per_cond = {cond: boot(list(bases.values()), lambda bs, cond=cond: rate([[r for r, _ in b.get(cond, [])] for b in bs])) for cond in {c for b in bases.values() for c in b}}
            diffs = {}
            for a, b_ in pairs:
                diffs[f"{a} - {b_}"] = boot(list(bases.values()), lambda bs, a=a, b_=b_: float(np.nanmean([np.mean([r for r, _ in b[a]]) - np.mean([r for r, _ in b[b_]]) for b in bs if b.get(a) and b.get(b_)])))
            if exp == "remove":
                diffs["keep - remove (re-verbalizing removed)"] = boot(list(bases.values()), lambda bs: float(np.nanmean([np.mean([r for r, _ in b["keep"]]) - np.mean([r for r, v in b["remove"] if not v]) for b in bs if b.get("keep") and any(not v for _, v in b.get("remove", []))])))
            s["interventions"][exp] = {"n_bases": len(bases), "n_continuations": sum(len(v) for b in bases.values() for v in b.values()), "refusal_by_condition": per_cond, "differences": diffs}
    return s


def bar(ax, i, x, est, label, color):
    m, lo, hi = est
    ax.bar(x, 100 * m, 0.34, color=color, label=label)
    ax.errorbar(x, 100 * m, [[100 * (m - lo)], [100 * (hi - m)]], color="#c3c2b7", capsize=5, lw=1.5)
    ax.text(x, 100 * hi + 0.8, f"{100 * m:.1f}", ha="center", va="bottom", fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--max-idx", type=int, default=5, help="rollout indices below this: the primary design uses 0-4 for both models")
    args = ap.parse_args()
    S = {name: load(name, args.max_idx) for name in args.names}
    (OUT / f"summary.first{args.max_idx}.json").write_text(json.dumps(S, indent=1, default=lambda o: o.item()))
    for name, s in S.items():
        print(f"== {name}: {s['n_valid']} valid of {s['n_scored']} scored of {s['n_rollouts']} ({s['status']}), median reasoning {s['median_reasoning_tokens']:.0f} tokens")
        print(f"   VEA {s['n_vea']}/{s['n_valid']} = {100 * s['vea_rate'][0]:.2f}% [{100 * s['vea_rate'][1]:.2f}, {100 * s['vea_rate'][2]:.2f}] on {s['questions_with_vea']} questions; quote_check {s['quote_check']}; subcategories {s.get('vea_subcategories')}; safety share {s.get('safety_share_of_vea')}")
        print(f"   refusal {100 * s['refusal_rate'][0]:.1f}% [{100 * s['refusal_rate'][1]:.1f}, {100 * s['refusal_rate'][2]:.1f}]; within-question diff {100 * s['within_q_refusal_diff'][0]:+.1f} pp [{100 * s['within_q_refusal_diff'][1]:+.1f}, {100 * s['within_q_refusal_diff'][2]:+.1f}] over {s['n_informative_questions']} questions")
        for exp, iv in s.get("interventions", {}).items():
            print(f"   {exp}: {iv['n_bases']} bases, {iv['n_continuations']} scored continuations; refusal " + ", ".join(f"{c} {100 * e[0]:.1f}%" for c, e in iv["refusal_by_condition"].items()))
            for k, e in iv["differences"].items():
                print(f"      {k}: {100 * e[0]:+.1f} pp [{100 * e[1]:+.1f}, {100 * e[2]:+.1f}]")

    plt.rcParams.update({"figure.facecolor": "#1a1a19", "axes.facecolor": "#1a1a19", "savefig.facecolor": "#1a1a19", "text.color": "#f0efec", "axes.labelcolor": "#f0efec",
                         "xtick.color": "#c3c2b7", "ytick.color": "#c3c2b7", "axes.edgecolor": "#55554f", "axes.grid": True, "grid.color": "#2e2e2c", "axes.axisbelow": True, "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["vea_rate", "refusal_rate"], ["verbalized eval awareness (VEA) in the CoT", "refusal (judge on the first 384 answer tokens)"]):
        for i, name in enumerate(args.names):
            bar(ax, i, i, S[name][metric], LABELS[name], COLORS[name])
        ax.set_xticks(range(len(args.names)), [LABELS[n].split(" (")[0] for n in args.names])
        ax.set_ylabel("% of valid rollouts")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(100 * S[n][metric][2] for n in args.names) * 1.3 + 1)
    fig.suptitle(f"Fortress adversarial prompts, rollouts 0-{args.max_idx - 1} per question, reasoning_effort medium, judges Qwen3.8-27B (vea_cal / F2)")
    fig.text(0.01, 0.01, "95% percentile CIs, 2,000 bootstrap draws over questions. Methodology: github.com/johntrob14/eval_awareness_scaling (unified protocol).", fontsize=9, color="#c3c2b7")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / f"fortress_vea_refusal{'' if args.max_idx == 5 else f'_first{args.max_idx}'}.png", dpi=130)

    if all("interventions" in S[n] for n in args.names):
        fig, ax = plt.subplots(figsize=(10, 5))
        keys = ["vea - none", "placebo - none", "vea - placebo", "keep - remove (re-verbalizing removed)"]
        exp_of = lambda k: "remove" if k.startswith("keep") else "insert"
        shown = [k for k in keys if any(k in S[n]["interventions"].get(exp_of(k), {}).get("differences", {}) for n in args.names)]
        for i, name in enumerate(args.names):
            for j, k in enumerate(shown):
                iv = S[name]["interventions"].get(exp_of(k), {})
                if k not in iv.get("differences", {}):
                    continue
                m, lo, hi = iv["differences"][k]
                x = j + (i - 0.5) * 0.36
                ax.bar(x, 100 * m, 0.34, color=COLORS[name], label=LABELS[name] if j == 0 else None)
                ax.errorbar(x, 100 * m, [[100 * (m - lo)], [100 * (hi - m)]], color="#c3c2b7", capsize=5, lw=1.5)
                ax.text(x, 100 * max(hi, 0) + 0.5, f"{100 * m:+.1f}\nn {iv['n_bases']}", ha="center", va="bottom", fontsize=10)
        ax.axhline(0, color="#55554f", lw=1)
        ax.set_xticks(range(len(shown)), [k.replace(" - ", " minus ").replace(" (re-verbalizing removed)", "\n(re-verbalizing removed)") for k in shown])
        ax.set_ylabel("refusal difference (pp)")
        ax.set_title("CoT sentence interventions: change in refusal rate")
        ax.grid(axis="x", visible=False)
        lo_all = min(S[n]["interventions"].get(e, {}).get("differences", {}).get(k, (0, 0, 0))[1] for n in args.names for e in ("insert", "remove") for k in shown)
        hi_all = max(S[n]["interventions"].get(e, {}).get("differences", {}).get(k, (0, 0, 0))[2] for n in args.names for e in ("insert", "remove") for k in shown)
        ax.set_ylim(100 * lo_all - 2, 100 * hi_all + 9)
        ax.legend(frameon=False, loc="upper left")
        fig.text(0.01, 0.01, "Insertion at the sentence boundary nearest 40% of the CoT, 8 paired continuations per base: the post's VEA sentence vs its manipulation minimal pair (placebo).\nRemoval: the model's own VEA sentence kept vs cut, continuations that re-verbalize VEA dropped. 95% CIs bootstrapped over bases; n = bases.", fontsize=9, color="#c3c2b7")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(FIGS / "fortress_interventions.png", dpi=130)
    print("wrote", OUT / f"summary.first{args.max_idx}.json", "and figures")


if __name__ == "__main__":
    main()
