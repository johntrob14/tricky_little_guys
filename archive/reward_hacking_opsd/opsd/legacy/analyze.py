"""Analysis of the step-2 eval grid (+ step 0) -> markdown tables, a JSON dump of every number, examples file.

Metric: strict type-hint rate = grade.success_strict (hint_score == 100; no code block, unparsable code,
grader error and truncated responses all count as "no"; denominator = all responses), Wilson 95% CI.
Also: mean hint_score among responses whose code was graded (hint_score not None), counts of no-code /
missing grades and truncation (finish_reason == "length").
Differences of proportions: Newcombe hybrid score interval (Newcombe 1998, method 10) for two independent
proportions. Pooled robustness rate: Wilson CI (treats the 960 responses as independent) and a cluster
bootstrap over the 64 questions (each question appears under all 15 templates), seed 0, 2000 resamples.

Condition labels:
  step0 rows teacher=False             -> sdf_only            (main grid)
  step0 rows teacher=True              -> sdf_only+teacher    (main grid, reference only)
  eval rows lora_name=base on SDF model -> sdf_only (robust) / sdf_only_rerun (main, when step0 is given)
  eval rows lora_name=base, other model -> base:<model dir name>
  eval rows with a LoRA                -> lora_name (rename with --label old=new)

python -m opsd.analyze --step0 results/step0.jsonl --eval runs/eval/grid_sdf.jsonl runs/eval/grid_nemo.jsonl \
    --out runs/eval/analysis.md --examples runs/eval/examples.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import OrderedDict, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = 1.959963984540054
MAIN_CTX = ["wood_neutral", "wood_deploy", "eval_eval", "deploy_deploy"]
GAPS = [("eval_eval", "deploy_deploy"), ("wood_neutral", "deploy_deploy"), ("wood_deploy", "deploy_deploy")]
ORDER_HINT = ["sdf_only", "sdf_only_rerun", "sdf_only+teacher", "opsd", "cd", "ei1", "ei4"]


# ---------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = Z):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1: int, n1: int, k2: int, n2: int):
    """p1 - p2 with the Newcombe hybrid score 95% CI (independent samples)."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def cluster_boot(rows, key="qid", B=2000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r.get(key)].append(int(r["_s"]))
    ks = list(by)
    if len(ks) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    est = []
    for _ in range(B):
        s = n = 0
        for _ in ks:
            v = by[ks[rng.randrange(len(ks))]]
            s += sum(v)
            n += len(v)
        est.append(s / n)
    est.sort()
    return est[int(0.025 * B)], est[int(0.975 * B) - 1]


def pct(x):
    return "nan" if x != x else f"{100 * x:.1f}"


def fmt_ci(p, lo, hi):
    return f"{pct(p)} [{pct(lo)}, {pct(hi)}]"


def fmt_diff(d, lo, hi):
    return f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


# ---------------------------------------------------------------- loading
def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def strict(r) -> bool:
    g = r.get("grade") or {}
    return bool(g.get("success_strict"))


def label_rows(rows, src, have_step0, renames):
    out = []
    for r in rows:
        r = dict(r)
        grid = r.get("grid") or ("robust" if str(r.get("context", "")).startswith("DD") else "main")
        r["grid"] = grid
        if src == "step0":
            lab = "sdf_only+teacher" if r.get("teacher") else "sdf_only"
        elif r.get("lora_name") in (None, "base"):
            m = os.path.basename(str(r.get("model", "")).rstrip("/"))
            if "sdf" in m:
                lab = "sdf_only_rerun" if (grid == "main" and have_step0) else "sdf_only"
            else:
                lab = f"base:{m}"
        else:
            lab = r["lora_name"]
        r["_label"] = renames.get(lab, lab)
        r["_src"] = src
        r["_s"] = strict(r)
        out.append(r)
    return out


def sort_labels(labels):
    def key(l):
        for i, h in enumerate(ORDER_HINT):
            if l == h:
                return (i, 0, l)
        for i, h in enumerate(ORDER_HINT):
            if l.startswith(h):
                return (i, 1, l)
        return (len(ORDER_HINT), 0, l)
    return sorted(labels, key=key)


def cell_stats(rows):
    n = len(rows)
    k = sum(r["_s"] for r in rows)
    p, lo, hi = wilson(k, n)
    hs = [(r.get("grade") or {}).get("hint_score") for r in rows]
    hs = [h for h in hs if h is not None]
    return OrderedDict(
        n=n, k=k, rate=p, lo=lo, hi=hi,
        mean_hint_score_with_code=(sum(hs) / len(hs) if hs else None), n_hint_graded=len(hs),
        n_no_code=sum(not (r.get("grade") or {}).get("code_found") for r in rows),
        n_missing_grade=n - len(hs),
        n_truncated=sum(r.get("finish_reason") == "length" for r in rows),
        n_truncated_and_hinted=sum(r.get("finish_reason") == "length" and r["_s"] for r in rows),
        mean_tokens=(sum(r.get("n_tokens", 0) for r in rows) / n if n else None))


# ---------------------------------------------------------------- report
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--step0", default=os.path.join(REPO, "results", "step0.jsonl"),
                   help="step-0 jsonl ('' to skip)")
    p.add_argument("--eval", nargs="*", default=None, help="eval_gen outputs (default runs/eval/*.jsonl)")
    p.add_argument("--label", action="append", default=[], help="rename condition old=new (repeatable)")
    p.add_argument("--out", default=os.path.join(REPO, "runs", "eval", "analysis.md"))
    p.add_argument("--json", default=None, help="default: <out>.json")
    p.add_argument("--examples", default=None, help="default: <out dir>/examples.md")
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref", default="sdf_only", help="reference condition for the vs-reference table")
    a = p.parse_args(argv)
    renames = dict(s.split("=", 1) for s in a.label)
    evals = a.eval if a.eval is not None else sorted(glob.glob(os.path.join(REPO, "runs", "eval", "*.jsonl")))
    a.json = a.json or os.path.splitext(a.out)[0] + ".json"
    a.examples = a.examples or os.path.join(os.path.dirname(os.path.abspath(a.out)), "examples.md")

    rows, sources = [], []
    have_step0 = bool(a.step0) and os.path.exists(a.step0)
    if have_step0:
        rows += label_rows(read_jsonl(a.step0), "step0", True, renames)
        sources.append(a.step0)
    for f in evals:
        rows += label_rows(read_jsonl(f), os.path.basename(f), have_step0, renames)
        sources.append(f)
    # duplicates (same condition, prompt id, sample) across files: keep the last file's row, report it
    seen, dups = {}, 0
    for r in rows:
        key = (r["_label"], r["id"], r.get("sample", 0))
        dups += key in seen
        seen[key] = r
    rows = list(seen.values())
    ungraded = sum(r.get("grade") is None for r in rows)

    cells = defaultdict(list)
    for r in rows:
        cells[(r["_label"], r["grid"], r["context"])].append(r)
    labels = sort_labels({r["_label"] for r in rows})
    main_labels = [l for l in labels if any((l, "main", c) in cells for c in MAIN_CTX)]
    rob_ctx = sorted({c for (_, g, c) in cells if g == "robust"}, key=lambda c: int(c[2:]) if c[2:].isdigit() else 99)
    rob_labels = [l for l in labels if any((l, "robust", c) in cells for c in rob_ctx)]
    ctx_label, dd_text = {}, {}
    try:
        ctx_label = {k: v["label"] for k, v in json.load(open(os.path.join(REPO, "spec", "contexts.json")))["contexts"].items()}
        dd_text = {d["id"]: d["template"].replace("{question}", "{Q}").replace("\n", "\\n")
                   for d in json.load(open(os.path.join(REPO, "spec", "declarative_deploy.json")))["prompts"]}
    except Exception:
        pass

    J = OrderedDict(sources=sources, n_rows=len(rows), duplicates_replaced=dups, ungraded=ungraded,
                    main={}, gaps={}, robust={}, robust_vs_wood={}, vs_ref={})
    L = ["# Step 2 eval analysis", "",
         f"Sources: {', '.join(f'`{s}`' for s in sources)}. Rows: {len(rows)}; duplicate rows replaced by a later "
         f"file: {dups}; rows without a grade: {ungraded} (counted as no type hints).", "",
         "Strict type-hint rate = hint_score == 100 (full annotation of every def), with no code, unparsable code "
         "and truncated responses counted as no; denominator all responses. Brackets: 95% CI (Wilson for rates, "
         "Newcombe hybrid score for differences of independent proportions). Values in percent.", ""]

    # main grid
    if main_labels:
        L += ["## Main grid (no teacher lines, 128 BCB questions per context)", "",
              "| condition | " + " | ".join(ctx_label.get(c, c) for c in MAIN_CTX) + " |",
              "|---|" + "---|" * len(MAIN_CTX)]
        for l in main_labels:
            cs = []
            for c in MAIN_CTX:
                rr = cells.get((l, "main", c), [])
                s = cell_stats(rr)
                J["main"][f"{l}|{c}"] = s
                cs.append(f"{fmt_ci(s['rate'], s['lo'], s['hi'])} ({s['k']}/{s['n']})" if rr else "-")
            L.append(f"| {l} | " + " | ".join(cs) + " |")
        L += ["", "### Main grid details", "",
              "| condition | context | n | strict k | strict % [95% CI] | mean hint_score (n graded) | no code | "
              "truncated (of which hinted) | mean tokens |", "|---|---|---|---|---|---|---|---|---|"]
        for l in main_labels:
            for c in MAIN_CTX:
                if (l, "main", c) not in cells:
                    continue
                s = J["main"][f"{l}|{c}"]
                mh = "-" if s["mean_hint_score_with_code"] is None else f"{s['mean_hint_score_with_code']:.1f}"
                L.append(f"| {l} | {c} | {s['n']} | {s['k']} | {fmt_ci(s['rate'], s['lo'], s['hi'])} | "
                         f"{mh} ({s['n_hint_graded']}) | {s['n_no_code']} | {s['n_truncated']} "
                         f"({s['n_truncated_and_hinted']}) | {s['mean_tokens']:.0f} |")
        L += ["", "## Eval-minus-deploy gaps (main grid; difference of strict rates, percentage points)", "",
              "| condition | " + " | ".join(f"{x} − {y}" for x, y in GAPS) + " |", "|---|" + "---|" * len(GAPS)]
        for l in main_labels:
            cs = []
            for x, y in GAPS:
                rx, ry = cells.get((l, "main", x)), cells.get((l, "main", y))
                if not rx or not ry:
                    cs.append("-")
                    continue
                d = newcombe(sum(r["_s"] for r in rx), len(rx), sum(r["_s"] for r in ry), len(ry))
                J["gaps"][f"{l}|{x}-{y}"] = dict(diff=d[0], lo=d[1], hi=d[2])
                cs.append(fmt_diff(*d))
            L.append(f"| {l} | " + " | ".join(cs) + " |")
        if a.ref in main_labels:
            L += ["", f"## Change vs {a.ref} (main grid, percentage points, Newcombe CI)", "",
                  "| condition | " + " | ".join(MAIN_CTX) + " |", "|---|" + "---|" * len(MAIN_CTX)]
            for l in main_labels:
                if l == a.ref:
                    continue
                cs = []
                for c in MAIN_CTX:
                    r1, r0 = cells.get((l, "main", c)), cells.get((a.ref, "main", c))
                    if not r1 or not r0:
                        cs.append("-")
                        continue
                    d = newcombe(sum(r["_s"] for r in r1), len(r1), sum(r["_s"] for r in r0), len(r0))
                    J["vs_ref"][f"{l}|{c}"] = dict(diff=d[0], lo=d[1], hi=d[2], ref=a.ref)
                    cs.append(fmt_diff(*d))
                L.append(f"| {l} | " + " | ".join(cs) + " |")
            L.append("")
            L.append("Note: conditions on the raw Nemotron base (EI adapters) and on the SDF-merged base share prompts, "
                     "so these are different-model comparisons on identical inputs; the CI treats them as independent.")

    # robustness
    if rob_labels:
        L += ["", "## Robustness grid (Wood Labs system prompt + declarative deployment user prefix, 64 questions each)",
              "", "Per-template strict rate (%); pooled = all 15 templates (n = 960 per condition).", "",
              "| template | " + " | ".join(rob_labels) + " |", "|---|" + "---|" * len(rob_labels)]
        for c in rob_ctx:
            cs = []
            for l in rob_labels:
                rr = cells.get((l, "robust", c), [])
                s = cell_stats(rr)
                J["robust"][f"{l}|{c}"] = s
                cs.append(f"{fmt_ci(s['rate'], s['lo'], s['hi'])}" if rr else "-")
            L.append(f"| {c} | " + " | ".join(cs) + " |")
        pooled = {}
        cs, cs_b, cs_t, cs_m = [], [], [], []
        for l in rob_labels:
            rr = [r for c in rob_ctx for r in cells.get((l, "robust", c), [])]
            s = cell_stats(rr)
            s["boot_lo"], s["boot_hi"] = cluster_boot(rr, seed=a.seed)
            pooled[l] = (s["k"], s["n"])
            J["robust"][f"{l}|pooled"] = s
            cs.append(f"**{fmt_ci(s['rate'], s['lo'], s['hi'])}** ({s['k']}/{s['n']})")
            cs_b.append(f"[{pct(s['boot_lo'])}, {pct(s['boot_hi'])}]")
            cs_t.append(f"{s['n_truncated']} ({s['n_truncated_and_hinted']})")
            cs_m.append("-" if s["mean_hint_score_with_code"] is None else
                        f"{s['mean_hint_score_with_code']:.1f} ({s['n_hint_graded']})")
        L += [f"| pooled (Wilson) | " + " | ".join(cs) + " |",
              f"| pooled, question-cluster bootstrap CI | " + " | ".join(cs_b) + " |",
              f"| pooled mean hint_score (n graded) | " + " | ".join(cs_m) + " |",
              f"| pooled truncated (of which hinted) | " + " | ".join(cs_t) + " |", ""]
        if dd_text:
            L += ["Templates: " + "; ".join(f"{c} = \"{dd_text.get(c, '?')}\"" for c in rob_ctx), ""]
        # robust pooled vs plain Wood Labs contexts
        L += ["## Robustness pooled vs plain Wood Labs contexts (pooled robust − main context, percentage points)", "",
              "| condition | robust pooled | wood_neutral | robust − wood_neutral | wood_deploy | robust − wood_deploy |",
              "|---|---|---|---|---|---|"]
        for l in rob_labels:
            k1, n1 = pooled[l]
            cs = [fmt_ci(*wilson(k1, n1))]
            for c in ("wood_neutral", "wood_deploy"):
                rr = cells.get((l, "main", c))
                if not rr:
                    cs += ["-", "-"]
                    continue
                k2, n2 = sum(r["_s"] for r in rr), len(rr)
                d = newcombe(k1, n1, k2, n2)
                J["robust_vs_wood"][f"{l}|{c}"] = dict(diff=d[0], lo=d[1], hi=d[2], robust=[k1, n1], main=[k2, n2])
                cs += [fmt_ci(*wilson(k2, n2)), fmt_diff(*d)]
            L.append(f"| {l} | " + " | ".join(cs) + " |")
        L += ["", "Caveat: the Newcombe interval assumes independent samples. The 960 robust responses reuse 64 "
              "questions (the first 64 of the 128 main-grid questions) under 15 templates, so the pooled rate's "
              "effective n is smaller than 960; compare the cluster-bootstrap CI above. Robust system prompts are "
              "spread over 64 questions, main-grid ones over 128, so the system prompt paired with a question can differ.", ""]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(a.json, "w") as f:
        json.dump(J, f, indent=1)
    print(f"[analyze] {a.out}, {a.json}", flush=True)

    # examples: n per (condition, main context) and per (condition, robust pooled), seeded choice
    rng = random.Random(a.seed)
    E = ["# Example transcripts", "",
         f"{a.n_examples} per condition x context (robust grid pooled), chosen with random.Random({a.seed}).sample "
         "in the order below. `strict` = grader success_strict.", ""]
    groups = []
    for l in labels:
        for c in MAIN_CTX:
            if (l, "main", c) in cells:
                groups.append((l, c, sorted(cells[(l, "main", c)], key=lambda r: (r["id"], r.get("sample", 0)))))
        rr = [r for c in rob_ctx for r in cells.get((l, "robust", c), [])]
        if rr:
            groups.append((l, "robust (pooled)", sorted(rr, key=lambda r: (r["id"], r.get("sample", 0)))))
    for l, c, rr in groups:
        E += [f"## {l} / {c}", ""]
        for r in rng.sample(rr, min(a.n_examples, len(rr))):
            g = r.get("grade") or {}
            E += [f"### {r['id']} (strict={r['_s']}, hint_score={g.get('hint_score')}, finish={r.get('finish_reason')}, "
                  f"tokens={r.get('n_tokens')}, source={r['_src']})", "",
                  "**System:**", "", "`````text", r["system"], "`````", "", "**User:**", "", "`````text", r["user"], "`````", "",
                  "**Response:**", "", "`````text", r.get("response", ""), "`````", ""]
    with open(a.examples, "w") as f:
        f.write("\n".join(E) + "\n")
    print(f"[analyze] {a.examples} ({len(groups)} groups)", flush=True)


if __name__ == "__main__":
    main()
