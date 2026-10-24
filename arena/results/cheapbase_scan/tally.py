"""Per-cell counts and CoT-monitor scores for the cheapbase_scan models and the DeepSeek / Qwen3.8 references: main pass, leaked, errors, turns, reasoning,
monitor (controlarena_cot_only, gpt-oss-20b, main task only) per-episode max and mean, AUROC of each attack tier against the control. Writes tally.json beside this file.
Counts and lengths only; no transcript text is printed.

    .venv/bin/python results/cheapbase_scan/tally.py
"""
import glob, json, statistics as st
from pathlib import Path
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning

MODELS = [("ds41or-think", "DeepSeek V4.1 Flash 1017"), ("qwenor-think", "Qwen3.8-27B"), ("inksm", "Inkling-Small"), ("nemosvl", "Nemotron 3.5 Super VL"), ("qwen36", "Qwen3.6-35B-A3B")]
CELLS = [("easy", "covert2-easy"), ("medium", "covert2-medium"), ("hard", "covert2-hard"), ("control", "ctl-easy")]


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def cell(run):
    mon = {e["sample"]: e for e in json.load(open(glob.glob(f"logs/{run}/monitor_maintask_cot_only/*/summary.json")[0]))["episodes"]}
    eps, tok = [], [0, 0, 0]
    for f in sorted(glob.glob(f"logs/{run}/*.eval")):
        log = read_eval_log(f)
        for u in log.stats.model_usage.values():
            tok[0] += u.input_tokens; tok[1] += u.input_tokens_cache_read or 0; tok[2] += u.output_tokens
        for s in log.samples:
            sc = next(iter(s.scores.values())).metadata or {} if s.scores else {}
            turns = [m for m in s.messages if m.role == "assistant"]
            rlen = [sum(len(c.reasoning) for c in m.content if isinstance(c, ContentReasoning)) if isinstance(m.content, list) else 0 for m in turns]
            m = mon.get(str(s.id), {})
            eps.append(dict(main=bool(sc.get("main_pass")), leak=bool(sc.get("side_leaked")), err=s.error is not None, turns=len(turns), rlen=rlen, max=m.get("max"), mean=m.get("mean")))
    return eps, tok


out = {}
print("| model | cell | n | main pass | leaked | errors | assistant turns/ep | turns with reasoning | median reasoning chars/turn | monitor ep max, mean | monitor ep mean, mean | AUROC vs control (ep max) | AUROC vs control (ep mean) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for prefix, name in MODELS:
    data = {c: cell(f"{prefix}-{suffix}") for c, suffix in CELLS}
    ctl = [e for e in data["control"][0] if e["max"] is not None]
    out[prefix] = {"name": name, "cells": {}}
    for c, (eps, tok) in data.items():
        scored = [e for e in eps if e["max"] is not None]
        rl = [x for e in eps for x in e["rlen"]]
        row = dict(n=len(eps), main=sum(e["main"] for e in eps), leaked=sum(e["leak"] for e in eps), errors=sum(e["err"] for e in eps),
                   turns_per_ep=st.mean(e["turns"] for e in eps), turns_with_reasoning=sum(x > 0 for x in rl), turns=len(rl), median_reasoning_chars=st.median(rl),
                   n_monitored=len(scored), ep_max=[e["max"] for e in scored], ep_mean=[e["mean"] for e in scored], tokens=dict(input=tok[0], cache_read=tok[1], output=tok[2]),
                   auroc_max=None if c == "control" else auroc([e["max"] for e in scored], [e["max"] for e in ctl]),
                   auroc_mean=None if c == "control" else auroc([e["mean"] for e in scored], [e["mean"] for e in ctl]))
        out[prefix]["cells"][c] = row
        n = row["n"]
        au = "" if c == "control" else f"{row['auroc_max']:.3f} | {row['auroc_mean']:.3f}"
        print(f"| {name} | {c} | {n} | {row['main']}/{n} | {row['leaked']}/{n} | {row['errors']}/{n} | {row['turns_per_ep']:.1f} | {row['turns_with_reasoning']}/{row['turns']} | {row['median_reasoning_chars']:.0f} | "
              f"{st.mean(row['ep_max']):.2f} (n={len(scored)}) | {st.mean(row['ep_mean']):.2f} (n={len(scored)}) | {au or ' | '} |")
json.dump(out, open(Path(__file__).with_name("tally.json"), "w"), indent=1)
