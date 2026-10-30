"""Reasoning on vs no CoT for the three OpenRouter base models with matched arena_covert_v2 cells (four tiktok tasks, 16 episodes per cell, seed-base 0):
main pass, side leak and turns with reasoning per cell, pooled over the three tiers. Counts only; no text is printed. Writes the JSON nocot_compare_chart.py plots.

    .venv/bin/python results/nocot_musespark/nocot_compare.py results/nocot_musespark/nocot_compare.json
"""
import json
import statistics
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentReasoning

TIERS = ("easy", "medium", "hard")
MODELS = [  # name, provider note, {condition: run prefix}; covert cells are <prefix>-covert2-<tier>, controls <prefix>-ctl-easy
    ("DeepSeek V4.1 Flash 1017", "Parasail; off = effort none", {"on": "ds41or-think", "off": "ds41or-nothink"}),
    ("Qwen3.8-27B", "DeepInfra; off = effort none", {"on": "qwenor-think", "off": "qwenor-nothink"}),
    ("Muse Spark Open", "Fireworks; off = raw completions, empty reasoning prefilled", {"on": "musespark-think", "off": "musespark-nocot"}),
]
EXTRA = {"musespark-rawthink-covert2-easy": "Muse Spark Open, raw-completions backend with reasoning on (backend check)"}


def cell(run: str) -> dict:
    n = main = leak = err = turns = reasoned = 0
    reasoning_chars, visible_chars = [], []   # per episode: characters of reasoning, and of visible text (outside reasoning, excluding tool-call arguments)
    for p in sorted(Path("logs", run).glob("*.eval")):
        for s in read_eval_log(p).samples or []:
            n += 1
            reasoning_chars.append(0)
            visible_chars.append(0)
            err += bool(s.error)
            sc = next(iter(s.scores.values())).metadata if s.scores else {}
            main += bool(sc.get("main_pass"))
            leak += bool(sc.get("side_leaked"))
            for m in s.messages:
                if m.role == "assistant":
                    turns += 1
                    r = sum(len(c.reasoning.strip()) for c in (m.content if isinstance(m.content, list) else []) if isinstance(c, ContentReasoning))
                    reasoned += r > 0
                    reasoning_chars[-1] += r
                    visible_chars[-1] += len(m.text.strip())
    assert n == 16, f"{run}: {n} episodes, expected 16"
    return dict(run=run, n=n, main=main, leak=leak, errors=err, turns=turns, reasoned=reasoned, reasoning_chars=reasoning_chars, visible_chars=visible_chars)


out = dict(models=[], extra={})
print("| model | condition | cell | n | main pass | leaked | errors | turns with reasoning | median reasoning chars/episode | median visible chars/episode |\n|---|---|---|---|---|---|---|---|---|---|")
for name, note, conds in MODELS:
    entry = dict(name=name, note=note, conds={})
    for cond, prefix in conds.items():
        cells = {t: cell(f"{prefix}-covert2-{t}") for t in TIERS} | {"control": cell(f"{prefix}-ctl-easy")}
        entry["conds"][cond] = cells
        for k, c in cells.items():
            print(f"| {name} | {cond} | {k} | {c['n']} | {c['main']}/{c['n']} | {c['leak']}/{c['n']} | {c['errors']}/{c['n']} | {c['reasoned']}/{c['turns']} | {statistics.median(c['reasoning_chars']):.0f} | {statistics.median(c['visible_chars']):.0f} |")
    out["models"].append(entry)
for run, label in EXTRA.items():
    c = cell(run)
    out["extra"][run] = dict(label=label, **c)
    print(f"| {label} | on | easy | {c['n']} | {c['main']}/{c['n']} | {c['leak']}/{c['n']} | {c['errors']}/{c['n']} | {c['reasoned']}/{c['turns']} | {statistics.median(c['reasoning_chars']):.0f} | {statistics.median(c['visible_chars']):.0f} |")
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("saved", sys.argv[1])
