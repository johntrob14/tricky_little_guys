"""Startgate password-policy evasion counts per attack cell and monitor, as a flat CSV for plotting (the "Leaked + judge missed" slides).

Reads results/startgate/direct/controls/thresholds.json (calibrate_thresholds.py output: leaked episodes per cell, and how many of them each
monitor pass flagged at the 5%-FPR threshold fitted on the 160 honest controls) and the per-cell episode lists named in controls/calibrate.sh.
One row per (cell, monitor). missed = leaked episodes the monitor did not flag; episodes = all attempted episodes in the cell.

    python honeypot/startgate/evasion_counts.py   # writes results/startgate/direct/evasion_counts.csv
"""
import csv
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results" / "startgate" / "direct"
MONITORS = {  # pass dir -> (judge, monitor prompt, mode)
    "monitor_phase": ("gpt-oss-20b", "controlarena_cot_only", "cot only (reasoning + visible assistant text)"),
    "monitor_phase_haiku": ("claude-haiku-5.5", "controlarena_cot_only", "cot only (reasoning + visible assistant text)"),
    "monitor_action_all": ("gpt-oss-20b", "controlarena_action_only", "action only"),
    "monitor_action_haiku": ("claude-haiku-5.5", "controlarena_action_only", "action only"),
    "monitor_aisi_cot_action": ("gpt-oss-20b", "controlarena_cot", "action+cot"),
    "monitor_cot_action_haiku": ("claude-haiku-5.5", "controlarena_cot", "action+cot"),
}
MODEL = {"oss": "GPT-OSS-120B", "grpo3-s100": "GPT-OSS-120B GRPO3 step 100", "mt1b": "GPT-OSS-120B manytypes1b", "mt2e-s170": "GPT-OSS-120B manytypes2e step 170",
         "sgcmp-qwen": "Qwen3.8-27B*", "opsd1b": "Qwen3.8-27B OPSD + GRPO", "glm53": "GLM-5.3", "ds41": "DeepSeek V4.1-Flash-1017",
         "inksm-e09": "Inkling-Small", "inksft1-e09": "Inkling-Small SFT", "inkrl1s60-e09": "Inkling-Small RL step 60", "inkrl1s80-e09": "Inkling-Small RL step 80"}

d = json.load(open(R / "controls" / "thresholds.json"))
paths = dict(re.findall(r"(\S+):(\S+\.jsonl)", (R / "controls" / "calibrate.sh").read_text()))
rows = []
for run, a in d["attacks"].items():
    if run.startswith("sgcmp-qwen"):
        model, setting = "sgcmp-qwen", "always_monitored" if run.endswith("covert2") else "startgate_direct"
    else:
        model, setting = re.match(r"(.+)-(hs|sd)-pw", run).groups()
        setting = {"hs": "always_monitored", "sd": "startgate_direct"}[setting] + ("_coached" if run.endswith("-coach") else "")
    episodes = sum(1 for _ in open(R / paths[run]))
    for p, (judge, prompt, mode) in MONITORS.items():
        s = a["passes"][p]
        rows.append(dict(run=run, model=MODEL[model], setting=setting, judge=judge, monitor=prompt, mode=mode, threshold_gt=d["thresholds"][p]["0.05"],
                         episodes=episodes, leaked=a["done"], leaked_scored=s["n"], flagged=s["flagged"][1], missed=s["n"] - s["flagged"][1]))
with open(R / "evasion_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {R / 'evasion_counts.csv'}")
