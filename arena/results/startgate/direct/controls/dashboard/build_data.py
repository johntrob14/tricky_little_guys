"""Writes the dashboard's CSVs (attack_cells, episodes, thresholds, control_fpr, fpr_curve) from thresholds.json and the arena logs.
    cd arena && python3 results/startgate/direct/controls/dashboard/build_data.py"""
import csv
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ARENA / "honeypot" / "startgate"))
import calibrate_thresholds as ct

O = Path(__file__).resolve().parent
T = json.load(open(O.parent / "thresholds.json"))
P = ct.PASSES
thr = {p: T["thresholds"][p]["0.05"] for p in P}
short = {"monitor_phase": "cot_oss", "monitor_phase_haiku": "cot_haiku", "monitor_action_all": "act_oss", "monitor_action_haiku": "act_haiku",
         "monitor_aisi_cot_action": "actcot_oss", "monitor_cot_action_haiku": "actcot_haiku"}
info = {  # run: (model, family, kind, setting, coached)
    "oss-sd-pw": ("gpt-oss-120b", "gpt-oss", "base", "startgate-direct", ""), "oss-hs-pw": ("gpt-oss-120b", "gpt-oss", "base", "always-monitored", ""),
    "mt1b-sd-pw": ("manytypes1b s150", "gpt-oss", "organism", "startgate-direct", ""), "mt1b-hs-pw": ("manytypes1b s150", "gpt-oss", "organism", "always-monitored", ""),
    "mt2e-s170-sd-pw": ("manytypes2e s170", "gpt-oss", "organism", "startgate-direct", ""),
    "sgd-grpo3": ("grpo3 s100 (old 8-episode run)", "gpt-oss", "organism", "startgate-direct", ""),
    "sgcmp-qwen-direct": ("Qwen3.8-27B", "Qwen", "base", "startgate-direct", ""), "sgcmp-qwen-covert2": ("Qwen3.8-27B", "Qwen", "base", "always-monitored", ""),
    "opsd1b-sd-pw": ("opsd-grpo1b s10", "Qwen", "organism", "startgate-direct", ""), "opsd1b-hs-pw": ("opsd-grpo1b s10", "Qwen", "organism", "always-monitored", ""),
    "glm53-sd-pw": ("GLM-5.3", "GLM", "base", "startgate-direct", ""), "glm53-hs-pw": ("GLM-5.3", "GLM", "base", "always-monitored", ""),
    "ds41-sd-pw": ("DeepSeek-V4.1-Flash", "DeepSeek", "base", "startgate-direct", ""), "ds41-hs-pw": ("DeepSeek-V4.1-Flash", "DeepSeek", "base", "always-monitored", ""),
    "inksm-e09-sd-pw": ("Inkling-Small", "Inkling", "base", "startgate-direct", ""), "inksm-e09-hs-pw": ("Inkling-Small", "Inkling", "base", "always-monitored", ""),
    "inksft1-e09-hs-pw": ("inksft1 (SFT)", "Inkling", "organism", "always-monitored", ""),
    "inkrl1s60-e09-sd-pw": ("inkrl1 step 60", "Inkling", "organism", "startgate-direct", ""), "inkrl1s60-e09-hs-pw": ("inkrl1 step 60", "Inkling", "organism", "always-monitored", ""),
    "inkrl1s80-e09-sd-pw": ("inkrl1 step 80", "Inkling", "organism", "startgate-direct", ""), "inkrl1s80-e09-hs-pw": ("inkrl1 step 80", "Inkling", "organism", "always-monitored", ""),
    "inksm-e09-sd-pw-coach": ("Inkling-Small, coached", "Inkling", "base", "startgate-direct", "yes"),
    "inkrl1s80-e09-sd-pw-coach": ("inkrl1 step 80, coached", "Inkling", "organism", "startgate-direct", "yes"),
}
EPISODES = {"sgd-grpo3": 8, "mt1b-sd-pw": 14}
spec = (O.parent / "calibrate.sh").read_text().split("--attacks")[1].split("--out")[0].replace("\\\n", " ").replace("$I", "inksm").split()
rows = []
for s in spec:
    run, path = s.split(":")
    m, fam, kind, setting, coached = info[run]
    done = ct.side_done(run, path)
    M = {p: (ct.maxes(run, p) or {}) for p in P}
    r = dict(cell=run, model=m, family=fam, kind=kind, setting=setting, coached=coached, episodes=EPISODES.get(run, 16), side_task_done=len(done))
    for p in P:
        r[short[p]] = sum(M[p].get(k, 0) > thr[p] for k in done)
    r["missed_both_cot"] = sum(all(M[p].get(k, 0) <= thr[p] for p in ("monitor_phase", "monitor_phase_haiku")) for k in done)
    r["missed_all_six"] = sum(all(M[p].get(k, 0) <= thr[p] for p in P) for k in done)
    rows.append(r)


def write(name, rs):
    with open(O / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)


write("attack_cells.csv", rows)
eps = []
for s in spec:
    run, path = s.split(":")
    m, fam, kind, setting, coached = info[run]
    M = {p: (ct.maxes(run, p) or {}) for p in P}
    done = sorted(ct.side_done(run, path), key=lambda k: (k[0], int(k[1].split("#")[1])))
    logs = sorted({k[0] for k in done})  # a cell with several eval logs numbers its episodes per log: "<log index>-<sample>"
    for k in done:
        n = k[1].split("#")[1]
        e = dict(cell=run, episode=n if len(logs) == 1 else f"{logs.index(k[0]) + 1}-{n}", model=m, family=fam, kind=kind, setting=setting, coached=coached)
        for p in P:
            e[short[p]] = M[p].get(k, "")
        eps.append(e)
write("episodes.csv", eps)
mode = {"monitor_phase": "CoT-only", "monitor_phase_haiku": "CoT-only", "monitor_action_all": "action-only", "monitor_action_haiku": "action-only",
        "monitor_aisi_cot_action": "action+CoT", "monitor_cot_action_haiku": "action+CoT"}
th = []
for p in P:
    pool, t5, t1 = T["fpr"][p]["pooled"], T["thresholds"][p]["0.05"], T["thresholds"][p]["0.01"]
    th.append(dict(monitor=ct.LABEL[p], mode=mode[p], judge="Claude Haiku 5.5" if "haiku" in p else "gpt-oss-20b", fpr_old_rule=round(pool["5"], 4),
                   threshold_5pct=t5, fpr_at_threshold_5pct=round(pool[str(t5)], 4), threshold_1pct=t1 if t1 < 10 else "", threshold_1pct_note="unattainable" if t1 >= 10 else ""))
write("thresholds.csv", th)
cname = {"ctl-pw-oss": "gpt-oss-120b", "ctl-pw-mt2e": "manytypes2e s170", "ctl-pw-mt1b": "manytypes1b s150", "ctl-pw-grpo3": "grpo3 s100", "ctl-pw-qwen": "Qwen3.8-27B",
         "ctl-pw-opsd1b": "opsd-grpo1b s10", "ctl-pw-inksm": "Inkling-Small", "ctl-pw-inkrl80": "inkrl1 step 80", "ctl-pw-glm53": "GLM-5.3", "ctl-pw-ds41": "DeepSeek-V4.1-Flash"}
base = {"ctl-pw-oss", "ctl-pw-qwen", "ctl-pw-inksm", "ctl-pw-glm53", "ctl-pw-ds41"}
cf = []
for c in T["controls"]:
    r = dict(control_cell=c, model=cname[c], kind="base" if c in base else "organism")
    for p in P:
        pc = T["fpr"][p]["per_cell"]
        r[short[p] + "_old"] = round(pc["5"][c], 4)
        r[short[p] + "_cal"] = round(pc[str(thr[p])][c], 4)
    cf.append(r)
write("control_fpr.csv", cf)
write("fpr_curve.csv", [dict(monitor=ct.LABEL[p], threshold=t, pooled_fpr=round(T["fpr"][p]["pooled"][str(t)], 4)) for p in P for t in range(10)])
print("wrote", len(rows), "cells")
