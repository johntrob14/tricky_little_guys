"""Pivot the leak-turn scores (leak_build.py -> leakturns.json) into one row per cell with a column per monitor, using the weak/strong names the dashboard shows."""
import json
key = {"CoT-only, gpt-oss-20b (weak)": "weak_cot", "CoT-only, claude-haiku-5.5 (strong)": "strong_cot", "Action-only, gpt-oss-20b": "weak_act",
       "CoT+action, UK AISI prompt (gpt-oss-20b)": "weak_ca", "CoT+action, claude-haiku-5.5": "strong_ca", "Action-only, claude-haiku-5.5": "strong_act"}
wide = {}
for r in json.load(open("leakturns.json")):
    w = wide.setdefault((r["model"], r["setting"]), dict(model=r["model"], setting=r["setting"], leak_turns=r["leak_turns"]))
    w[key[r["monitor"]]] = f"{r['flagged']}/{r['scored']}"
json.dump(list(wide.values()), open("leakturns_wide.json", "w"))
