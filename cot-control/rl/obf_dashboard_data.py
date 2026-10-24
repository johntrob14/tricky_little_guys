"""Aggregate the 2026-10-23/24 DeepSeek obf experiments into the datasets of the "DeepSeek obfuscation runs" dashboard (https://claude.ai/artifact/KJk6widkn2nkGboPGr3Jd7). No problem or CoT text.

    python3 obf_dashboard_data.py ../results <out dir>
"""
import json, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1]); OUT = Path(sys.argv[2])
OBF = ["ignore_question", "misdirect", "answer_hiding", "cover_story", "sad", "pirate", "happy"]
EASY = ["max_words", "shorthand", "no_backtracking", "word_suppression", "alternating_case"]
rows_of = lambda p: [json.loads(l) for l in open(p)]
full = lambda r: bool(r.get("strict")) and r["fmt_ok"] and r.get("meta") is True

# held-out evals (60 eval-pool problems, no prefill)
EVALS = [("start", "obf2e step 10 (after prefill curriculum)", "DeepSeek V4.1 Flash", ["obf2e-s10-e25"]),
         ("rl_final", "obf2g step 25 (final RL)", "DeepSeek V4.1 Flash", ["obf2g-s25-e25"]),
         ("sft1", "sft1 (SFT, 4 epochs)", "DeepSeek V4.1 Flash", ["sft1-ep4-e25"]),
         ("sft2", "sft2 (SFT, 12 epochs)", "DeepSeek V4.1 Flash", ["sft2-ep8-e25", "sft2-ep8-e25-more"]),
         ("gptoss_mt", "gpt-oss many-types (RL only, reference)", "gpt-oss-120b", ["mt1b-s150-obf"])]
held = []
for cid, label, model, dirs in EVALS:
    rows = [r for d in dirs for r in rows_of(R / "obf_eval" / d / "rows.jsonl")]
    for t in OBF + EASY + ["baseline"]:
        x = [r for r in rows if r["type"] == t]
        if not x: continue
        ok = [r for r in x if r["fmt_ok"]]
        held.append(dict(checkpoint=cid, label=label, model=model, type=t,
                         group="never trained (held-out type)" if t == "happy" else "obfuscating" if t in OBF else "baseline" if t == "baseline" else "easy",
                         n=len(x), partial=round(sum((r["partial"] or 0) for r in x) / len(x), 3), full=sum(full(r) for r in x),
                         mentions=sum(r.get("meta") is False for r in ok), n_fmt_ok=len(ok),
                         mention_rate=round(sum(r.get("meta") is False for r in ok) / len(ok), 3) if ok else None,
                         correct=sum(r["correct"] for r in x), accuracy=round(sum(r["correct"] for r in x) / len(x), 3),
                         truncated=sum(r["truncated"] for r in x)))
json.dump(held, open(OUT / "heldout.json", "w"), indent=0)

# training steps along the lineage (obf2 -> obf2c -> obf2d -> obf2e step 10 -> obf2f -> obf2g) and side branches
LIN = [("obf2", range(1, 6)), ("obf2c", range(6, 8)), ("obf2d", range(8, 10)), ("obf2e", range(10, 11)), ("obf2f", range(11, 21)), ("obf2g", range(21, 26))]
SIDE = [("obf2b", range(6, 8)), ("obf2e", range(11, 13))]
CFG = {"obf2": dict(lr=4e-5, effort=75, floor=0, band="0.5-2.0"), "obf2b": dict(lr=4e-5, effort=75, floor=3, band="0.5-2.0"),
       "obf2c": dict(lr=4e-5, effort=75, floor=2, band="0.5-3.0"), "obf2d": dict(lr=4e-5, effort=75, floor=2, band="0.5-3.0"),
       "obf2e": dict(lr=4e-5, effort=75, floor=2, band="0.5-3.0"), "obf2f": dict(lr=4e-5, effort=25, floor=2, band="0.5-3.0"),
       "obf2g": dict(lr=8e-5, effort=25, floor=2, band="0.5-3.0")}
train, obft = [], []
prev = {}
for lineage, segs in ((True, LIN), (False, SIDE)):
    for run, steps in segs:
        S = {s["step"]: s for s in rows_of(R / "rl" / run / "steps.jsonl")}
        roll = rows_of(R / "rl" / run / "rollouts.jsonl")
        for k in steps:
            s = S[k]; spent_prev = S[k - 1]["spent_est"] if (k - 1) in S else 0.0
            row = dict(step=k, run=run, lineage=lineage, **CFG[run], p_prefill=s.get("p_prefill", 0.0), mu=s["mu"],
                       strict_credited=s["strict_credited"], comp=s["comp"], mention_rate=round(s["meta_found"] / s["meta_asked"], 3),
                       acc_vs_base=s["acc_vs_base"], len_pen=s["len_pen"], groups_comp_var=s.get("groups_comp_var"), step_cost=round(s["spent_est"] - spent_prev, 2))
            for t in EASY:
                b = s["by_type"].get(t); row[t] = b["partial"] if b else None
            train.append(row)
            for arm in ("prefilled", "plain"):
                x = [r for r in roll if r["step"] == k and r["type"] in OBF and bool(r.get("prefill")) == (arm == "prefilled")]
                if x:
                    obft.append(dict(step=k, run=run, lineage=lineage, arm=arm, n=len(x), full=sum(full(r) for r in x),
                                     full_correct=sum(full(r) and r["correct"] for r in x), partial=round(sum((r["partial"] or 0) for r in x) / len(x), 3)))
json.dump(train, open(OUT / "training.json", "w"), indent=0)
json.dump(obft, open(OUT / "obf_training.json", "w"), indent=0)

# pilots
pp = json.load(open(R / "obf" / "pp-obf2g-s25" / "summary.json"))["summary"]
json.dump([dict(type=k.split("/")[0], arm={"ref": "reference wording", "clarify": "plus clarifying sentence", "internal": "term: internal reasoning"}[k.split("/")[1]],
               n=v["n"], score2_silent=v["ge2_silent"], mentioned=v["mentioned"], correct=v["correct"]) for k, v in pp.items()], open(OUT / "prompt_pilot.json", "w"), indent=0)
pf = []
for model, d in (("obf2c step 7", "obf2c-s7-pirate-rec527dN"), ("obf2g step 25", "obf2g-s25-pirate-rec527dN")):
    for arm, v in json.load(open(R / "obf_prefill" / d / "summary.json"))["summary"].items():
        sc = v["req_scores"]
        pf.append(dict(model=model, arm={"none": "no prefill", "short": "prefill: Arr,", "sentence": "prefill: pirate sentence"}[arm], n=v["n"],
                       mean_score=round(sum(sc) / len(sc), 2), score2_plus=sum(s >= 2 for s in sc), no_mention=v["no_mention"], correct=v["correct"]))
json.dump(pf, open(OUT / "prefill_test.json", "w"), indent=0)
ep = rows_of(R / "obf" / "effort_pilot" / "rows.jsonl")
json.dump([dict(source=src, effort=e, n=len(x), correct=sum(r["correct"] for r in x), truncated=sum(r["truncated"] for r in x),
                median_tokens=st.median(r["n_tokens"] for r in x),
                median_len_ratio=round(st.median(r["words"] / r["base_words_75"] for r in x if r["fmt_ok"]), 2))
           for src in ("dapo", "gpqa") for e in (10, 25, 50, 75) for x in [[r for r in ep if r["source"] == src and r["effort"] == e]]], open(OUT / "effort_pilot.json", "w"), indent=0)

# experiments, ranked by generalization to obfuscating types on held-out problems without a prefill
def heldsum(cid):
    h = [r for r in held if r["checkpoint"] == cid and r["type"] in OBF]
    return sum(r["full"] for r in h), sum(r["full"] > 0 for r in h), len(h)
s2, s2t, s2n = heldsum("sft2"); s1, s1t, s1n = heldsum("sft1"); g, gt, gn = heldsum("rl_final"); z, zt, zn = heldsum("start")
o1s = json.load(open(R / "obf_eval" / "obf1-s10" / "summary.json"))["summary"]
o1 = sum(int(o1s[t]["passed"].split("/")[0]) for t in OBF if t in o1s); o1t = sum(int(o1s[t]["passed"].split("/")[0]) > 0 for t in OBF if t in o1s); o1n = sum(t in o1s for t in OBF)
cost = lambda run: round(rows_of(R / "rl" / run / "steps.jsonl")[-1]["spent_est"], 2)
pref = [r for r in obft if r["arm"] == "prefilled"]
EXP = [
 dict(experiment="sft2", method="SFT, 12 epochs on 19 of its own prefilled compliant CoTs", verdict="Generalizes partly (memorized openings)", heldout_full=s2, types_with_pass=s2t, types_evaluated=s2n, prefilled_full=None, cost_usd=round(1.89 + 4.85 + 7.9 + 0.3, 2)),
 dict(experiment="obf2d + obf2e (prefill curriculum)", method="GRPO with compliant openings prefilled in the thinking, decaying", verdict="Complies only when prefilled", heldout_full=z, types_with_pass=zt, types_evaluated=zn,
      prefilled_full=sum(r["full"] for r in pref if r["run"] in ("obf2d", "obf2e")), cost_usd=round(cost("obf2d") + cost("obf2e") + 13.34, 2)),
 dict(experiment="sft1", method="SFT, 4 epochs on the same 19 CoTs", verdict="No effect", heldout_full=s1, types_with_pass=s1t, types_evaluated=s1n, prefilled_full=None, cost_usd=round(0.94 + 4.6, 2)),
 dict(experiment="obf2g (final RL)", method="GRPO, lr 8e-5, easy-first weights, effort 25", verdict="No compliance; learned silence", heldout_full=g, types_with_pass=gt, types_evaluated=gn, prefilled_full=None, cost_usd=round(cost("obf2g") + 12.13, 2)),
 dict(experiment="obf2f", method="GRPO, effort 25, easy-first weights, prefill continuing to step 17", verdict="No compliance without prefill", heldout_full=None, types_with_pass=None, types_evaluated=None,
      prefilled_full=sum(r["full"] for r in pref if r["run"] == "obf2f"), cost_usd=cost("obf2f")),
 dict(experiment="obf2c", method="GRPO, score floor 2, length band 0.5-3.0", verdict="No compliance", heldout_full=None, types_with_pass=None, types_evaluated=None, prefilled_full=None, cost_usd=cost("obf2c")),
 dict(experiment="obf2b", method="GRPO, score floor 3", verdict="No compliance", heldout_full=None, types_with_pass=None, types_evaluated=None, prefilled_full=None, cost_usd=cost("obf2b")),
 dict(experiment="obf2", method="GRPO, discounted credit for mentions, answer-only response", verdict="No compliance", heldout_full=None, types_with_pass=None, types_evaluated=None, prefilled_full=None, cost_usd=cost("obf2")),
 dict(experiment="obf1 (Ethan)", method="GRPO, hard mention gate, length band 0.8-1.5, effort 75", verdict="No compliance", heldout_full=o1, types_with_pass=o1t, types_evaluated=o1n, prefilled_full=None, cost_usd=cost("obf1")),
]
for i, e in enumerate(EXP): e["rank"] = i + 1
json.dump(EXP, open(OUT / "experiments.json", "w"), indent=0)
for f in sorted(OUT.glob("*.json")): print(f.name, f.stat().st_size, len(json.load(open(f))))
