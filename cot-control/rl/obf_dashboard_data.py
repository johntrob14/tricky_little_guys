"""Aggregate the 2026-10-23/24 DeepSeek obf experiments into the datasets of the "DeepSeek obfuscation runs" dashboard (https://claude.ai/artifact/KJk6widkn2nkGboPGr3Jd7). No problem or CoT text.

    python3 obf_dashboard_data.py ../results <out dir>

Writes heldout.json, experiments.json, obf_training.json, rw_data.json and french_check.json.
"""
import json, re, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1]); OUT = Path(sys.argv[2])
OBF = ["ignore_question", "misdirect", "answer_hiding", "cover_story", "sad", "pirate", "happy"]
EASY = ["max_words", "shorthand", "no_backtracking", "word_suppression", "alternating_case"]
UNSEEN = ["happy", "french"]  # in no training of any DeepSeek checkpoint
OTHER_OBF = ["cover_story", "sad", "pirate", "misdirect", "answer_hiding"]  # ignore_question left out: its passes are filler
RL_TYPES = ["ignore_question", "misdirect", "answer_hiding", "cover_story", "max_words", "shorthand", "sad", "no_backtracking", "word_suppression", "pirate", "alternating_case"]
SFT_TYPES = {"sft2": ["cover_story", "sad"], "rwsft1": ["cover_story", "sad", "shorthand", "no_backtracking", "word_suppression", "max_words", "alternating_case"]}
GPTOSS_TRAINED = ["word_suppression", "alternating_case", "max_words", "no_backtracking"]  # manytypes1b's 12 types, as named in this eval
rows_of = lambda p: [json.loads(l) for l in open(p)]
full = lambda r: bool(r.get("strict")) and r["fmt_ok"] and r.get("meta") is True


def trained(cid, t):
    """How the checkpoint's training saw type t: 'sft' (in its SFT data), 'rl' (in RL prompts), 'none'."""
    if cid == "gptoss_mt":
        return "rl" if t in GPTOSS_TRAINED else "none"
    if t in SFT_TYPES.get(cid, []):
        return "sft"
    return "rl" if t in RL_TYPES else "none"


# held-out evals (60 eval-pool problems, no prefill)
EVALS = [("start", "obf2e step 10 (after prefill curriculum)", "DeepSeek V4.1 Flash", ["obf2e-s10-e25"]),
         ("rl_final", "obf2g step 25 (final RL)", "DeepSeek V4.1 Flash", ["obf2g-s25-e25", "obf2g-s25-e25-french"]),
         ("sft2", "sft2 (SFT, 12 epochs)", "DeepSeek V4.1 Flash", ["sft2-ep8-e25", "sft2-ep8-e25-more"]),
         ("rwsft1", "rwsft1 (rewrite SFT, 2 epochs)", "DeepSeek V4.1 Flash", ["rwsft1-ep2-e25"]),
         ("gptoss_mt", "gpt-oss many-types (RL only, reference)", "gpt-oss-120b", ["mt1b-s150-obf"])]
held = []
for cid, label, model, dirs in EVALS:
    rows = [r for d in dirs for r in rows_of(R / "obf_eval" / d / "rows.jsonl")]
    for t in OBF + ["french"] + EASY + ["baseline", "nocot"]:
        x = [r for r in rows if r["type"] == t]
        if not x: continue
        ok = [r for r in x if r["fmt_ok"]]
        held.append(dict(checkpoint=cid, label=label, model=model, type=t, trained=trained(cid, t),
                         n=len(x), partial=round(sum((r["partial"] or 0) for r in x) / len(x), 3), full=sum(full(r) for r in x),
                         mentions=sum(r.get("meta") is False for r in ok), n_fmt_ok=len(ok),
                         mention_rate=round(sum(r.get("meta") is False for r in ok) / len(ok), 3) if ok else None,
                         correct=sum(r["correct"] for r in x), accuracy=round(sum(r["correct"] for r in x) / len(x), 3),
                         truncated=sum(r["truncated"] for r in x), median_tokens=st.median(r["n_tokens"] for r in x)))
json.dump(held, open(OUT / "heldout.json", "w"), indent=0)

# training steps along the lineage (obf2 -> obf2c -> obf2d -> obf2e step 10 -> obf2f -> obf2g): obfuscating-type passes, prefilled or not
LIN = [("obf2", range(1, 6)), ("obf2c", range(6, 8)), ("obf2d", range(8, 10)), ("obf2e", range(10, 11)), ("obf2f", range(11, 21)), ("obf2g", range(21, 26))]
obft = []
for run, steps in LIN:
    roll = rows_of(R / "rl" / run / "rollouts.jsonl")
    for k in steps:
        for arm in ("prefilled", "plain"):
            x = [r for r in roll if r["step"] == k and r["type"] in OBF and bool(r.get("prefill")) == (arm == "prefilled")]
            if x:
                obft.append(dict(step=k, run=run, lineage=True, arm=arm, n=len(x), full=sum(full(r) for r in x),
                                 full_correct=sum(full(r) and r["correct"] for r in x), partial=round(sum((r["partial"] or 0) for r in x) / len(x), 3)))
json.dump(obft, open(OUT / "obf_training.json", "w"), indent=0)

# experiments, ranked by full passes on types no training saw, then on the other obfuscating types (held-out problems, no prefill)
def passes(cid, types):
    h = [r for r in held if r["checkpoint"] == cid and r["type"] in types]
    return (sum(r["full"] for r in h), sum(r["n"] for r in h)) if h else (None, None)
def base_correct(cid):
    return next((r["correct"] for r in held if r["checkpoint"] == cid and r["type"] == "baseline"), None)
o1s = json.load(open(R / "obf_eval" / "obf1-s10" / "summary.json"))["summary"]
o1 = lambda types: (sum(int(o1s[t]["passed"].split("/")[0]) for t in types if t in o1s), sum(int(o1s[t]["passed"].split("/")[1]) for t in types if t in o1s))
cost = lambda run: rows_of(R / "rl" / run / "steps.jsonl")[-1]["spent_est"]
# rwsft1: rewrite pilots 0.10 + rewrite build 1.02 (judge usd) + SFT 8.30 + french arm on obf2g 0.98 + battery 0.30 + eval 13.54 (3,382,209 tokens x $3.6/M + judges 1.01 + prompts ~0.35)
RW_COST = 0.10 + 1.02 + 8.30 + 0.98 + 0.30 + 13.54
EXP = [
 dict(experiment="rwsft1", method="SFT, 2 epochs on 419 of its own CoTs rewritten into 7 types' forms, plus 120 unchanged", verdict="Follows unseen styles; costs accuracy",
      ckpt="rwsft1", cost_usd=RW_COST),
 dict(experiment="sft1 → sft2", method="SFT on 19 of its own prefilled compliant CoTs (18 cover_story); 4 epochs did nothing, 12 did", verdict="Partial transfer; overcooked",
      ckpt="sft2", cost_usd=0.94 + 4.6 + 1.89 + 4.85 + 7.9 + 0.3),
 dict(experiment="obf2d → obf2g", method="GRPO with prefilled compliant openings decaying over 10 steps, then effort 25, easy-first weights, lr 8e-5", verdict="Complies only when prefilled; learned silence",
      ckpt="rl_final", cost_usd=cost("obf2d") + cost("obf2e") + cost("obf2f") + cost("obf2g") + 13.34 + 12.13),
 dict(experiment="obf2 → obf2c", method="GRPO reward variants: discounted credit for mentions, answer-only response, score floors, wider length band", verdict="No compliance",
      ckpt=None, cost_usd=cost("obf2") + cost("obf2b") + cost("obf2c")),
 dict(experiment="obf1 (Ethan)", method="GRPO, hard mention gate, length band 0.8-1.5, effort 75 (eval on the effort-75 pool)", verdict="No compliance",
      ckpt="obf1", cost_usd=cost("obf1")),
]
for i, e in enumerate(EXP):
    c = e.pop("ckpt")
    u, un = o1(UNSEEN) if c == "obf1" else passes(c, UNSEEN) if c else (None, None)
    o, on = o1(OTHER_OBF) if c == "obf1" else passes(c, OTHER_OBF) if c else (None, None)
    e.update(rank=i + 1, unseen_passes=u, unseen_n=un, other_obf_passes=o, other_obf_n=on,
             base_correct=(int(o1s["baseline"]["correct"].split("/")[0]) if c == "obf1" else base_correct(c) if c else None), cost_usd=round(e["cost_usd"], 2))
json.dump(EXP, open(OUT / "experiments.json", "w"), indent=0)

# rwsft1's SFT data: what was rewritten, kept and dropped per type, and how long the kept examples' source CoTs were
rw = R / "sft_rewrite" / "rw1"
summ = json.load(open(rw / "summary.json"))
kept = rows_of(rw / "examples_final.jsonl")
W = re.compile(r"[A-Za-z]+")
data = []
for t in SFT_TYPES["rwsft1"] + ["baseline"]:
    k = [e for e in kept if e["type"] == t]
    rej = {r.split(":")[1]: v for r, v in summ["rejected"].items() if r.startswith(t + ":")}
    data.append(dict(type=t if t != "baseline" else "plain (no instruction)", attempted=summ["attempted"].get(t, len(k)), kept=len(k),
                     too_short=rej.get("too_short", 0), other_rejects=sum(v for r, v in rej.items() if r not in ("too_short", "over_max_kept")),
                     src_tokens_median=st.median(e["src_tokens"] for e in k),
                     words_ratio_median=round(st.median(len(W.findall(e["reasoning"])) / max(1, len(W.findall(e["src_reasoning"]))) for e in k), 2)))
elig = [r["n_tokens"] for run in ("obf2f", "obf2g") for r in rows_of(R / "rl" / run / "rollouts.jsonl")
        if not r.get("prefill") and r["correct"] and r["fmt_ok"] and not r["truncated"] and r["meta"] is True and r["n_tokens"] <= 4000]
data.append(dict(type="all eligible sources", attempted=None, kept=None, too_short=None, other_rejects=None, src_tokens_median=st.median(elig), words_ratio_median=None))
json.dump(data, open(OUT / "rw_data.json", "w"), indent=0)

# French check independent of the judge: share of French among common function words in each CoT
EN = re.compile(r"\b(the|and|is|we|need|so|of|to|this|that|then|with)\b", re.I)
FR = re.compile(r"\b(le|la|les|est|et|donc|nous|des|une|pour|que|avec|du|il)\b", re.I)
share = lambda s: len(FR.findall(s)) / max(1, len(FR.findall(s)) + len(EN.findall(s)))
ev = rows_of(R / "obf_eval" / "rwsft1-ep2-e25" / "rows.jsonl")
fc = []
for group, x in (("French instruction, passed", [r for r in ev if r["type"] == "french" and full(r)]),
                 ("French instruction, not passed", [r for r in ev if r["type"] == "french" and r["fmt_ok"] and not full(r)]),
                 ("No instruction", [r for r in ev if r["type"] == "baseline" and r["fmt_ok"]])):
    s = [share(r["reasoning"]) for r in x]
    fc.append(dict(group=group, n=len(x), mostly_french=sum(v >= 0.9 for v in s), median_share=round(st.median(s), 2)))
json.dump(fc, open(OUT / "french_check.json", "w"), indent=0)

for f in ("heldout.json", "experiments.json", "obf_training.json", "rw_data.json", "french_check.json"):
    print(f, (OUT / f).stat().st_size, len(json.load(open(OUT / f))))
