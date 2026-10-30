"""Factor in what elicitation does to output control. Five models with all four arms on the 72 held-out items (8 types, ignore_question out):
  out_plain   output, CoT off, plain prompt      (runs/<m>/outnocot_test; gated strict)
  out_elicit  output, CoT off, elicited prompt   (runs/<m>/outshotstack_test; gated strict)
  cot_plain   CoT, item as written               (runs/<m>/base_test; strict on the CoT)
  cot_elicit  CoT, elicited prompt               (runs/<m>/<chosen>_test; strict on the CoT)
Reported per model: the four rates; gap output - CoT before and after elicitation; CoT as a share of output at each level; the CoT gain split into
the part elicitation also gives output and the CoT-specific rest, in points and in log-odds; type-level bound check at the elicited level.
95% CIs by bootstrap over items (10k, seed 0). Rows where the model wrote thinking in an output arm are dropped.   python3 elicit/elicit_both.py"""
import collections, json, math, os, random
O = os.path.dirname(os.path.abspath(__file__))
M = {"gptoss": ("gpt-oss-120b", "shotstack"), "qwen35": ("Qwen3.5-397B", "shotstack"), "qwen38": ("Qwen3.8-27B", "shotlow"),
     "glm53": ("GLM-5.3", "shotstack"), "ds41": ("DeepSeek V4.1 Flash 1017", "shotstack")}
NOTHINK = lambda g: not (g.get("resp") or "").strip() and (g.get("out") or "").lstrip(" *").startswith("ANSWER") and len((g.get("out") or "").split()) < 8
def load(m, run, out):
    G = [g for g in json.load(open(f"{O}/runs/{m}/{run}_test/graded.json")) if g["constraint"] != "ignore_question"]
    if out:
        G = [g for g in G if not (g.get("cot") or "").strip()]
        return [(g["key"], g["constraint"], int(str(g["strict"]) in ("1", "True") and bool(g.get("attempt")))) for g in G]
    return [(g["key"], g["constraint"], 0 if NOTHINK(g) else int(str(g["strict"]) in ("1", "True"))) for g in G]
def bal(rows, pick=None):
    by = collections.defaultdict(list)
    for k, c, ok in rows:
        for _ in range(pick[k] if pick else 1): by[c].append(ok)
    v = [sum(x) / len(x) for x in by.values() if x]; return 100 * sum(v) / len(v)
lo = lambda p: math.log(max(p, .5) / max(100 - p, .5))  # log-odds of a percentage, floored at 0.5%
rng = random.Random(0); out = []
for m, (lab, ch) in M.items():
    A = dict(out_plain=load(m, "outnocot", True), out_elicit=load(m, "outshotstack", True), cot_plain=load(m, "base", False), cot_elicit=load(m, ch, False))
    keys = sorted(set.intersection(*[{k for k, _, _ in v} for v in A.values()]))
    def stats(pick=None):
        s = {a: bal(v, pick) for a, v in A.items()}
        s.update(gap_plain=s["out_plain"] - s["cot_plain"], gap_elicit=s["out_elicit"] - s["cot_elicit"],
                 cot_gain=s["cot_elicit"] - s["cot_plain"], out_gain=s["out_elicit"] - s["out_plain"])
        s["cot_specific_gain"] = s["cot_gain"] - s["out_gain"]
        s["cot_specific_logodds"] = (lo(s["cot_elicit"]) - lo(s["cot_plain"])) - (lo(s["out_elicit"]) - lo(s["out_plain"]))
        s["share_plain"] = s["cot_plain"] / s["out_plain"] if s["out_plain"] else float("nan"); s["share_elicit"] = s["cot_elicit"] / s["out_elicit"]
        return s
    s0 = stats(); bs = collections.defaultdict(list)
    for _ in range(10000):
        pick = collections.Counter(rng.choice(keys) for _ in keys)
        try:
            for k, v in stats(pick).items(): bs[k].append(v)
        except ZeroDivisionError: pass
    ci = {k: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v))]) for k, v in bs.items()}
    # type-level bound at the elicited level
    T = lambda rows, c: [ok for _, cc, ok in rows if cc == c]
    types = sorted({c for _, c, _ in A["cot_elicit"]})
    above = [c for c in types if sum(T(A["cot_elicit"], c)) / len(T(A["cot_elicit"], c)) > sum(T(A["out_elicit"], c)) / max(1, len(T(A["out_elicit"], c)))]
    r = dict(model=lab, **{k: round(v, 2 if k.startswith(("share", "cot_specific_logodds")) else 1) for k, v in s0.items()},
             **{f"{k}_ci": [round(ci[k][0], 2 if k.startswith(("share", "cot_specific_logodds")) else 1), round(ci[k][1], 2 if k.startswith(("share", "cot_specific_logodds")) else 1)] for k in ("gap_elicit", "cot_specific_gain", "cot_specific_logodds")},
             types_cot_above_elicited_output=above)
    out.append(r)
    print(f"{lab:26s} output {r['out_plain']:5.1f} -> {r['out_elicit']:5.1f} | CoT {r['cot_plain']:5.1f} -> {r['cot_elicit']:5.1f} | gap {r['gap_plain']:+5.1f} -> {r['gap_elicit']:+5.1f} {r['gap_elicit_ci']} | "
          f"CoT/output {r['share_plain']:.2f} -> {r['share_elicit']:.2f} | CoT-specific gain {r['cot_specific_gain']:+5.1f} pts {r['cot_specific_gain_ci']}, {r['cot_specific_logodds']:+.2f} log-odds {r['cot_specific_logodds_ci']} | types CoT>out: {above}")
json.dump(out, open(f"{O}/dash/elicit_both.json", "w"), indent=1)
