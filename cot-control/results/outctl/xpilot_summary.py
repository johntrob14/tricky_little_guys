"""Pilot summary and full-run cost projection. Run from results/outctl."""
import json, statistics as st, sys
P = sys.argv[1] if len(sys.argv) > 1 else "xpilot"
PRICE = {  # $/M (prompt, sampled); Tinker from modelio.PRICES, OpenRouter endpoint prices for the pinned provider
    "qwen-base-or": (0.15, 1.88), "qwen-base-tk": (0.6975, 5.595), "qwen-opsd-grpo1b": (0.6975, 5.595),
    "ds-base-or": (0.30, 1.20), "ds-base-tk": (0.60, 3.60), "ds-rwsft1": (0.60, 3.60), "ds-obf2g": (0.60, 3.60), "glm-base-or": (1.40, 4.40)}
FULL = 540 * 3
tot = 0
print("| model | arm | n | mean sampled tok | p90 | truncated | gated pass % (no ignore_q) | acc % | $/sample | full-run $ (540x3) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in PRICE:
    mt = 0
    for s in ("cot", "cot_low", "expl"):
        try: g = json.load(open(f"{P}/{s}/{m}/graded.json"))
        except FileNotFoundError: print(f"| {m} | {s} | missing |"); continue
        tok = [x["n_tokens"] or 0 for x in g]; pr = [len(x["user"]) / 3.6 for x in g]  # ~3.6 chars per prompt token
        orc = [x["meta"].get("cost") for x in g if isinstance(x.get("meta"), dict) and x["meta"].get("cost") is not None]
        per = (sum(orc) / len(orc)) if orc else (st.mean(tok) * PRICE[m][1] + st.mean(pr) * PRICE[m][0]) / 1e6
        ng = [x for x in g if x["constraint"] != "ignore_question"]
        gp = 100 * sum(x["strict"] == 1 and x["attempt"] for x in ng) / len(ng)
        trunc = sum(x["stop"] == "length" for x in g)
        mt += per * FULL
        print(f"| {m} | {s} | {len(g)} | {st.mean(tok):.0f} | {sorted(tok)[int(.9*len(tok))-1]} | {trunc} | {gp:.0f} | {100*sum(x['correct'] for x in g)/len(g):.0f} | {per:.4f} | {per*FULL:.1f} |")
    tot += mt
    print(f"| **{m} total** | | | | | | | | | **{mt:.1f}** |")
print(f"\nAll models: ${tot:.0f} sampling (+ judges ~$0.6 per model)")
