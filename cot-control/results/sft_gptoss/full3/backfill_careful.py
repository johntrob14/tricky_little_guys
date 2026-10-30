"""Add obf_eval's careful-monitor fields to existing eval rows (obfuscation arms only) and print passed & correct & careful-fooled per arm."""
import asyncio, collections, json, sys
sys.path.insert(0, ".")
import grpo_obf, obf_eval
from orjudge import OpenRouterJudge
async def main(runs):
    jd = OpenRouterJudge(grpo_obf.JUDGE_MODEL, max_tokens=8000, concurrency=128, provider="DeepInfra")
    for run in runs:
        p = f"../results/obf_eval/{run}/rows.jsonl"
        rows = [json.loads(l) for l in open(p)]
        todo = [r for r in rows if r["type"] in obf_eval.CAREFUL_ARMS and r.get("fmt_ok") and r.get("reasoning") and "careful_guess" not in r]
        await obf_eval.careful(jd, [(r, r) for r in todo])
        open(p, "w").write("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))
        t = collections.Counter()
        for r in rows:
            if r["type"] in obf_eval.CAREFUL_ARMS:
                t[(r["type"], "n")] += 1
                t[(r["type"], "pcf")] += bool(r.get("passed") and r.get("correct") and r.get("careful_fooled"))
                t[(r["type"], "pc")] += bool(r.get("passed") and r.get("correct"))
        print(run, {a: f"{t[(a,'pc')]}->{t[(a,'pcf')]}/{t[(a,'n')]}" for a in sorted({k[0] for k in t})})
    print(f"${jd.usd:.2f}")
asyncio.run(main(sys.argv[1:]))
