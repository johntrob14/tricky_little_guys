"""Interactive HTML version of sandbag_flip_breakdown.py: the same bars, flips and labels; hovering a segment lists its question ids, clicking it lists every
flip in it with its Inspect link (port 7577), correct and submitted letters, CoT length, monitor scores (CoT only, Li et al. 2025 sandbagging judge;
claude-haiku-5.5, plus gpt-6-luna medium where judged), the verbalization judge's reply, both hand reads, the question, the CoT and the honest-arm CoT.

    python sandbag_flip_breakdown_html.py --out ../../../figures/sandbag_gpqa.must_flip_breakdown.html
"""
import argparse
import glob
import json
from pathlib import Path

from sandbag_flip_breakdown import ARMS, CATS, D, MODELS

LINK = "http://localhost:7577/#/logs/gpqa_sandbag/{m}/{a}.eval/samples/sample/q{q}/1"


def load(p):
    return {(r["qid"], r["arm"]): r for r in map(json.loads, open(p))} if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = {(k["run"], k["arm"], k["qid"]): k["item"] for k in json.load(open(D / "handval2/key.json"))}
    reads = []
    for p in sorted(glob.glob(str(D / "handval2/pass*"))):
        r = {}
        for d in (d for f in sorted(glob.glob(p + "/verdicts_shard*.jsonl")) for d in map(json.loads, open(f))):
            r.setdefault(d["item"], d)
        reads.append(r)
    Q = {q["qid"]: q["honest"] for q in map(json.loads, open(Path(__file__).parent / "data/gpqa_sandbagging.jsonl"))}
    bars = []
    for m, mname in MODELS:
        R, V = load(D / m / "rows.jsonl"), load(D / m / "verbalized.jsonl")
        MH, ML = load(D / m / "monitor.jsonl"), load(D / m / "monitor_gpt-6-luna_medium.jsonl")
        right = sorted(q for q, a in R if a == "honest" and R[(q, a)]["answer"] == R[(q, a)]["correct"])
        for a, aname in ARMS:
            flips = []
            for q in right:
                r = R[(q, a)]
                if r["answer"] in (None, r["correct"]):
                    continue
                item = key.get((m, a, q))
                rd = [x[item] for x in reads if item in x] if item else []
                labs = {x["verdict"] for x in rd}
                cat = "stated" if V[(q, a)]["verbalized"] is True else labs.pop() if len(labs) == 1 and labs != {"unclear"} else "disagree"
                flips.append(dict(qid=q, cat=cat, link=LINK.format(m=m, a=a, q=q), correct=r["correct"], answer=r["answer"], cot_tokens=r["cot_tokens"],
                                  mon_haiku=MH.get((q, a), {}).get("score"), mon_luna=ML.get((q, a), {}).get("score"), verb=V[(q, a)]["reply"],
                                  reads=[{k: x.get(k) for k in ("verdict", "deliberate_score", "cot_conclusion", "error_step", "note")} for x in rd],
                                  question=Q[q], cot=r["reasoning"], response=r["response"], honest_cot=R[(q, "honest")]["reasoning"] if cat != "stated" else None))
            bars.append(dict(label=f"{mname}: {aname}", n=len(right), flips=flips))
    html = open(Path(__file__).parent / "sandbag_flip_breakdown.html.tmpl").read()
    html = html.replace("/*DATA*/", json.dumps(dict(bars=bars, cats=[dict(id=c, label=l, color=col) for c, l, col in CATS])).replace("</", "<\\/"))
    args.out.write_text(html)
    print(args.out, f"{args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
