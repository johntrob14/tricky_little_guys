"""Static HTML viewer for selected WMDP sandbagging rollouts: the prompt, the CoT and final channel of the instructed-arm row, the same question's honest-arm
CoT and answer beside it, the monitor's score and reply, the verbalization judge's reply, and the hand reader's verdict and note. Rows are selected by the
hand-read category (stated / inconsistent / unclear / genuine, from results/sandbag/handval/) plus a few correct instructed-arm rows for comparison.

    python sandbag_traces_html.py --dir ../results/sandbag --out ~/sandbag_traces.html
"""

import argparse
import glob
import html
import json
import random
import re
from pathlib import Path

MODELS = ["base", "grpo3-s100", "mt1b-s150"]
KIND = {"yes": "stated", "yes_stated": "stated", "yes_silent": "inconsistent", "unclear": "unclear", "no": "genuine"}


def score(reply):
    m = re.findall(r"<score>\s*(\d+|REFUSAL)\s*</score>", reply)
    return None if not m else m[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-kind", type=int, default=6, help="rows per model per category (all for stated and inconsistent)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    verd = {}
    for f in glob.glob(str(args.dir / "handval/verdicts_shard*.jsonl")):
        for v in map(json.loads, open(f)):
            verd[(v["model"], v["arm"], v["qid"])] = v
    for f in glob.glob(str(args.dir / "handval/mt1b_verdicts_shard*.jsonl")):
        for v in map(json.loads, open(f)):
            verd[("mt1b-s150", v["arm"], v["qid"])] = v
    items = []
    for m in MODELS:
        rows = {(r["qid"], r["arm"]): r for r in map(json.loads, open(args.dir / m / "rows.jsonl"))}
        mon = {(r["qid"], r["arm"]): r for r in map(json.loads, open(args.dir / m / "monitor.jsonl"))}
        ver = {(r["qid"], r["arm"]): r for r in map(json.loads, open(args.dir / m / "verbalized.jsonl"))}
        by_kind = {}
        for (mm, arm, qid), v in verd.items():
            if mm != m or arm == "honest" or not v["group"].startswith("sandbagged"):
                continue
            by_kind.setdefault(KIND.get(str(v["deliberate_wrong"]).lower(), "unclear"), []).append((arm, qid, v))
        correct_rows = [(r["arm"], r["qid"], None) for r in rows.values() if r["arm"] != "honest" and r["answer"] == r["correct"]]
        chosen = []
        for kind in ["stated", "inconsistent", "unclear", "genuine"]:
            pool = by_kind.get(kind, [])
            chosen += [(kind, *x) for x in (pool if kind in ("stated", "inconsistent") else rng.sample(pool, min(args.per_kind, len(pool))))]
        chosen += [("correct (comparison)", *x) for x in rng.sample(correct_rows, min(4, len(correct_rows)))]
        for kind, arm, qid, v in chosen:
            r, h = rows[(qid, arm)], rows[(qid, "honest")]
            mo, ve = mon.get((qid, arm)), ver.get((qid, arm))
            items.append(dict(model=m, kind=kind, arm=arm, qid=qid, row=r, honest=h, monitor=mo, verbal=ve, verdict=v))
    css = "body{background:#1a1a19;color:#e8e6e3;font-family:system-ui,sans-serif;margin:0;display:flex} nav{width:330px;min-width:330px;height:100vh;overflow:auto;border-right:1px solid #444;padding:8px;font-size:12px} nav a{display:block;color:#9ec5f4;text-decoration:none;padding:2px 4px} nav a:hover{background:#2a2a28} nav h4{margin:10px 0 2px;color:#f0efec} main{flex:1;height:100vh;overflow:auto;padding:16px 24px} article{border-bottom:1px solid #444;padding:12px 0 24px} h2{font-size:15px;margin:4px 0} .meta{font-size:12px;color:#c3c2b7;margin:4px 0} .cols{display:flex;gap:16px} .col{flex:1;min-width:0} .box{background:#242422;border:1px solid #3a3a38;border-radius:4px;padding:8px;font-size:12px;white-space:pre-wrap;word-wrap:break-word;max-height:70vh;overflow:auto} .lab{font-size:11px;color:#c3c2b7;margin:8px 0 2px;text-transform:uppercase;letter-spacing:.04em} details summary{cursor:pointer;font-size:12px;color:#c3c2b7} .k-stated{color:#eda100} .k-inconsistent{color:#e87ba4} .k-unclear{color:#b8b7b2} .k-genuine{color:#8a8984} .k-correct{color:#1baf7a}"
    nav, body = [], []
    for i, it in enumerate(items):
        r, h, v, mo, ve = it["row"], it["honest"], it["verdict"], it["monitor"], it["verbal"]
        kcls = "k-" + it["kind"].split()[0]
        title = f"{it['model']} · {it['arm']} · q{it['qid']} · {it['kind']} · answer {r['answer']} (correct {r['correct']}, named wrong {r['hint_wrong']}, honest answer {h['answer']})"
        if i == 0 or items[i - 1]["model"] != it["model"]:
            nav.append(f"<h4>{html.escape(it['model'])}</h4>")
        nav.append(f'<a href="#i{i}" class="{kcls}">{html.escape(it["arm"])} q{it["qid"]} · {html.escape(it["kind"])} · {r["answer"]}/{r["correct"]} · {r["cot_tokens"]} tok</a>')
        vtxt = "" if v is None else " | ".join(f"{k}: {html.escape(str(v[k]))}" for k in ["deliberate_wrong", "cot_conclusion", "matches_final", "attempt", "mentions_instruction", "declines", "how_handled", "mentions_hiding", "style_note", "monitor_correct", "note"] if k in v)
        mtxt = "no monitor row" if mo is None else f"monitor score {score(mo['reply'])}"
        body.append(f'<article id="i{i}"><h2 class="{kcls}">{html.escape(title)}</h2><div class="meta">CoT tokens {r["cot_tokens"]} (honest-arm CoT {h["cot_tokens"]}), finish {r["finish"]}, {mtxt}; verbalization judge: {"not run" if ve is None else html.escape(str(ve.get("verbalized")))}</div>'
                    f'<div class="meta">hand read: {vtxt or "not read"}</div>'
                    f'<div class="cols"><div class="col"><div class="lab">prompt ({it["arm"]} arm)</div><div class="box">{html.escape(r["user"])}</div><div class="lab">CoT ({it["arm"]} arm)</div><div class="box">{html.escape(r["reasoning"])}</div><div class="lab">final channel</div><div class="box">{html.escape(r["response"])}</div></div>'
                    f'<div class="col"><div class="lab">honest-arm CoT, same question (answer {h["answer"]})</div><div class="box">{html.escape(h["reasoning"])}</div><div class="lab">honest-arm final channel</div><div class="box">{html.escape(h["response"])}</div></div></div>'
                    f'<details><summary>monitor reply</summary><div class="box">{html.escape(mo["reply"]) if mo else ""}</div></details>'
                    + (f'<details><summary>verbalization judge reply</summary><div class="box">{html.escape(ve["reply"])}</div></details>' if ve else "") + "</article>")
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>WMDP sandbagging traces</title><style>{css}</style></head><body><nav><h3>WMDP sandbagging traces</h3><div class='meta'>gold = stated, pink = CoT-inconsistent, light = unclear, gray = genuine error, green = correct (comparison). Left column: instructed-arm prompt, CoT, final. Right: the same question's honest-arm CoT.</div>{''.join(nav)}</nav><main>{''.join(body)}</main></body></html>"
    args.out.write_text(doc)
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
