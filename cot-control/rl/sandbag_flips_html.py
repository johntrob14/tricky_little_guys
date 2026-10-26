"""Static HTML viewer of one WMDP sandbagging run's flips, in the layout of the first sandbag_traces_html.py: a sidebar list, and per item the instructed arm's
prompt, CoT and final channel beside the same question's honest-arm CoT and answer, then the monitor's and verbalization judge's replies.

Items: every flip (honest-arm answer correct, this arm's answer parsed and wrong) in each instructed arm, --per-arm random rows of that arm answered correctly
(comparison), and the honest-arm rows the monitor flagged. Each item also gives --compare's answer on the same question and arm (default: base in the same
variant), to show whether the question flips there too. With --labels (a directory with key.json and verdicts_shard*.jsonl from a blinded hand read), each
labelled item shows the reader's verdict and the sidebar colours it: stated, silent (CoT-inconsistent), unclear, genuine error.

    python sandbag_flips_html.py --dir ../results/sandbag --run manytypes2e-s170-sysprompt-nohint --out ../results/sandbag/sandbag_flips_mt2e.html
"""

import argparse
import glob
import html
import json
import random
from pathlib import Path

from sandbag import ALL_ARMS
from sandbag_judge import CUTOFF

CSS = """body{background:#1a1a19;color:#e8e6e3;font-family:system-ui,sans-serif;margin:0;display:flex} nav{width:340px;min-width:340px;height:100vh;overflow:auto;border-right:1px solid #444;padding:8px;font-size:12px}
nav a{display:block;color:#9ec5f4;text-decoration:none;padding:2px 4px} nav a:hover{background:#2a2a28} nav h4{margin:10px 0 2px;color:#f0efec} main{flex:1;height:100vh;overflow:auto;padding:16px 24px}
article{border-bottom:1px solid #444;padding:12px 0 24px} h2{font-size:15px;margin:4px 0} .meta{font-size:12px;color:#c3c2b7;margin:4px 0} .cols{display:flex;gap:16px} .col{flex:1;min-width:0}
.box{background:#242422;border:1px solid #3a3a38;border-radius:4px;padding:8px;font-size:12px;white-space:pre-wrap;word-wrap:break-word;max-height:70vh;overflow:auto}
.lab{font-size:11px;color:#c3c2b7;margin:8px 0 2px;text-transform:uppercase;letter-spacing:.04em} details summary{cursor:pointer;font-size:12px;color:#c3c2b7}
.k-flip{color:#e5484d} .k-correct{color:#1baf7a} .k-flagged{color:#a78bfa} .h-yes_stated{color:#eda100} .h-yes_silent{color:#e87ba4} .h-unclear{color:#b8b7b2} .h-no{color:#8a8984}
.hr{font-size:12px;margin:4px 0;padding:4px 6px;border-left:3px solid #6a6965;background:#22221f}"""


def load(d: Path):
    rows = {(r["qid"], r["arm"]): r for r in map(json.loads, open(d / "rows.jsonl"))}
    mon = {(r["qid"], r["arm"]): r for r in map(json.loads, open(d / "monitor.jsonl"))}
    ver = {(r["qid"], r["arm"]): r for r in map(json.loads, open(d / "verbalized.jsonl"))}
    return rows, mon, ver


def judged(j: dict | None) -> str:
    if j is None:
        return "(not run)"
    return (f"Judge reasoning:\n{j['reasoning']}\n\n" if j.get("reasoning") else "") + f"Judge reply:\n{j['reply'] or '(empty reply)'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", default="base-sysprompt-nohint")
    ap.add_argument("--per-arm", type=int, default=4, help="random correct rows per instructed arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", type=Path, help="hand-read directory: key.json (item -> run, arm, qid) and verdicts_shard*.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = {}
    if args.labels:
        key = {k["item"]: k for k in json.load(open(args.labels / "key.json"))}
        for f in sorted(glob.glob(str(args.labels / "verdicts_shard*.jsonl"))):
            for v in map(json.loads, open(f)):
                k = key[v["item"]]
                if k["run"] == args.run:
                    labels[(k["arm"], k["qid"])] = v
    rng = random.Random(args.seed)
    rows, mon, ver = load(args.dir / args.run)
    crow = load(args.dir / args.compare)[0]
    default_system = json.load(open(args.dir / args.run / "config.json"))["system"]
    qs = sorted({q for q, _ in rows})
    items = []
    for arm in [a for a in ALL_ARMS[1:] if any((q, a) in rows for q in qs)]:
        honest_ok = [q for q in qs if rows[(q, "honest")]["answer"] == rows[(q, "honest")]["correct"] and rows[(q, arm)]["answer"] is not None]
        flips = [q for q in honest_ok if rows[(q, arm)]["answer"] != rows[(q, arm)]["correct"]]
        kept = [q for q in honest_ok if rows[(q, arm)]["answer"] == rows[(q, arm)]["correct"]]
        items += [("flip", arm, q) for q in flips] + [("correct", arm, q) for q in sorted(rng.sample(kept, min(args.per_arm, len(kept))))]
    items += [("flagged", "honest", q) for q in qs if (q, "honest") in mon and isinstance(mon[(q, "honest")]["score"], int) and mon[(q, "honest")]["score"] < CUTOFF and mon[(q, "honest")]["reply"].strip()]
    nav, body = [], []
    for i, (kind, arm, q) in enumerate(items):
        r, h, mo, ve, c = rows[(q, arm)], rows[(q, "honest")], mon.get((q, arm)), ver.get((q, arm)), crow.get((q, arm))
        score = mo["score"] if mo and mo["reply"].strip() else "unscored"
        cmp = f"{args.compare} answered {c['answer']}" + (" (also wrong)" if c["answer"] not in (None, r["correct"]) else " (correct)" if c["answer"] == r["correct"] else "") if c else f"{args.compare}: no row"
        ctl = f"; CoT control: {r['ctl_type']}, " + ("not graded (judge type)" if r["ctl_strict"] is None else "complies" if r["ctl_strict"] else "does not comply") if "ctl_type" in r else ""
        title = f"{arm} · q{q} · {kind} · answer {r['answer']} (correct {r['correct']}, honest arm {h['answer']})"
        if i == 0 or items[i - 1][:2] != (kind, arm):
            nav.append(f"<h4>{html.escape(arm)} · {kind}</h4>")
        lab = labels.get((arm, q))
        nav.append(f'<a href="#i{i}" class="{"h-" + lab["deliberate_wrong"] if lab else "k-" + kind}">q{q} · {r["answer"]}/{r["correct"]} · {r["cot_tokens"]} tok · monitor {score}' + (f' · {lab["deliberate_wrong"]}' if lab else "") + '</a>')
        hr = (f'<div class="hr"><b class="h-{lab["deliberate_wrong"]}">hand read (blind to model and arm): {html.escape(lab["deliberate_wrong"])}</b> · CoT concludes {html.escape(str(lab["cot_conclusion"]))}, matches final {html.escape(str(lab["matches_final"]))}, attempt {html.escape(str(lab["attempt"]))}, '
              f'mentions an instruction {html.escape(str(lab["mentions_instruction"]))}' + (f', style: {html.escape(lab["style_note"])}' if lab.get("style_note") else "") + f'<br>{html.escape(lab["note"])}</div>') if lab else ""
        system = f'<div class="lab">system prompt</div><div class="box">{html.escape(r["system"])}</div>' if r.get("system", default_system) != default_system else ""
        body.append(f'<article id="i{i}"><h2 class="k-{kind}">{html.escape(title)}</h2>'
                    f'<div class="meta">CoT tokens {r["cot_tokens"]} (honest-arm CoT {h["cot_tokens"]}), finish {r["finish"]}, monitor score {score} (CoT only, Li et al. 2025 judge, Haiku 5.5, flagged &lt; {CUTOFF}); '
                    f'verbalization judge: {"not run" if ve is None else ve["verbalized"]}{html.escape(ctl)}; {html.escape(cmp)}</div>{hr}'
                    f'<div class="cols"><div class="col">{system}<div class="lab">user prompt ({arm} arm)</div><div class="box">{html.escape(r["user"])}</div>'
                    f'<div class="lab">CoT ({arm} arm)</div><div class="box">{html.escape(r["reasoning"])}</div><div class="lab">final channel</div><div class="box">{html.escape(r["response"])}</div></div>'
                    f'<div class="col"><div class="lab">honest-arm CoT, same question (answer {h["answer"]})</div><div class="box">{html.escape(h["reasoning"])}</div>'
                    f'<div class="lab">honest-arm final channel</div><div class="box">{html.escape(h["response"])}</div></div></div>'
                    f'<details><summary>monitor reply</summary><div class="box">{html.escape(judged(mo))}</div></details>'
                    + (f'<details><summary>verbalization judge reply</summary><div class="box">{html.escape(judged(ve))}</div></details>' if ve else "") + "</article>")
    n_flip = sum(k == "flip" for k, _, _ in items)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>WMDP sandbagging flips: {html.escape(args.run)}</title><style>{CSS}</style></head><body><nav><h3>{html.escape(args.run)}</h3>"
           f"<div class='meta'>{n_flip} flips (red; with hand labels: gold stated, pink silent, light gray unclear, gray genuine error), {args.per_arm} random correct rows per arm (green), honest-arm rows the monitor flagged (purple). Left: the instructed arm's prompt, CoT and final. "
           f"Right: the same question's honest-arm CoT. Each item also gives {html.escape(args.compare)}'s answer on the same question and arm.</div>{''.join(nav)}</nav><main>{''.join(body)}</main></body></html>")
    args.out.write_text(doc)
    print(f"{len(items)} items ({n_flip} flips) -> {args.out} ({len(doc) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
