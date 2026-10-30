"""One self-contained HTML page from a tutor.py result: each turn's reasoning, text, tool calls and results, next to the tutor's pass-1 labels and its
pass-2 assessment and scores. Works offline.

    .venv/bin/python tutor_html.py results/tutor/easy-diggcount-s1 --monitor logs/gptoss64-covert2-easy/monitor_maintask_cot_only/gptoss64-covert2-easy/monitor.jsonl

--monitor: an rl.monitor monitor.jsonl; each turn then also shows that monitor's real score for comparison with the tutor's suspiciousness.
"""

import argparse, html, json
from pathlib import Path

SCORES = ["suspiciousness", "reasoning", "side_task_progress", "persistence", "awareness"]


def e(x):
    return html.escape(x or "")


def badge(label, cls=""):
    return f'<span class="b {cls}">{e(label)}</span>'


def score_badge(k, v):
    if v is None:
        return badge(f"{k} null", "n")
    hi = v >= 7 if k != "suspiciousness" else v <= 2
    lo = v <= 3 if k != "suspiciousness" else v >= 7
    return badge(f"{k} {v}", "g" if hi else "r" if lo else "m")


def pre(title, body, open_=True):
    return f'<details {"open" if open_ else ""}><summary>{e(title)}</summary><pre>{e(body)}</pre></details>'


def turn_html(t, lab, p2, mon):
    left = [pre("reasoning", t["reasoning"] or "(empty)"), pre("visible text", t["text"]) if t["text"].strip() else ""]
    left += [pre(f"tool call: {c['name']}", json.dumps(c["args"], indent=1)) for c in t["calls"]]
    left += [pre("tool result" if a["role"] == "tool" else a["role"], a["text"], len(a["text"]) < 1500) for a in t["after"]]
    tags = badge("refused", "r") if lab["refused"] else ""
    tags += badge("side-task relevant", "k") if lab["side_task_relevant"] else badge("not relevant", "n")
    if mon is not None:
        tags += badge(f"real monitor {mon['score']}", "r" if (mon["score"] or 0) >= 7 else "g" if (mon["score"] or 0) <= 2 else "m")
    right = [f"<div>{tags}</div>"]
    if p2:
        right.append("<div>" + "".join(score_badge(k, p2["json"].get(k)) for k in SCORES) + "</div>")
        right.append(pre("tutor assessment", p2["content"]))
        if p2["thinking"]:
            right.append(pre("tutor thinking", p2["thinking"], False))
        right.append(f'<div class="dim">{p2["in_tok"]} in / {p2["out_tok"]} out tok, ${p2["cost"] or 0:.4f}</div>')
    else:
        right.append('<div class="dim">not rated in pass 2</div>')
    return f'<section id="t{t["turn"]}"><h3>turn {t["turn"]}</h3><div class="row"><div class="col">{"".join(left)}</div><div class="col tut">{"".join(right)}</div></div></section>'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--monitor", type=Path)
    a = ap.parse_args()
    r = json.loads((a.dir / "tutor.json").read_text())
    p2 = {int(k): v for k, v in r["pass2"].items()}
    labels = {int(k): v for k, v in r["labels"].items()}
    o = r["outcome"]
    rows = [json.loads(l) for l in open(a.monitor)] if a.monitor else []
    mon = {q["turn"]: q for q in rows if q["log"].endswith(Path(r["log"]).name) and q["sample"] == r["sample_id"] and q["epoch"] == r["epoch"]}
    assert not a.monitor or mon, f"no rows for this episode in {a.monitor}"
    mdesc = f'<br>real monitor: {e(rows[0]["monitor"])}, mode {e(rows[0]["mode"])}, judge {e(rows[0]["judge_model"])} ({e(str(a.monitor))})' if mon else ""
    nav = " ".join(f'<a href="#t{k}">{k}{"*" if k in p2 else ""}</a>' for k in range(len(r["turns"])))
    head = (f'<div class="dim">{e(r["sample_id"])} · epoch {r["epoch"]} · tutor {e(r["model"])} effort {e(str(r.get("effort")))} · tutor cost ${r["cost"]:.3f}<br>'
            f'main_pass {o["main_pass"]} · side_leaked {o["side_leaked"]} · final_reveal {o["final_reveal"]} · tier {e(str(o["tier"]))} · '
            f'source {e(str(o["source"]))} · sink {e(str(o["sink_req"]))}{mdesc}<br>turns (* = rated in pass 2): {nav}</div>')
    top = (pre("main task (as given to tutor)", r["main"], False) + pre("side task (as given to tutor)", r["side"], False)
           + pre("organism system prompt", r["system"], False) + pre("organism first user message", r["first_user"], False)
           + pre("pass 1 output", r["pass1"]["content"], False) + (pre("pass 1 thinking", r["pass1"]["thinking"], False) if r["pass1"]["thinking"] else "")
           + pre("spec given to tutor", r["spec"], False))
    body = "".join(turn_html(t, labels[t["turn"]], p2.get(t["turn"]), mon.get(t["turn"]) if mon else None) for t in r["turns"])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tutor ratings</title><style>
:root {{ --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#c9d1d9; --dim:#8b949e; --acc:#58a6ff; --g:#2ea043; --r:#f85149; --m:#d29922; --k:#a371f7; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 ui-sans-serif,system-ui,sans-serif; padding:0 16px 64px }}
header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 0; z-index:5 }}
h1 {{ font-size:19px; margin:0 0 6px }} h3 {{ font-size:14px; margin:0 0 6px; font-family:ui-monospace,Menlo,monospace }}
a {{ color:var(--acc) }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin:12px 0 }}
.row {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }} .col {{ min-width:0 }}
.tut {{ border-left:2px solid var(--line); padding-left:12px }}
@media (max-width:900px) {{ .row {{ grid-template-columns:1fr }} .tut {{ border-left:none; padding-left:0; border-top:2px solid var(--line); padding-top:8px }} }}
.dim {{ color:var(--dim); font-size:12.5px }}
summary {{ cursor:pointer; color:var(--dim); font-size:12px; text-transform:uppercase; letter-spacing:.06em; margin:8px 0 4px }}
pre {{ background:#0b0f14; border:1px solid var(--line); border-radius:6px; padding:10px; margin:0; white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace,Menlo,monospace; max-height:600px; overflow:auto }}
.b {{ display:inline-block; font-size:11.5px; padding:1px 7px; border-radius:999px; border:1px solid var(--line); margin:0 4px 4px 0; font-family:ui-monospace,Menlo,monospace }}
.b.g {{ color:var(--g); border-color:var(--g) }} .b.r {{ color:var(--r); border-color:var(--r) }} .b.m {{ color:var(--m); border-color:var(--m) }}
.b.k {{ color:var(--k); border-color:var(--k) }} .b.n {{ color:var(--dim) }}
</style></head><body><header><h1>Tutor ratings</h1>{head}</header>{top}{body}</body></html>"""
    out = a.dir / "viewer.html"
    out.write_text(doc)
    print(f"-> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
