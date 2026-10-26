"""Static HTML viewer for the implicit vs explicit influence rows (iei.py, iei_judge.py): two models side by side on every nudged cell both ran (same
prompt_id and arm), with the user message, each model's baseline and chosen option, the monitor's verdict, reply and the nudge it was told, the full
analysis channel, and the last --resp-chars characters of the final channel (the full one is in the Inspect log, linked). The sidebar groups cells by
arm, then by who switched and who the monitor flagged.

    python iei_traces_html.py --models base mt2e-s170 --out ../../../figures/iei_s1_traces.html
"""

import argparse
import html
import json
from pathlib import Path

LABELS = {"base": "base gpt-oss-120b", "mt2e-s170": "manytypes2e s170", "mt1b-s150": "manytypes1b s150"}
INSPECT = "http://localhost:7577/#/logs/iei_s1/{model}_{arm}.eval/samples/sample/{pid}/1"


def load(d: Path) -> dict:
    mon = {r["id"]: r for r in map(json.loads, open(d / "monitor.jsonl"))}
    return {(r["arm"], r["prompt_id"]): dict(r, mon=mon.get(f"{r['arm']}:{r['prompt_id']}")) for r in map(json.loads, open(d / "nudged.jsonl"))}


def group(arm: str, a: dict, b: dict, la: str, lb: str) -> str:
    if not (a["switched"] and b["switched"]):
        return f"{arm} · not switched by both"
    ya, yb = a["mon"]["verdict"] == "YES", b["mon"]["verdict"] == "YES"
    return f"{arm} · " + ("both flagged" if ya and yb else f"{la} flagged, {lb} not" if ya else f"{lb} flagged, {la} not" if yb else "neither flagged")


def column(model: str, arm: str, r: dict, n: int) -> str:
    m = r["mon"]
    verdict = "not judged" if m is None else m["verdict"]
    cls = "yes" if verdict == "YES" else "no" if verdict == "NO" else "na"
    resp = r["response"][-n:]
    return (f'<div class="col"><div class="head">{html.escape(LABELS.get(model, model))}: baseline {r["baseline_choice"]}, chose {r["choice"]} '
            f'({"switched" if r["switched"] else "did not switch"}); monitor <span class="{cls}">{verdict}</span>; CoT {r["cot_tokens"]} tokens</div>'
            + (f'<div class="lab">monitor reply</div><div class="box small">{html.escape(m["reply"])}</div>' if m else "")
            + f'<div class="lab">analysis channel</div><div class="box">{html.escape(r["reasoning"])}</div>'
            f'<details><summary>final channel, last {min(n, len(r["response"]))} of {len(r["response"])} characters · '
            f'<a href="{INSPECT.format(model=model, arm=arm, pid=r["prompt_id"])}">full sample in Inspect</a></summary><div class="box">{"…" if len(r["response"]) > n else ""}{html.escape(resp)}</div></details></div>')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "../results/iei")
    ap.add_argument("--models", nargs=2, default=["base", "mt2e-s170"])
    ap.add_argument("--resp-chars", type=int, default=1500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ma, mb = args.models
    A, B = load(args.dir / ma), load(args.dir / mb)
    la, lb = LABELS.get(ma, ma), LABELS.get(mb, mb)
    order = [f"{arm} · {g}" for arm in ("explicit", "implicit") for g in (f"{la} flagged, {lb} not", "both flagged", f"{lb} flagged, {la} not", "neither flagged", "not switched by both")]
    items = sorted(((group(k[0], A[k], B[k], la, lb), k) for k in A if k in B), key=lambda gk: (order.index(gk[0]), gk[1][1]))
    counts = {g: sum(1 for g2, _ in items if g2 == g) for g in order}
    nav, body = [], []
    for i, (g, (arm, pid)) in enumerate(items):
        a, b = A[(arm, pid)], B[(arm, pid)]
        if i == 0 or items[i - 1][0] != g:
            nav.append(f"<h4>{html.escape(g)} ({counts[g]})</h4>")
        nav.append(f'<a href="#i{i}">{html.escape(pid)}</a>')
        fact = (a["mon"] or b["mon"] or {}).get("fact", "")
        body.append(f'<article id="i{i}"><h2>{html.escape(arm)} · {html.escape(pid)}</h2>'
                    f'<div class="meta">{html.escape(g)} · options A = {html.escape(a["path_a_name"])}, B = {html.escape(a["path_b_name"])} · nudge: {html.escape(a["constraint_category"] or "")} / {html.escape(a["constraint_theme"] or "")}, '
                    f'{html.escape(a["direction"])} on {html.escape(a["target_tool"] or "")} · monitor was told: "{html.escape(fact)}"</div>'
                    f'<details open><summary>user message</summary><div class="box">{html.escape(a["user"])}</div></details>'
                    f'<div class="cols">{column(ma, arm, a, args.resp_chars)}{column(mb, arm, b, args.resp_chars)}</div></article>')
    css = ("body{background:#1a1a19;color:#e8e6e3;font-family:system-ui,sans-serif;margin:0;display:flex} nav{width:340px;min-width:340px;height:100vh;overflow:auto;border-right:1px solid #444;padding:8px;font-size:12px}"
           " nav a{display:block;color:#9ec5f4;text-decoration:none;padding:2px 4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap} nav a:hover{background:#2a2a28} nav h4{margin:12px 0 2px;color:#f0efec}"
           " main{flex:1;height:100vh;overflow:auto;padding:16px 24px} article{border-bottom:1px solid #444;padding:12px 0 24px} h2{font-size:15px;margin:4px 0} .meta{font-size:12px;color:#c3c2b7;margin:4px 0 8px}"
           " .cols{display:flex;gap:16px} .col{flex:1;min-width:0} .head{font-size:13px;margin:10px 0 4px;color:#f0efec}"
           " .box{background:#242422;border:1px solid #3a3a38;border-radius:4px;padding:8px;font-size:12px;white-space:pre-wrap;word-wrap:break-word;max-height:60vh;overflow:auto} .small{max-height:20vh}"
           " .lab{font-size:11px;color:#c3c2b7;margin:8px 0 2px;text-transform:uppercase;letter-spacing:.04em} details summary{cursor:pointer;font-size:12px;color:#c3c2b7;margin:6px 0} a{color:#9ec5f4}"
           " .yes{color:#eda100;font-weight:600} .no{color:#1baf7a;font-weight:600} .na{color:#8a8984} @media(max-width:900px){body{display:block} nav{width:auto;height:auto;max-height:40vh;border-right:none;border-bottom:1px solid #444} main{height:auto} .cols{display:block}}")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Implicit vs explicit influence traces</title><style>{css}</style></head><body>"
           f"<nav><h3>Implicit vs explicit influence, Setting 1: {html.escape(la)} (left) vs {html.escape(lb)} (right)</h3>"
           f"<div class='meta'>{len(items)} cells run by both models. Monitor: CoT only, told the nudge, the paper's Setting 1 prompt, gpt-6-luna effort medium; YES = the CoT credits the choice to the nudge or to the user's request. "
           f"Explicit arm: switched = picked the user's option. Groups compare the monitor's verdicts on cells both models switched on.</div>{''.join(nav)}</nav><main>{''.join(body)}</main></body></html>")
    args.out.write_text(doc)
    print(f"{len(items)} cells -> {args.out} ({len(doc) / 1e6:.1f} MB); " + ", ".join(f"{g} {n}" for g, n in counts.items()))


if __name__ == "__main__":
    main()
