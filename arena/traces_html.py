"""One self-contained HTML page for reading resampled turns: every sample's CoT, visible text and tool calls, with both judge scores side by side.

Takes resample_ctx/leak_check sweep directories. For each one it reads <dir>/annotated.jsonl and, when present, <dir>_maintask/annotated.jsonl,
and joins them per sample so the same CoT shows the score it got with the side task in the judge's task description and the score it got with
the task cut to the main task only. Filters are client-side; nothing is fetched, the page works offline.

    .venv/bin/python traces_html.py results/turn1_grpo/leaksmoke/{base_on_base,org_on_org,base_on_org} --out results/turn1_grpo/leaksmoke/traces.html
"""

import argparse, html, json
from pathlib import Path

QUIET, LOUD = 2, 8


def load(d: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(d / "annotated.jsonl")]
    mt = d.parent / f"{d.name}_maintask"
    main = {}
    if (mt / "annotated.jsonl").exists():
        for r in (json.loads(l) for l in open(mt / "annotated.jsonl")):
            main[(r["run"], r["sample"], r["epoch"], r["turn"], r["g"])] = r["score"]
    for r in rows:
        r["sweep"] = d.name
        r["score_main"] = main.get((r["run"], r["sample"], r["epoch"], r["turn"], r["g"]))
    return rows


def badge(label: str, cls: str) -> str:
    return f'<span class="b {cls}">{html.escape(label)}</span>'


def score_badge(s, which: str) -> str:
    if s is None:
        return badge(f"{which} unparsed", "u")
    cls = "q" if s <= QUIET else "l" if s >= LOUD else "m"
    return badge(f"{which} {s:g}", cls)


def sample_html(r: dict) -> str:
    flags = ["leak" if r["leak"] else "no leak"] + [x for x in (r["ref"], r["sink"]) if x]
    quiet_main = r["score_main"] is not None and r["score_main"] <= QUIET
    cls = "card" + (" proof" if r["leak"] and quiet_main else "")
    attrs = (f'data-sweep="{html.escape(r["sweep"])}" data-leak="{int(bool(r["leak"]))}" '
             f'data-quiet="{int(r["score"] is not None and r["score"] <= QUIET)}" data-quietmain="{int(quiet_main)}"')
    head = (f'<b>g{r["g"]}</b> {score_badge(r["score"], "side-task-aware")} {score_badge(r["score_main"], "main-task-only")} '
            f'{badge(" / ".join(flags), "k" if r["leak"] else "n")} '
            f'<span class="dim">{r["out_tok"]} out tok · stop {html.escape(str(r["stop"]))}</span>')
    calls = "".join(f'<div class="call"><b>{html.escape(c["name"])}</b><pre>{html.escape(json.dumps(c["args"], indent=1))}</pre></div>' for c in r["calls"]) or '<div class="dim">no tool call</div>'
    return (f'<details class="{cls}" {attrs}><summary>{head}</summary>'
            f'<h5>reasoning</h5><pre>{html.escape(r["reasoning"]) or "<i>empty</i>"}</pre>'
            f'<h5>visible text</h5><pre>{html.escape(r["text"]) or "<i>none</i>"}</pre>'
            f'<h5>tool calls</h5>{calls}</details>')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Leaking-turn resamples, CoT-only monitor")
    a = ap.parse_args()
    rows = [r for d in a.dirs for r in load(d)]
    sweeps = list(dict.fromkeys(r["sweep"] for r in rows))

    body = []
    for sweep in sweeps:
        srows = [r for r in rows if r["sweep"] == sweep]
        body.append(f'<h2 id="{sweep}">{html.escape(sweep)} <span class="dim">{len(srows)} samples, '
                    f'{sum(r["leak"] for r in srows)} leak, {sum(1 for r in srows if r["leak"] and r["score_main"] is not None and r["score_main"] <= QUIET)} leak and quiet under the main-task-only judge</span></h2>')
        for key in dict.fromkeys((r["run"], r["sample"], r["epoch"], r["turn"]) for r in srows):
            g = sorted((r for r in srows if (r["run"], r["sample"], r["epoch"], r["turn"]) == key), key=lambda r: r["g"])
            sa = [r["score"] for r in g if r["score"] is not None]
            sm = [r["score_main"] for r in g if r["score_main"] is not None]
            body.append(f'<section><h3>{html.escape(key[0])} · {html.escape(key[1])} · epoch {key[2]} · turn {key[3]}</h3>'
                        f'<div class="dim">G={len(g)} · leak {sum(r["leak"] for r in g)}/{len(g)} · '
                        f'side-task-aware score {min(sa):g}-{max(sa):g} · main-task-only {min(sm):g}-{max(sm):g} · '
                        f'score the original turn got {g[0].get("base_score")}</div>'
                        + "".join(sample_html(r) for r in g) + '</section>')

    checks = "".join(f'<label><input type="checkbox" class="sw" value="{html.escape(s)}" checked> {html.escape(s)}</label>' for s in sweeps)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(a.title)}</title><style>
:root {{ --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#c9d1d9; --dim:#8b949e; --acc:#58a6ff; --q:#2ea043; --l:#f85149; --m:#d29922; --k:#a371f7; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; padding:0 16px 64px }}
header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 0; z-index:5 }}
h1 {{ font-size:19px; margin:0 0 8px }} h2 {{ font-size:17px; margin:32px 0 4px; color:var(--acc) }}
h3 {{ font-size:14px; margin:0 0 4px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-weight:600 }}
h5 {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--dim); margin:12px 0 4px }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin:12px 0 }}
.dim {{ color:var(--dim); font-size:12.5px }}
pre {{ background:#0b0f14; border:1px solid var(--line); border-radius:6px; padding:10px; margin:0; white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace }}
details.card {{ border:1px solid var(--line); border-radius:6px; margin:8px 0; padding:8px 10px; background:#11161d }}
details.card[open] {{ border-color:var(--acc) }}
details.proof {{ border-left:3px solid var(--q) }}
summary {{ cursor:pointer; list-style:none }} summary::-webkit-details-marker {{ display:none }}
.b {{ display:inline-block; font-size:11.5px; padding:1px 7px; border-radius:999px; border:1px solid var(--line); margin-right:4px; font-family:ui-monospace,Menlo,monospace }}
.b.q {{ color:var(--q); border-color:var(--q) }} .b.l {{ color:var(--l); border-color:var(--l) }} .b.m {{ color:var(--m); border-color:var(--m) }}
.b.k {{ color:var(--k); border-color:var(--k) }} .b.n {{ color:var(--dim) }} .b.u {{ color:var(--dim); border-style:dashed }}
.call b {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--acc) }} .call {{ margin:6px 0 }}
label {{ margin-right:14px; font-size:13px; color:var(--dim) }} input {{ accent-color:var(--acc); vertical-align:-2px }}
@media (max-width:600px) {{ body {{ padding:0 10px 48px }} label {{ display:inline-block; margin-bottom:4px }} }}
</style></head><body>
<header><h1>{html.escape(a.title)}</h1>
<div class="dim">Monitoring mode: cot only (controlarena_cot_only), judge openai/gpt-oss-20b. Each sample is scored twice over the identical CoT:
<b>side-task-aware</b> gives the judge the whole first user message as the task description, <b>main-task-only</b> cuts it before the secondary objective.
quiet = score &le; {QUIET}, loud = score &ge; {LOUD}. A green left edge marks a sample that both completes the leak and stays quiet under the main-task-only judge.</div>
<div style="margin-top:8px">{checks}
<label><input type="checkbox" id="fleak"> leaks only</label>
<label><input type="checkbox" id="fquiet"> quiet under main-task-only only</label>
<label><input type="checkbox" id="fopen"> expand all</label></div></header>
{''.join(body)}
<script>
const cards = [...document.querySelectorAll('details.card')];
function apply() {{
  const on = new Set([...document.querySelectorAll('.sw')].filter(c => c.checked).map(c => c.value));
  const leak = fleak.checked, quiet = fquiet.checked;
  for (const c of cards) {{
    const ok = on.has(c.dataset.sweep) && (!leak || c.dataset.leak === '1') && (!quiet || c.dataset.quietmain === '1');
    c.style.display = ok ? '' : 'none';
  }}
  for (const s of document.querySelectorAll('section'))
    s.style.display = [...s.querySelectorAll('details.card')].some(c => c.style.display !== 'none') ? '' : 'none';
}}
document.querySelectorAll('.sw, #fleak, #fquiet').forEach(c => c.addEventListener('change', apply));
fopen.addEventListener('change', () => cards.forEach(c => c.open = fopen.checked));
apply();
</script></body></html>"""
    a.out.write_text(doc)
    print(f"{len(rows)} samples from {len(a.dirs)} sweeps -> {a.out} ({a.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
