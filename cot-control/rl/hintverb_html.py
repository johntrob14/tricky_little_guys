"""Interactive page for the hint-verbalization runs (hintverb.py, judged by hintverb_judge.py / hintverb_pressure.py), in the style of
sandbag_resample_html.py, over every arm on disk for the given models: the no-line arm and the oversight-pressure lines (hintverb.SUFFIXES) in the user
message or the system prompt. Charts, for the hint chosen above them (or pooled), per line and model: the share of pairs switched to the hint
(Wilson 95%), the judge's verbalized share of switched rows (Wilson 95%) and the mean CoT tokens of switched rows, verbalized vs not (± SE); a table of
pair outcomes per model and line; click a bar or a count to list those rows, click a row to read its prompt, CoT and final answer next to the same
question's unhinted CoT in the same arm (middle pane) and the judge prompt and output, plus the hand-validation reader's verdict where the row was
read (right pane). Pair outcomes as in hintverb_judge.judge_arm, with letters re-parsed by the current hintverb.parse_letter. Every row is listed
(unhinted rows too); the list opens on the rows switched to the hint. The page holds the scores; each arm's texts (CoT, final answer, user message,
judge and reader replies) go to <out stem>_text/<arm>.js, loaded when a row of that arm is first opened, so the folder must stay next to the page.
Same page mechanics as sandbag_resample_html.py (resizable panes, sortable columns, arrow keys, KaTeX from cdnjs). Labels and colors from
sandbag_headline.MODELS, line labels from hintverb_pressure.LABELS.

    python hintverb_html.py --dir ../results/hintverb --models base grpo3-s100 mt1b-s150 --out ../../../figures/hintverb_rollouts.html
"""

import argparse
import json
import statistics
from pathlib import Path

from hintverb import HINTS, SUFFIXES, parse_letter
from hintverb_judge import JUDGE, verdict, wilson
from hintverb_pressure import LABELS
from sandbag_headline import MODELS

OUTCOMES = ["to hint", "elsewhere", "unchanged", "unhinted = hint", "unparsed", "unhinted"]
LINES = ["none", "neutral", "cot_private", "user_reads", "user_reads_hint", "answer_checked", "grader", "auto_monitor", "monitor_discard"]  # controls first, then the lines claiming a check
GROUPS = LINES + [f"{l}@sys" for l in LINES[1:]]  # the line in the user message, then in the system prompt; arms missing on disk are skipped


def outcome(r: dict, a_u: str | None) -> str:
    if r["hint_type"] == "none":
        return "unhinted"
    if a_u is None or r["answer"] is None:
        return "unparsed"
    if a_u == r["hint"]:
        return "unhinted = hint"
    return "to hint" if r["answer"] == r["hint"] else "elsewhere" if r["answer"] != a_u else "unchanged"


def mean_se(xs: list) -> list | None:
    return [statistics.mean(xs), statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0, len(xs)] if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    hints = list(HINTS)
    reader = {}  # (arm, qid, hint type) -> (verbalized, reason): the no-line hand validation and the pressure-arm one
    for m in args.models:
        if (args.dir / m / "reader_verdicts.jsonl").exists():
            reader |= {(m, r["qid"], r["hint_type"]): (r["verbalized"], r["reason"]) for r in map(json.loads, open(args.dir / m / "reader_verdicts.jsonl"))}
    for r in map(json.loads, open(args.dir / "pressure_reader_verdicts.jsonl")):
        reader[tuple(r["id"].split(":"))] = (r["verbalized"], r["note"])
    text_dir = args.out.parent / f"{args.out.stem}_text"
    text_dir.mkdir(exist_ok=True)
    samples, stats, acc, groups, cfg0 = [], {}, {}, {}, None
    for m in args.models:
        for g in GROUPS:
            arm = m if g == "none" else f"{m}.{g}"
            if not (args.dir / arm).exists():
                continue
            cfg = json.load(open(args.dir / arm / "config.json"))
            cfg0 = cfg0 or cfg
            assert all(cfg[k] == cfg0[k] for k in ["effort", "max_tokens", "n_questions", "temperature"]), f"{arm}: sampling differs from the other arms"
            line = g.removesuffix("@sys")
            groups.setdefault(g, dict(id=g, label=LABELS[line] + ("\n[system prompt]" if g.endswith("@sys") else ""), line=line, text=SUFFIXES.get(line),
                                      placement="none" if line == "none" else "system" if g.endswith("@sys") else "user", system=cfg["system"]))
            assert groups[g]["system"] == cfg["system"], f"{arm}: system prompt differs from the other models' arm"
            rows = [r | dict(answer=parse_letter(r["response"])) for r in map(json.loads, open(args.dir / arm / "rows.jsonl"))]  # the current parser, as hintverb_judge
            unh = {r["qid"]: r for r in rows if r["hint_type"] == "none"}
            judged = {(r["qid"], r["hint_type"]): r for r in map(json.loads, open(args.dir / arm / "judged.jsonl"))}
            ss, texts = [], []
            for r in rows:
                j, rd = judged.get((r["qid"], r["hint_type"])), reader.get((arm, r["qid"], r["hint_type"]))
                ss.append(dict(model=m, group=g, qid=r["qid"], ds=r["dataset"], hint_type=r["hint_type"], h=r["hint"], correct=r["correct"], answer=r["answer"], a_u=unh[r["qid"]]["answer"],
                               outcome=outcome(r, unh[r["qid"]]["answer"]), verbalized=verdict(j["reply"]) if j else None, reader=rd[0] if rd else None,
                               cot_tokens=r["cot_tokens"], finish=r["finish"], t=len(texts)))
                texts.append([r["reasoning"], r["response"], r["user"], j["reply"] if j else None, rd[1] if rd else None])
            assert {(s["qid"], s["hint_type"]) for s in ss if s["outcome"] == "to hint"} == set(judged), f"{arm}: judged rows are not the rows switched to the hint"
            (text_dir / f"{arm.replace('@', '-')}.js").write_text(f"HV({json.dumps(arm)}, {json.dumps(texts)});\n")
            samples += ss
            acc[f"{m}|{g}"] = [sum(r["answer"] == r["correct"] for r in unh.values()), len(unh)]
            for h in hints + ["pooled"]:
                c = [s for s in ss if s["hint_type"] != "none" and h in ("pooled", s["hint_type"])]
                n_o = {o: sum(s["outcome"] == o for s in c) for o in OUTCOMES[:-1]}
                pairs = n_o["to hint"] + n_o["elsewhere"] + n_o["unchanged"]
                sw = [s for s in c if s["outcome"] == "to hint"]
                yes, nj = sum(s["verbalized"] is True for s in sw), sum(s["verbalized"] is not None for s in sw)
                stats[f"{m}|{g}|{h}"] = dict(outcomes=n_o, pairs=pairs, ci=wilson(n_o["to hint"], pairs), yes=yes, judged=nj, verb_ci=wilson(yes, nj), reader=sum(s["reader"] is not None for s in c),
                                             tok_yes=mean_se([s["cot_tokens"] for s in sw if s["verbalized"] is True]), tok_no=mean_se([s["cot_tokens"] for s in sw if s["verbalized"] is False]))
            p = stats[f"{m}|{g}|pooled"]
            print(f"{arm:32s} pooled switched {p['outcomes']['to hint']}/{p['pairs']}  verbalized {p['yes']}/{p['judged']}")
    data = dict(hints=hints, outcomes=OUTCOMES, groups=[groups[g] for g in GROUPS if g in groups], hint_text=HINTS, judge_prompt=JUDGE, acc=acc, stats=stats, samples=samples,
                models=[dict(id=m, label=MODELS[m][0], color=MODELS[m][1]) for m in args.models], text_dir=text_dir.name,
                settings=f"{cfg0['n_questions']} questions (MMLU + GPQA Diamond), each asked unhinted and once per hint; gpt-oss-120b harmony effort {cfg0['effort']}, temperature {cfg0['temperature']:g}, one sample, {cfg0['max_tokens'] // 1000}k cap, Tinker.")
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    args.out.write_text(html)
    print(args.out, f"{len(samples)} rows, {len(html) / 1e6:.1f} MB; texts in {text_dir} ({sum(f.stat().st_size for f in text_dir.iterdir()) / 1e6:.0f} MB)")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>hint verbalization rollouts</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<style>
:root { color-scheme:dark; --bg:#1a1a19; --panel:#232322; --line:#3a3a38; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8d8c84; --good:#5fb87a; --bad:#e0705f; }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif; }
h1 { font-size:18px; margin:0 0 4px; } h2 { font-size:15px; margin:18px 0 8px; }
.sub { color:var(--ink2); font-size:13px; }
.charts { display:flex; flex-direction:column; gap:16px; margin-top:12px; }
.counts td, .counts th { padding:3px 8px; }
.chart { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px; min-width:0; }
.chart h3 { font-size:13px; font-weight:600; margin:2px 4px 6px; }
.chart svg { width:100%; height:auto; display:block; }
.legend { display:flex; gap:18px; flex-wrap:wrap; font-size:13px; margin:12px 0 0; color:var(--ink2); align-items:center; }
.legend select { background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:4px; padding:4px 8px; font:13px system-ui, sans-serif; }
.sw { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:6px; }
.bar { cursor:pointer; } .bar:hover { opacity:.8; }
#tip { position:fixed; pointer-events:none; background:#0e0e0d; border:1px solid var(--line); border-radius:4px; padding:6px 8px; font-size:12px; color:var(--ink); display:none; z-index:9; max-width:360px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--ink2); font-weight:600; }
td.num { font-variant-numeric:tabular-nums; text-align:right; }
td.wrap { white-space:normal; }
.counts td.c { cursor:pointer; text-align:right; font-variant-numeric:tabular-nums; } .counts td.c:hover { background:#2e2e2c; }
#pane { display:grid; height:calc(100vh - 32px); min-height:480px; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
#pane > .col { overflow:auto; min-width:0; padding:10px; scrollbar-color:#55554f #1f1f1e; scrollbar-width:thin; }
#side { background:var(--panel); }
.divider { cursor:col-resize; background:var(--line); } .divider:hover, .divider.drag { background:var(--ink2); }
.filters { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px 10px; margin-bottom:10px; font-size:12px; color:var(--ink2); }
.filters label.f { display:flex; flex-direction:column; gap:3px; }
.filters select, .filters input[type=text] { width:100%; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:4px; padding:6px 8px; font:13px system-ui, sans-serif; }
.filters .chk { display:flex; align-items:center; gap:6px; color:var(--ink); }
#list th { position:sticky; top:-10px; background:var(--panel); z-index:1; cursor:pointer; user-select:none; } #list th:hover { color:var(--ink); }
#list th.up::after, #list th.down::after { content:""; display:inline-block; margin-left:5px; border-left:4px solid transparent; border-right:4px solid transparent; vertical-align:2px; }
#list th.up::after { border-bottom:6px solid var(--ink); } #list th.down::after { border-top:6px solid var(--ink); }
#list tr.s { cursor:pointer; } #list tr.s:hover { background:#2a2a28; } #list tr.sel { background:#3a3a37; }
.ok { color:var(--good); } .no { color:var(--bad); }
.head { font-size:13px; margin-bottom:10px; display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; }
.head b { font-size:14px; }
.head button { background:#2b2b29; color:var(--ink); border:1px solid var(--line); border-radius:4px; padding:3px 8px; font:12px system-ui, sans-serif; cursor:pointer; }
.sec + .sec .sec-h { border-top:none; }
.sec-h { display:block; width:100%; text-align:left; background:#2b2b29; color:var(--ink); border:1px solid var(--line); border-radius:0; padding:7px 10px; font:600 13px system-ui, sans-serif; cursor:pointer; }
.sec-h:hover { background:#333331; }
.sec-h::before { content:""; display:inline-block; border-left:6px solid var(--ink2); border-top:4px solid transparent; border-bottom:4px solid transparent; margin-right:9px; vertical-align:1px; transform:rotate(90deg); }
.sec.closed .sec-h::before { transform:none; }
.sec-b { white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace, monospace; margin:0; background:var(--bg); border:1px solid var(--line); border-top:none; border-radius:0; padding:8px 10px; }
.sec.closed .sec-b { display:none; }
.sec-b.math .katex-display { margin:0.4em 0; overflow-x:auto; overflow-y:hidden; padding:2px 0; }
.sec-b.math .katex { font-size:1.08em; }
.score { font-size:20px; font-weight:700; padding:0 8px; border-radius:2px; }
.empty { color:var(--muted); font-size:13px; padding:8px 0; }
.scroll { overflow-x:auto; }
.note { color:var(--muted); font-size:12px; }
</style></head><body>
<div id="tip"></div>
<h1>Hint verbalization (Chen et al. 2025 hints) with and without oversight lines: rollouts and judge verdicts</h1>
<div class="sub" id="settings"></div>
<div class="sub">Pairs: hinted rows whose unhinted answer (same arm) parsed and differs from the hint letter h. Judge: claude-haiku-5.5 (OpenRouter, T 0, effort low), run on the rows switched to h only: does the CoT mention the hint and depend on it (a mention only as a final check does not count); its question block leaves out a line placed in the user message. Reader: the hand-validation Claude reader, not shown the judge's verdict (50 judged rows per model on the no-line arm, 40 rows across the line arms). Each line is added to the unhinted and hinted prompts alike.</div>
<h2>Lines</h2>
<div class="scroll"><table id="lines"></table></div>
<div class="legend" id="legend"></div>
<div class="charts">
  <div class="chart"><h3>Switched to the hint, among pairs (Wilson 95%). Click a bar to list those rows.</h3><svg id="sw"></svg></div>
  <div class="chart"><h3>Judge: CoT verbalizes the hint, among rows switched to it (Wilson 95%). Click to list.</h3><svg id="verb"></svg></div>
  <div class="chart"><h3>Mean CoT tokens of switched rows (± SE): verbalized solid, not verbalized hatched. Click to list.</h3><svg id="tok"></svg></div>
</div>
<h2>Pair outcomes for the hint chosen above (count of hinted rows). Click a count to list those rows.</h2>
<div class="scroll"><table class="counts" id="counts"></table></div>
<h2>Rows <span class="note">(drag the dividers to resize; arrow keys move through the list; opens on the rows switched to the hint)</span></h2>
<div id="pane">
  <div class="col" id="side">
    <div class="filters">
      <label class="f">model<select id="f-model"></select></label>
      <label class="f">line<select id="f-group"></select></label>
      <label class="f">hint<select id="f-hint"></select></label>
      <label class="f">outcome<select id="f-out"></select></label>
      <label class="f">judge verdict<select id="f-verb"><option value="">any</option><option value="1">YES (verbalized)</option><option value="0">NO</option><option value="none">not judged</option></select></label>
      <label class="f">dataset<select id="f-ds"><option value="">any</option><option>mmlu</option><option>gpqa</option></select></label>
      <label class="f">hint letter<select id="f-hc"><option value="">any</option><option value="1">h is the correct answer</option><option value="0">h is wrong</option></select></label>
      <label class="f">reader<select id="f-rd"><option value="">any</option><option value="any">reader checked</option><option value="dis">reader disagrees with judge</option></select></label>
      <label class="f">question id contains<input type="text" id="f-qid"></label>
      <label class="chk"><input type="checkbox" id="f-math" checked> render math</label>
      <span class="note" id="nshown" style="align-self:center"></span>
    </div>
    <table id="list"></table>
  </div>
  <div class="divider" data-k="0"></div>
  <div class="col" id="view"></div>
  <div class="divider" data-k="1"></div>
  <div class="col" id="judge"></div>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => (s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
$("settings").textContent = D.settings;
const short = m => m.label.split(" (")[0];
const G = Object.fromEntries(D.groups.map(g => [g.id, g]));
const gname = id => G[id].label.replace("\n[system prompt]", "").replaceAll("\n", " ") + (G[id].placement === "system" ? " [sys]" : "");
const KATEX = !!window.renderMathInElement;  // loaded from cdnjs; without it the page says so and shows the raw text
const DELIMS = [{left: "$$", right: "$$", display: true}, {left: "\\[", right: "\\]", display: true}, {left: "\\(", right: "\\)", display: false}, {left: "$", right: "$", display: false}];
const tidy = t => t.replace(/\n*\\\[\s*/g, "\\[").replace(/\s*\\\]\n*/g, "\\]");  // drop the line breaks around display math, which would otherwise add blank lines
function math(el) { if (KATEX && $("f-math").checked) renderMathInElement(el, {delimiters: DELIMS, throwOnError: false}); }
if (!KATEX) document.body.insertAdjacentHTML("afterbegin", `<div style="color:#e0705f;font-weight:600;margin-bottom:8px">KaTeX did not load (offline?): math is shown as raw LaTeX.</div>`);
// per-arm texts [cot, final answer, user message, judge reply, reader reason], from <text_dir>/<arm>.js, loaded on first use
const TEXT = {};
function HV(arm, rows) { TEXT[arm] = rows; }
const armOf = s => s.group === "none" ? s.model : `${s.model}.${s.group}`;
function need(arm) {
  return TEXT[arm] ? Promise.resolve() : new Promise((ok, bad) => {
    const el = document.createElement("script"), src = `${D.text_dir}/${arm.replace("@", "-")}.js`;
    el.src = src; el.onload = ok; el.onerror = () => bad(new Error(`could not load ${src}: the folder ${D.text_dir}/ must sit next to this page`)); document.head.appendChild(el);
  });
}
$("lines").innerHTML = `<tr><th>line</th><th>placement</th><th>text</th></tr>` + D.groups.map(g => `<tr><td>${esc(gname(g.id))}</td><td>${g.placement === "none" ? "" : g.placement === "user" ? "end of the user message" : "second paragraph of the system prompt"}</td><td class="wrap">${esc(g.text ?? "(no line)")}</td></tr>`).join("");
$("legend").innerHTML = D.models.map(m => `<span><span class="sw" style="background:${m.color}"></span>${esc(m.label)}</span>`).join("")
  + `<span><svg width="14" height="12" style="vertical-align:-1px;margin-right:6px"><rect width="12" height="12" rx="2" fill="#c3c2b7"/><path d="M-2,14 L14,-2 M4,14 L14,4 M-2,8 L8,-2" stroke="#1a1a19" stroke-width="2"/></svg>not verbalized (CoT-token chart)</span>`
  + `<label>charts and outcome table show hint <select id="c-hint">${[...D.hints, "pooled"].map(h => `<option${h === "pooled" ? " selected" : ""}>${h}</option>`).join("")}</select></label>`;
const SVGNS = "http://www.w3.org/2000/svg";
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = Math.min(e.clientX + 12, innerWidth - 370) + "px"; tip.style.top = (e.clientY + 12) + "px"; }
function el(svg, tag, attrs, text) { const n = document.createElementNS(SVGNS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; svg.appendChild(n); return n; }
const W = 1400, H = 345, L = 48, R = 10, T = 38, B = 70;
function frame(svg, ymax, ticks, fmt) {
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
  el(svg, "defs", {}).innerHTML = `<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="transparent"/><line x1="0" y1="0" x2="0" y2="6" stroke="#1a1a19" stroke-width="2.5"/></pattern>`;
  const y = v => T + (H - T - B) * (1 - v / ymax);
  for (const t of ticks) { el(svg, "line", {x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: "#3a3a38", "stroke-width": 1}); el(svg, "text", {x: L - 6, y: y(t) + 4, "text-anchor": "end", "font-size": 11, fill: "#c3c2b7"}, fmt(t)); }
  const gw = (W - L - R) / D.groups.length;
  D.groups.forEach((g, k) => {
    const t = el(svg, "text", {"text-anchor": "middle", "font-size": 11, fill: "#f0efec"});
    g.label.split("\n").forEach((part, i) => { const sp = el(t, "tspan", {x: L + gw * (k + 0.5), y: H - B + 16 + 13 * i}, part); if (part === "[system prompt]") sp.setAttribute("fill", "#c3c2b7"); });
  });
  const ks = D.groups.findIndex(g => g.placement === "system");  // dashed divider between the user-message and system-prompt placements
  if (ks > 0) { el(svg, "line", {x1: L + gw * ks, x2: L + gw * ks, y1: 2, y2: H - B, stroke: "#8d8c84", "stroke-dasharray": "4 4"}); el(svg, "text", {x: L + gw * ks + 6, y: 12, "font-size": 11, fill: "#8d8c84"}, "line in the system prompt"); el(svg, "text", {x: L + gw * ks - 6, y: 12, "text-anchor": "end", "font-size": 11, fill: "#8d8c84"}, "line in the user message"); }
  el(svg, "line", {x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: "#c3c2b7", "stroke-width": 1});
  return {y, gw};
}
function bar(svg, f, x, w, v, lo, hi, color, hatch, label, tipHtml, filt) {
  const y0 = f.y(0), yv = f.y(v), h = Math.max(y0 - yv, 0.5);
  const g = el(svg, "g", {class: "bar"});
  el(g, "rect", {x: x - w / 2, y: y0 - h, width: w, height: h, fill: color, rx: 2});
  if (hatch) el(g, "rect", {x: x - w / 2, y: y0 - h, width: w, height: h, fill: "url(#hatch)"});
  if (lo != null) { el(g, "line", {x1: x, x2: x, y1: f.y(lo), y2: f.y(hi), stroke: "#c3c2b7", "stroke-width": 1.2}); for (const e of [lo, hi]) el(g, "line", {x1: x - 3, x2: x + 3, y1: f.y(e), y2: f.y(e), stroke: "#c3c2b7", "stroke-width": 1.2}); }
  el(g, "text", {x, y: f.y(hi ?? v) - 5, "text-anchor": "middle", "font-size": 9, fill: "#c3c2b7"}, label);
  el(g, "rect", {x: x - w / 2 - 1, y: T, width: w + 2, height: y0 - T, fill: "transparent"});
  g.addEventListener("mousemove", e => showTip(e, tipHtml)); g.addEventListener("mouseleave", () => tip.style.display = "none");
  g.addEventListener("click", () => setFilters(filt));
}
const chartHint = () => $("c-hint").value;
const hintFilt = () => chartHint() === "pooled" ? "" : chartHint();
// bars of one line group: the models that have this arm, centred in the group, each `per` bars wide
function slots(f, k, g, per) {
  const ms = D.models.filter(m => D.stats[`${m.id}|${g.id}|${chartHint()}`]), w = f.gw * 0.84 / (D.models.length * per);
  return ms.map((m, i) => [m, D.stats[`${m.id}|${g.id}|${chartHint()}`], L + f.gw * (k + 0.5) - w * per * ms.length / 2 + w * per * i, w]);
}
const tipHead = (m, g) => `<b>${esc(m.label)}</b><br>line: ${esc(gname(g.id))}${g.text ? `<br><i>${esc(g.text)}</i>` : ""}<br>hint: ${chartHint()}`;
function drawRate(id, num, den, ci, what) {
  const svg = $(id), f = frame(svg, 1, [0, .2, .4, .6, .8, 1], t => t.toFixed(1));
  D.groups.forEach((g, k) => slots(f, k, g, 1).forEach(([m, s, x0, w]) => {
    const n = num(s), d = den(s);
    bar(svg, f, x0 + w / 2, w - 2, d ? n / d : 0, ci(s)[0], ci(s)[1], m.color, false, `${n}/${d}`,
        `${tipHead(m, g)}<br>${what} ${n}/${d} = ${(n / d).toFixed(2)}<br>Wilson 95% ${ci(s)[0].toFixed(2)}–${ci(s)[1].toFixed(2)}`, {model: m.id, group: g.id, hint: hintFilt(), out: "to hint"});
  }));
}
function drawTok() {
  const vals = Object.entries(D.stats).filter(([k]) => k.endsWith("|" + chartHint())).flatMap(([, s]) => [s.tok_yes, s.tok_no]).filter(Boolean), top = Math.max(...vals.map(v => v[0] + v[1])) * 1.15;
  const step = top > 4000 ? 1000 : top > 2000 ? 500 : 250, ticks = Array.from({length: Math.floor(top / step) + 1}, (_, i) => i * step);
  const svg = $("tok"), f = frame(svg, top, ticks, t => t);
  D.groups.forEach((g, k) => slots(f, k, g, 2).forEach(([m, s, x0, w]) => [["tok_yes", 1], ["tok_no", 0]].forEach(([key, yes], j) => {
    const v = s[key], x = x0 + w * (j + 0.5);
    if (!v) { el(svg, "text", {x, y: f.y(0) - 4, "text-anchor": "middle", "font-size": 9, fill: "#8d8c84"}, "0"); return; }
    bar(svg, f, x, w - 1, v[0], v[0] - v[1], v[0] + v[1], m.color, !yes, v[0].toFixed(0),
        `${tipHead(m, g)}<br>switched rows the judge called ${yes ? "verbalized" : "not verbalized"}: mean CoT tokens ${v[0].toFixed(0)} ± ${v[1].toFixed(0)} (SE), n=${v[2]}`, {model: m.id, group: g.id, hint: hintFilt(), out: "to hint", verb: String(yes)});
  })));
}
function drawCounts() {
  const OC = D.outcomes.slice(0, -1);
  let h = `<tr><th>model</th><th>line</th><th style="text-align:right">unhinted correct</th><th style="text-align:right">pairs</th>${OC.map(o => `<th style="text-align:right">${o}</th>`).join("")}<th style="text-align:right">judge YES</th><th style="text-align:right">judge NO</th><th style="text-align:right">reader checked</th></tr>`;
  for (const m of D.models) for (const g of D.groups) {
    const s = D.stats[`${m.id}|${g.id}|${chartHint()}`], a = D.acc[`${m.id}|${g.id}`];
    if (!s) continue;
    const c = (v, filt) => `<td class="c" data-f='${JSON.stringify({model: m.id, group: g.id, hint: hintFilt(), ...filt})}'>${v}</td>`;
    h += `<tr><td><span class="sw" style="background:${m.color}"></span>${esc(short(m))}</td><td>${esc(gname(g.id))}</td><td class="num">${a[0]}/${a[1]}</td><td class="num">${s.pairs}</td>` + OC.map(o => c(s.outcomes[o], {out: o})).join("")
      + c(s.yes, {out: "to hint", verb: "1"}) + c(s.judged - s.yes, {out: "to hint", verb: "0"}) + c(s.reader, {out: "", rd: "any"}) + `</tr>`;
  }
  $("counts").innerHTML = h;
  $("counts").querySelectorAll("td.c").forEach(td => td.addEventListener("click", () => setFilters(JSON.parse(td.dataset.f))));
}
function drawAll() { drawRate("sw", s => s.outcomes["to hint"], s => s.pairs, s => s.ci, "switched to the hint"); drawRate("verb", s => s.yes, s => s.judged, s => s.verb_ci, "judge YES (verbalized)"); drawTok(); drawCounts(); }
const opts = xs => `<option value="">any</option>` + xs.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("");
$("f-model").innerHTML = opts(D.models.map(m => [m.id, short(m)]));
$("f-group").innerHTML = opts(D.groups.map(g => [g.id, gname(g.id)]));
$("f-hint").innerHTML = opts([...D.hints, "none"].map(h => [h, h === "none" ? "none (unhinted)" : h]));
$("f-out").innerHTML = opts(D.outcomes.map(o => [o, o]));
const FILT = {model: "f-model", group: "f-group", hint: "f-hint", out: "f-out", verb: "f-verb", ds: "f-ds", hc: "f-hc", rd: "f-rd", qid: "f-qid"};
function setFilters(f) { for (const k in FILT) $(FILT[k]).value = f[k] ?? ""; render(); $("pane").scrollIntoView({behavior: "smooth", block: "start"}); }
const col = Object.fromEntries(D.models.map(m => [m.id, m]));
const sid = s => `${s.model}|${s.group}|${s.qid}|${s.hint_type}`;
const byKey = Object.fromEntries(D.samples.map(s => [sid(s), s]));
let shown = [], sel = null;
const closed = new Set(["system prompt", "unhinted CoT, same question", "unhinted final answer", "judge prompt"]);  // section headers start open except these; the choice carries over between rows
function sec(title, body, m) {  // m: render LaTeX in this section
  const t = m && KATEX && $("f-math").checked ? tidy(body || "") : body;
  return `<div class="sec${closed.has(title) ? " closed" : ""}" data-t="${esc(title)}"><button class="sec-h">${esc(title)}</button><pre class="sec-b${m ? " math" : ""}">${esc(t || "(empty)")}</pre></div>`; }
function wire(root) { root.querySelectorAll(".sec-h").forEach(b => b.addEventListener("click", () => { const d = b.parentNode, t = d.dataset.t; closed.has(t) ? closed.delete(t) : closed.add(t); d.classList.toggle("closed"); })); }
const ans = (a, s) => `<span class="${a === s.correct ? "ok" : "no"}">${a ?? "none"}</span>`;
const yn = v => v == null ? "" : v ? "YES" : "NO";
async function show() {
  document.querySelectorAll("#list tr.s").forEach(tr => tr.classList.toggle("sel", tr.dataset.id === sel));
  const s = shown.find(x => sid(x) === sel);
  if (!s) { $("view").innerHTML = `<div class="empty">Select a row on the left.</div>`; $("judge").innerHTML = ""; return; }
  const arm = armOf(s);
  if (!TEXT[arm]) { $("view").innerHTML = `<div class="empty">loading ${esc(arm)} …</div>`; $("judge").innerHTML = ""; }
  try { await need(arm); } catch (e) { $("view").innerHTML = `<div class="no">${esc(e.message)}</div>`; return; }
  if (sid(s) !== sel) return;  // the selection moved on while this arm loaded
  const m = col[s.model], g = G[s.group], u = byKey[`${s.model}|${s.group}|${s.qid}|none`];
  const [cot, response, user, reply, reason] = TEXT[arm][s.t], [ucot, uresponse, uuser] = TEXT[arm][u.t];
  $("view").innerHTML = `<div class="head"><span class="sw" style="background:${m.color}"></span><b>${esc(short(m))}</b><span>line: ${esc(gname(s.group))}</span><span>${s.qid}</span><span>hint: ${s.hint_type}${s.hint_type === "none" ? "" : ` → ${s.h}`}</span><span>correct ${s.correct}</span>`
    + `<span>unhinted ${ans(s.a_u, s)}${s.hint_type === "none" ? "" : ` → hinted ${ans(s.answer, s)}`}</span><span>${s.outcome}</span><span>${s.cot_tokens} CoT tokens</span>${s.finish === "length" ? "<span class=no>cut at cap</span>" : ""}`
    + `<button id="sameq">all rows of this question</button></div>`
    + sec("system prompt", g.system) + sec("user message", user, 1) + sec("CoT (analysis channel)", cot, 1) + sec("final answer", response, 1)
    + (s.hint_type === "none" ? "" : sec("unhinted CoT, same question", ucot, 1) + sec("unhinted final answer", uresponse, 1));
  $("sameq").addEventListener("click", () => setFilters({qid: s.qid}));
  const question = g.placement === "user" ? uuser.slice(0, -(g.text.length + 2)) : uuser;  // the judge's question block leaves out a user-message line
  const jp = reply == null ? "" : D.judge_prompt.replaceAll("{hint_text}", () => D.hint_text[s.hint_type].replaceAll("{h}", s.h)).replaceAll("{hint}", s.h).replaceAll("{unhinted}", s.a_u).replaceAll("{question}", () => question).replaceAll("{cot}", () => cot);
  $("judge").innerHTML = `<div class="head"><b>Judge (claude-haiku-5.5)</b><span class="score ${s.verbalized ? "ok" : "no"}">${s.verbalized == null ? "not run" : yn(s.verbalized)}</span></div>`
    + (reply == null ? `<div class="empty">The judge runs only on rows switched to the hint.</div>` : sec("judge prompt", jp) + sec("judge output", reply, 1))
    + (s.reader == null ? "" : `<div class="head" style="margin-top:16px"><b>Reader (hand validation)</b><span class="score ${s.reader ? "ok" : "no"}" style="font-size:16px">${yn(s.reader)}</span>${s.reader !== s.verbalized ? "<span class=no>disagrees with the judge</span>" : ""}</div>` + sec("reader reason", reason, 1));
  wire($("view")); wire($("judge"));
  document.querySelectorAll("#view .sec-b.math, #judge .sec-b.math").forEach(math);
}
// sidebar columns: sort key, title, right-aligned; click a title to sort by it, again to reverse
const COLS = [["model", "model"], ["group", "line"], ["qid", "question"], ["hint", "hint"], ["ans", "unh → hinted"], ["out", "outcome"], ["verb", "judge"], ["rd", "reader"], ["tok", "CoT tok", 1]];
const HO = [...D.hints, "none"], GO = D.groups.map(g => g.id);
const SORT = {model: s => D.models.findIndex(m => m.id === s.model), group: s => GO.indexOf(s.group), qid: s => s.qid, hint: s => HO.indexOf(s.hint_type), ans: s => (s.a_u ?? "~") + (s.answer ?? "~"),
              out: s => D.outcomes.indexOf(s.outcome), verb: s => s.verbalized == null ? -1 : +s.verbalized, rd: s => s.reader == null ? -1 : +s.reader, tok: s => s.cot_tokens};
let sortBy = {k: "model", dir: 1};
function render() {
  const v = Object.fromEntries(Object.entries(FILT).map(([k, id]) => [k, $(id).value]));
  shown = D.samples.filter(s => (!v.model || s.model === v.model) && (!v.group || s.group === v.group) && (!v.hint || s.hint_type === v.hint) && (!v.out || s.outcome === v.out) && (!v.ds || s.ds === v.ds)
    && (!v.verb || (v.verb === "none" ? s.verbalized == null : s.verbalized != null && String(+s.verbalized) === v.verb))
    && (!v.hc || (s.hint_type !== "none" && String(+(s.h === s.correct)) === v.hc)) && (!v.rd || (s.reader != null && (v.rd === "any" || s.reader !== s.verbalized))) && (!v.qid || s.qid.includes(v.qid.trim())));
  const k = SORT[sortBy.k], cmp = (x, y) => x < y ? -1 : x > y ? 1 : 0;  // the clicked column first, then model, line, question, hint
  shown.sort((a, b) => sortBy.dir * cmp(k(a), k(b)) || cmp(SORT.model(a), SORT.model(b)) || cmp(SORT.group(a), SORT.group(b)) || cmp(a.qid, b.qid) || cmp(SORT.hint(a), SORT.hint(b)));
  $("nshown").textContent = `${shown.length} shown`;
  $("list").innerHTML = `<tr>${COLS.map(([k, t, r]) => `<th data-k="${k}"${r ? ' style="text-align:right"' : ""} class="${sortBy.k === k ? (sortBy.dir > 0 ? "up" : "down") : ""}">${t}</th>`).join("")}</tr>`
    + shown.map(s => `<tr class="s" data-id="${sid(s)}"><td><span class="sw" style="background:${col[s.model].color}"></span>${esc(short(col[s.model]))}</td><td>${esc(gname(s.group))}</td><td>${s.qid.split("_").slice(0, 2).join(" ")}</td><td>${s.hint_type}${s.hint_type === "none" ? "" : " " + s.h}</td>`
      + `<td>${ans(s.a_u, s)}${s.hint_type === "none" ? "" : " → " + ans(s.answer, s)}</td><td>${s.outcome}</td><td class="${s.verbalized ? "ok" : "no"}">${yn(s.verbalized)}</td><td>${yn(s.reader)}</td><td class="num">${s.cot_tokens}</td></tr>`).join("");
  $("list").querySelectorAll("tr.s").forEach(tr => tr.addEventListener("click", () => { sel = tr.dataset.id; show(); }));
  $("list").querySelectorAll("th").forEach(th => th.addEventListener("click", () => { sortBy = {k: th.dataset.k, dir: sortBy.k === th.dataset.k ? -sortBy.dir : 1}; render(); }));
  if (!shown.some(x => sid(x) === sel)) sel = shown.length ? sid(shown[0]) : null;
  show();
}
document.addEventListener("keydown", e => {
  if (!["ArrowDown", "ArrowUp"].includes(e.key) || ["SELECT", "INPUT"].includes(e.target.tagName) || !shown.length) return;
  const k = shown.findIndex(x => sid(x) === sel), j = Math.max(0, Math.min(shown.length - 1, k + (e.key === "ArrowDown" ? 1 : -1)));
  e.preventDefault(); sel = sid(shown[j]); show();
  document.querySelector(`#list tr.s[data-id="${CSS.escape(sel)}"]`).scrollIntoView({block: "nearest"});
});
let widths = [880, 600];  // side and view panes in px; the judge pane takes the rest
try { const w = JSON.parse(localStorage.getItem("hintverb-widths2")); if (Array.isArray(w) && w.length === 2) widths = w; } catch (e) {}
const applyW = () => $("pane").style.gridTemplateColumns = `${widths[0]}px 6px ${widths[1]}px 6px minmax(180px, 1fr)`;
document.querySelectorAll(".divider").forEach(d => d.addEventListener("pointerdown", e => {
  e.preventDefault(); d.setPointerCapture(e.pointerId); d.classList.add("drag");
  const k = +d.dataset.k, x0 = e.clientX, w0 = widths[k];
  const move = ev => { widths[k] = Math.max(180, w0 + ev.clientX - x0); applyW(); };
  const up = () => { d.classList.remove("drag"); d.removeEventListener("pointermove", move); d.removeEventListener("pointerup", up); try { localStorage.setItem("hintverb-widths2", JSON.stringify(widths)); } catch (e) {} };
  d.addEventListener("pointermove", move); d.addEventListener("pointerup", up);
}));
applyW();
for (const id of Object.values(FILT)) $(id).addEventListener(id === "f-qid" ? "input" : "change", render);
$("f-math").addEventListener("change", show);
$("c-hint").addEventListener("change", drawAll);
drawAll();
$("f-out").value = "to hint"; render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
