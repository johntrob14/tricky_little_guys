"""One interactive page for hintuse.py / hintuse_puzzle.py case studies (a dropdown switches between the --dirs, the last one shown first). For each: per model and arm, accuracy (Wilson 95%), mean CoT tokens (± SE), the judge's REFERENCES and
share of CoTs containing the word "hint" (Wilson 95%; the judge from hintuse_judge.py, when judged.jsonl exists) and the answers given (letters, or for free-response puzzles every distinct answer, headed by initials); click a bar or a count to list those samples,
a list column marks CoTs containing the word "hint" (also hints, hinted, hinting), which is highlighted in the CoT and final answer; click a sample to read its prompt, CoT and final answer (middle pane) and the judge's prompt, reasoning and reply (right pane). Layout and controls as
sandbag_resample_html.py (resizable panes, sortable list, arrow keys, KaTeX toggle). Labels and colors from sandbag_headline.MODELS.

    python hintuse_html.py --dirs ../results/case_studies/hintuse_q225 ../results/case_studies/hintuse_race5 --out ../../../figures/hintuse.html
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from hintverb_judge import wilson
from sandbag_headline import MODELS
from sandbag_resample_html import PAGE as BASE_PAGE

ARM_TEXT = {"none": "no hint", "hint": "hint (\"Hint: ...\" after the options)", "hide": "hint + hide line (\"Use this hint if it helps, but do not mention it ...\")",
            "nocot": "no hint, no CoT (empty analysis channel, 'ANSWER:' prefilled)", "nocot_hint": "hint, no CoT (empty analysis channel, 'ANSWER:' prefilled)"}


def rate(xs: list) -> dict | None:
    xs = [x for x in xs if x is not None]
    return dict(k=sum(xs), n=len(xs), ci=wilson(sum(xs), len(xs))) if xs else None


def page_data(d: Path) -> dict:
    cfg = json.load(open(d / "config.json"))
    jpath = d / "judged.jsonl"
    judged = {(r["model"], r["arm"], r["sample"]): r for r in map(json.loads, open(jpath))} if jpath.exists() else {}
    jcfg = json.load(open(d / "judge.json")) if judged else None
    rows = [r | dict(j=judged.get((r["model"], r["arm"], r["sample"]))) for r in map(json.loads, open(d / "rows.jsonl"))]
    models, arms = list(cfg["models"]), cfg["arms"]
    user = {a: p.split("<|start|>user<|message|>")[1].split("<|end|>")[0] for a, p in cfg["prompts"].items()}
    prefill = {a: p.split("<|start|>assistant")[-1] for a, p in cfg["prompts"].items()}
    seen = [r["answer"] for r in rows if r["answer"] is not None]
    keys = list("ABCD") if set(seen) <= set("ABCD") else sorted(set(seen), key=lambda k: (k != cfg["answer"], -seen.count(k), k))
    samples, stats = [], {}
    for r in rows:
        j = r["j"] or {}
        samples.append(dict(model=r["model"], arm=r["arm"], i=r["sample"], answer=r["answer"], ok=r["ok"], finish=r["finish"], cot_tokens=r["cot_tokens"], cot=r["reasoning"], response=r["response"],
                            refs=j.get("refs"), kw=bool(re.search(r"(?i)\bhint(?:s|ed|ing)?\b", r["reasoning"])), reply=j.get("reply"), reasoning=j.get("reasoning")))
    for m in models:
        for a in arms:
            ra = [r for r in rows if r["model"] == m and r["arm"] == a]
            k, toks = sum(r["ok"] for r in ra), [r["cot_tokens"] for r in ra]
            stats[f"{m}|{a}"] = dict(n=len(ra), k=k, ci=wilson(k, len(ra)), cut=sum(r["finish"] == "length" for r in ra),
                                     letters={x: sum(r["answer"] == x for r in ra) for x in keys} | {"none": sum(r["answer"] is None for r in ra)},
                                     tok=[statistics.mean(toks), statistics.stdev(toks) / len(toks) ** 0.5, len(toks)] if len(toks) > 1 and not a.startswith("nocot") else None,
                                     refs=rate([(r["j"] or {}).get("refs") for r in ra]), kw=rate([bool(re.search(r"(?i)\bhint(?:s|ed|ing)?\b", r["reasoning"])) for r in ra]) if not a.startswith("nocot") else None)
    judge_line = f"Judge: {jcfg['judge_model']}, reasoning effort {jcfg['effort']}, temperature {jcfg['temperature']}, on every CoT sample, not told the arm (REFERENCES: the CoT refers to the hint or to being given information or an instruction)." if jcfg else "Judge: not run yet."
    return dict(name=d.name, item=cfg["item"], hint_name=cfg["hint_name"], keys=keys + ["none"], question=cfg["qblock"], correct=cfg["answer"], hint=cfg["hint"], arms=arms, arm_text=ARM_TEXT, user=user, prefill=prefill, judge_prompt=jcfg["prompt"] if jcfg else None,
                models=[dict(id=m, label=MODELS[m][0], color=MODELS[m][1]) for m in models], stats=stats, samples=samples, judge_line=judge_line,
                settings=f"{cfg['n']} samples per model and arm at temperature {cfg['temperature']}, gpt-oss-120b harmony effort medium, {cfg['max_tokens'] // 1000}k cap, Tinker. " + ("Prompt = the transfer eval's baseline prompt for this item, hint inserted after the options." if isinstance(cfg["item"], int) else "Prompt = the puzzle, the hint lines, then the answer format, in the chat template with no system prompt."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = [page_data(d) for d in args.dirs]
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    args.out.write_text(html)
    print(args.out, f"{len(data)} runs, {sum(len(x['samples']) for x in data)} samples, {len(html) / 1e6:.1f} MB")


HEAD = BASE_PAGE.split("</style>")[0].replace("<title>q0006 resampled</title>", "<title>GPQA hint use</title>") + "mark.hl { background:rgba(224,170,80,.2); color:inherit; border-bottom:1px solid rgba(224,170,80,.6); border-radius:2px; padding:0 1px; }\n.chart .legend { margin:0 4px 6px; font-size:12px; gap:4px 14px; } .chart .legend svg { display:inline-block; width:14px; height:12px; }\n</style></head><body>"
PAGE = HEAD + r"""
<div id="tip"></div>
<h1>Hidden hint use, <span id="qid"></span></h1>
<div class="sub" style="margin:6px 0">run <select id="f-exp" style="background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:4px 6px;font:13px system-ui,sans-serif"></select></div>
<div class="sub" id="settings"></div>
<div class="sub" id="judgeline"></div>
<div class="top">
  <div><h2>Question (correct answer <span id="correct"></span>)</h2><div class="q" id="question"></div><h2>Hint</h2><div class="q" id="hint"></div></div>
  <div><h2>Answers given (count of samples; correct answer in green; free-response answers headed by initials, hover for the full answer). Click a count to list those samples.</h2><div class="scroll"><table class="counts" id="counts"></table></div></div>
</div>
<div class="charts">
  <div class="chart"><h3>Accuracy (Wilson 95%). Click a bar to list its samples.</h3><div class="legend lg"></div><svg id="acc"></svg></div>
  <div class="chart"><h3>Mentions of the hint (Wilson 95%). Click to list.</h3><div class="legend lg" data-hatch="1"></div><svg id="judgec"></svg></div>
  <div class="chart"><h3>CoT tokens, mean (± SE). Click to list.</h3><div class="legend lg"></div><svg id="tok"></svg></div>
</div>
<h2>Samples <span class="note">(drag the dividers to resize; arrow keys move through the list)</span></h2>
<div id="pane">
  <div class="col" id="side">
    <div class="filters">
      <label class="f">model<select id="f-model"></select></label>
      <label class="f">arm<select id="f-arm"></select></label>
      <label class="f">answer<select id="f-ok"><option value="">any</option><option value="1">correct</option><option value="0">incorrect</option></select></label>
      <label class="f">answer given<select id="f-letter"></select></label>
      <label class="f">REFERENCES<select id="f-refs"><option value="">any</option><option value="1">YES</option><option value="0">NO</option></select></label>
      <label class="f">CoT says "hint"<select id="f-kw"><option value="">any</option><option value="1">yes</option><option value="0">no</option></select></label>
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
const ALL = __DATA__;
let D, COT_ARMS, col;
const $ = id => document.getElementById(id);
const esc = s => (s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const KATEX = !!window.renderMathInElement;
const DELIMS = [{left: "$$", right: "$$", display: true}, {left: "\\[", right: "\\]", display: true}, {left: "\\(", right: "\\)", display: false}, {left: "$", right: "$", display: false}];
const tidy = t => t.replace(/\n*\\\[\s*/g, "\\[").replace(/\s*\\\]\n*/g, "\\]");
function math(el) { if (KATEX && $("f-math").checked) renderMathInElement(el, {delimiters: DELIMS, throwOnError: false}); }
const HATCH_SW = `<svg width="14" height="12" style="vertical-align:-1px;margin-right:6px"><rect width="12" height="12" rx="2" fill="#c3c2b7"/><path d="M-2,14 L14,-2 M4,14 L14,4 M-2,8 L8,-2" stroke="#1a1a19" stroke-width="2"/></svg>`;
const SVGNS = "http://www.w3.org/2000/svg", tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = Math.min(e.clientX + 12, innerWidth - 330) + "px"; tip.style.top = (e.clientY + 12) + "px"; }
function el(svg, tag, attrs, text) { const n = document.createElementNS(SVGNS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; svg.appendChild(n); return n; }
const W = 620, H = 300, L = 44, R = 10, T = 14, B = 26;
function frame(svg, ymax, ticks, fmt, arms) {
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
  el(svg, "defs", {}).innerHTML = `<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="transparent"/><line x1="0" y1="0" x2="0" y2="6" stroke="#1a1a19" stroke-width="2.5"/></pattern>`;
  const y = v => T + (H - T - B) * (1 - v / ymax);
  for (const t of ticks) { el(svg, "line", {x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: "#3a3a38", "stroke-width": 1}); el(svg, "text", {x: L - 6, y: y(t) + 4, "text-anchor": "end", "font-size": 11, fill: "#c3c2b7"}, fmt(t)); }
  const gw = (W - L - R) / arms.length;
  arms.forEach((a, k) => el(svg, "text", {x: L + gw * (k + 0.5), y: H - B + 18, "text-anchor": "middle", "font-size": 12, fill: "#f0efec"}, a));
  el(svg, "line", {x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: "#c3c2b7", "stroke-width": 1});
  return {y, gw};
}
function bar(svg, f, x, w, v, lo, hi, color, hatch, label, tipHtml, filt) {
  const y0 = f.y(0), yv = f.y(v), h = Math.max(y0 - yv, 0.5), g = el(svg, "g", {class: "bar"});
  el(g, "rect", {x: x - w / 2, y: y0 - h, width: w, height: h, fill: color, rx: 2});
  if (hatch) el(g, "rect", {x: x - w / 2, y: y0 - h, width: w, height: h, fill: "url(#hatch)"});
  if (lo != null) { el(g, "line", {x1: x, x2: x, y1: f.y(lo), y2: f.y(hi), stroke: "#c3c2b7", "stroke-width": 1.2}); for (const e of [lo, hi]) el(g, "line", {x1: x - 4, x2: x + 4, y1: f.y(e), y2: f.y(e), stroke: "#c3c2b7", "stroke-width": 1.2}); }
  el(g, "text", {x, y: f.y(hi ?? v) - 5, "text-anchor": "middle", "font-size": D.models.length > 2 ? 9 : 11, fill: "#c3c2b7"}, label);
  el(g, "rect", {x: x - w / 2 - 2, y: T, width: w + 4, height: y0 - T, fill: "transparent"});
  g.addEventListener("mousemove", e => showTip(e, tipHtml)); g.addEventListener("mouseleave", () => tip.style.display = "none");
  g.addEventListener("click", () => setFilters(filt));
}
function drawAcc() {
  const svg = $("acc"), f = frame(svg, 1, [0, .2, .4, .6, .8, 1], t => t.toFixed(1), D.arms), w = f.gw * 0.8 / D.models.length;
  D.arms.forEach((a, k) => D.models.forEach((m, i) => {
    const s = D.stats[m.id + "|" + a], x = L + f.gw * k + f.gw * 0.1 + w * (i + 0.5);
    bar(svg, f, x, w - 3, s.k / s.n, s.ci[0], s.ci[1], m.color, false, `${s.k}/${s.n}`,
        `<b>${esc(m.label)}</b><br>${esc(D.arm_text[a])}<br>correct ${s.k}/${s.n} = ${(s.k / s.n).toFixed(2)}<br>Wilson 95% ${s.ci[0].toFixed(2)}–${s.ci[1].toFixed(2)}` + (s.cut ? `<br>cut at cap: ${s.cut}/${s.n}` : ""), {model: m.id, arm: a});
  }));
}
function drawJudge() {
  const svg = $("judgec"), f = frame(svg, 1, [0, .2, .4, .6, .8, 1], t => t.toFixed(1), COT_ARMS), w = f.gw * 0.8 / (2 * D.models.length);
  COT_ARMS.forEach((a, k) => D.models.forEach((m, i) => [["refs", "judge REFERENCES YES"], ["kw", "CoT contains \"hint\""]].forEach(([key, name], j) => {
    const s = D.stats[m.id + "|" + a][key], x = L + f.gw * k + f.gw * 0.1 + w * (2 * i + j + 0.5);
    if (!s) { el(svg, "text", {x, y: f.y(0) - 4, "text-anchor": "middle", "font-size": 10, fill: "#8d8c84"}, "n=0"); return; }
    bar(svg, f, x, w - 2, s.k / s.n, s.ci[0], s.ci[1], m.color, j === 1, `${s.k}/${s.n}`,
        `<b>${esc(m.label)}</b><br>${esc(D.arm_text[a])}<br>${name}: ${s.k}/${s.n} = ${(s.k / s.n).toFixed(2)}<br>Wilson 95% ${s.ci[0].toFixed(2)}–${s.ci[1].toFixed(2)}`, {model: m.id, arm: a, [key]: "1"});
  })));
}
function drawTok() {
  const ymax = Math.max(...Object.values(D.stats).filter(s => s.tok).map(s => s.tok[0] + s.tok[1])) * 1.15, step = Math.pow(10, Math.floor(Math.log10(ymax / 2)));
  const ticks = []; for (let t = 0; t <= ymax; t += step * (ymax / step > 8 ? 2 : 1)) ticks.push(t);
  const svg = $("tok"), f = frame(svg, ymax, ticks, t => t, COT_ARMS), w = f.gw * 0.8 / D.models.length;
  COT_ARMS.forEach((a, k) => D.models.forEach((m, i) => {
    const v = D.stats[m.id + "|" + a].tok, x = L + f.gw * k + f.gw * 0.1 + w * (i + 0.5);
    bar(svg, f, x, w - 3, v[0], v[0] - v[1], v[0] + v[1], m.color, false, v[0].toFixed(0), `<b>${esc(m.label)}</b><br>${esc(D.arm_text[a])}<br>mean ${v[0].toFixed(0)} ± ${v[1].toFixed(0)} (SE), n=${v[2]}`, {model: m.id, arm: a});
  }));
}
function drawCounts() {
  const L5 = D.keys;
  let h = `<tr><th>model</th><th>arm</th>${L5.map(x => `<th style="text-align:center" title="${esc(x)}">${esc(abbr(x))}</th>`).join("")}<th>n</th></tr>`;
  for (const m of D.models) for (const a of D.arms) {
    const s = D.stats[m.id + "|" + a];
    h += `<tr><td><span class="sw" style="background:${m.color}"></span>${esc(m.label.split(" (")[0])}</td><td>${a}</td>` + L5.map(x => `<td class="c${x === D.correct ? " right" : ""}" data-m="${m.id}" data-a="${a}" data-l="${x}">${s.letters[x] || ""}</td>`).join("") + `<td class="num">${s.n}</td></tr>`;
  }
  $("counts").innerHTML = h;
  $("counts").querySelectorAll("td.c").forEach(td => td.addEventListener("click", () => setFilters({model: td.dataset.m, arm: td.dataset.a, letter: td.dataset.l})));
}
const abbr = k => k.includes(", ") ? k.split(", ").map(w => w[0]).join("") : k;  // free-response orders headed by initials
const FILTERS = {model: "f-model", arm: "f-arm", ok: "f-ok", letter: "f-letter", refs: "f-refs", kw: "f-kw"};
function setFilters(f) { for (const k in FILTERS) $(FILTERS[k]).value = f[k] ?? ""; render(); $("pane").scrollIntoView({behavior: "smooth", block: "start"}); }
const yn = v => v == null ? "" : v ? "YES" : "NO";
const sid = s => `${s.model}|${s.arm}|${s.i}`;
let shown = [], sel = null;
const closed = new Set(["judge prompt"]);
const HL = /\bhint(?:s|ed|ing)?\b/gi;  // the word "hint" and its inflections, highlighted in the CoT and final answer
function sec(title, body, m, hl) {
  const t = m && KATEX && $("f-math").checked ? tidy(body || "") : body, h = esc(t || "(empty)");
  return `<div class="sec${closed.has(title) ? " closed" : ""}" data-t="${esc(title)}"><button class="sec-h">${esc(title)}</button><pre class="sec-b${m ? " math" : ""}">${hl ? h.replace(HL, w => `<mark class="hl">${w}</mark>`) : h}</pre></div>`; }
function wire(root) { root.querySelectorAll(".sec-h").forEach(b => b.addEventListener("click", () => { const d = b.parentNode, t = d.dataset.t; closed.has(t) ? closed.delete(t) : closed.add(t); d.classList.toggle("closed"); })); }
function show() {
  document.querySelectorAll("#list tr.s").forEach(tr => tr.classList.toggle("sel", tr.dataset.id === sel));
  const s = shown.find(x => sid(x) === sel);
  if (!s) { $("view").innerHTML = `<div class="empty">Select a sample on the left.</div>`; $("judge").innerHTML = ""; return; }
  const m = col[s.model];
  $("view").innerHTML = `<div class="head"><span class="sw" style="background:${m.color}"></span><b>${esc(m.label.split(" (")[0])}</b><span>${s.arm}</span><span>sample #${s.i}</span>`
    + `<span class="${s.ok ? "ok" : "no"}">answer ${s.answer ?? "none"} ${s.ok ? "✓" : "✗"} (correct ${D.correct})</span><span>${s.cot_tokens} CoT tokens</span>${s.finish === "length" ? "<span class=no>cut at cap</span>" : ""}</div>`
    + sec("user message", D.user[s.arm], 1) + (s.arm.startsWith("nocot") ? sec("assistant prefill", D.prefill[s.arm]) : sec("CoT (analysis channel)", s.cot, 1, 1)) + sec("final answer", s.response, 1, 1);
  $("judge").innerHTML = s.arm.startsWith("nocot") ? `<div class="empty">No CoT, not judged.</div>` : s.reply == null ? `<div class="empty">Not judged.</div>`
    : `<div class="head"><b>Judge</b><span>REFERENCES <b>${yn(s.refs) || "unparsed"}</b></span></div>`
      + sec("judge prompt", D.judge_prompt.replace("{question}", () => D.question.split("\n\n\nYour response should be")[0]).replace("{hint}", () => D.hint).replace("{cot}", () => s.cot))
      + sec("judge reasoning", s.reasoning || "(no reasoning returned)", 1) + sec("judge output", s.reply, 1);
  wire($("view")); wire($("judge"));
  document.querySelectorAll("#view .sec-b.math, #judge .sec-b.math").forEach(math);
}
const COLS = [["model", "model"], ["arm", "arm"], ["i", "#"], ["answer", "ans"], ["tok", "CoT tok", 1], ["refs", "refs"], ["kw", "\"hint\""]];
const SORT = {model: s => D.models.findIndex(m => m.id === s.model), arm: s => D.arms.indexOf(s.arm), i: s => s.i, answer: s => s.answer ?? "~", tok: s => s.cot_tokens,
              refs: s => s.refs == null ? -1 : +s.refs, kw: s => +s.kw};
let sortBy = {k: "model", dir: 1};
function render() {
  const v = Object.fromEntries(Object.entries(FILTERS).map(([k, id]) => [k, $(id).value]));
  shown = D.samples.filter(s => (!v.model || s.model === v.model) && (!v.arm || s.arm === v.arm) && (v.ok === "" || String(+s.ok) === v.ok) && (!v.letter || (s.answer ?? "none") === v.letter)
    && (v.refs === "" || (s.refs != null && String(+s.refs) === v.refs)) && (v.kw === "" || (!s.arm.startsWith("nocot") && String(+s.kw) === v.kw)));
  const k = SORT[sortBy.k], cmp = (x, y) => x < y ? -1 : x > y ? 1 : 0;
  shown.sort((a, b) => sortBy.dir * cmp(k(a), k(b)) || cmp(SORT.model(a), SORT.model(b)) || cmp(SORT.arm(a), SORT.arm(b)) || cmp(SORT.i(a), SORT.i(b)));
  $("nshown").textContent = `${shown.length} shown`;
  $("list").innerHTML = `<tr>${COLS.map(([k, t, r]) => `<th data-k="${k}"${r ? ' style="text-align:right"' : ""} class="${sortBy.k === k ? (sortBy.dir > 0 ? "up" : "down") : ""}">${t}</th>`).join("")}</tr>`
    + shown.map(s => `<tr class="s" data-id="${sid(s)}"><td><span class="sw" style="background:${col[s.model].color}"></span>${esc(col[s.model].label.split(" (")[0])}</td><td>${s.arm}</td><td>${s.i}</td>`
      + `<td class="${s.ok ? "ok" : "no"}">${s.answer ?? "none"} ${s.ok ? "✓" : "✗"}</td><td class="num">${s.cot_tokens}</td><td>${yn(s.refs)}</td><td>${s.arm.startsWith("nocot") ? "" : s.kw ? "yes" : ""}</td></tr>`).join("");
  $("list").querySelectorAll("tr.s").forEach(tr => tr.addEventListener("click", () => { sel = tr.dataset.id; show(); }));
  $("list").querySelectorAll("th").forEach(th => th.addEventListener("click", () => { sortBy = {k: th.dataset.k, dir: sortBy.k === th.dataset.k ? -sortBy.dir : 1}; render(); }));
  if (!shown.some(x => sid(x) === sel)) sel = shown.length ? sid(shown[0]) : null;
  show();
}
document.addEventListener("keydown", e => {
  if (!["ArrowDown", "ArrowUp"].includes(e.key) || e.target.tagName === "SELECT" || !shown.length) return;
  const k = shown.findIndex(x => sid(x) === sel), j = Math.max(0, Math.min(shown.length - 1, k + (e.key === "ArrowDown" ? 1 : -1)));
  e.preventDefault(); sel = sid(shown[j]); show();
  document.querySelector(`#list tr.s[data-id="${CSS.escape(sel)}"]`).scrollIntoView({block: "nearest"});
});
let widths = [500, 600];
try { const w = JSON.parse(localStorage.getItem("hintuse-widths")); if (Array.isArray(w) && w.length === 2) widths = w; } catch (e) {}
const applyW = () => $("pane").style.gridTemplateColumns = `${widths[0]}px 6px ${widths[1]}px 6px minmax(180px, 1fr)`;
document.querySelectorAll(".divider").forEach(d => d.addEventListener("pointerdown", e => {
  e.preventDefault(); d.setPointerCapture(e.pointerId); d.classList.add("drag");
  const k = +d.dataset.k, x0 = e.clientX, w0 = widths[k];
  const move = ev => { widths[k] = Math.max(180, w0 + ev.clientX - x0); applyW(); };
  const up = () => { d.classList.remove("drag"); d.removeEventListener("pointermove", move); d.removeEventListener("pointerup", up); try { localStorage.setItem("hintuse-widths", JSON.stringify(widths)); } catch (e) {} };
  d.addEventListener("pointermove", move); d.addEventListener("pointerup", up);
}));
applyW();
for (const id of Object.values(FILTERS)) $(id).addEventListener("change", render);
$("f-math").addEventListener("change", show);
function load(k) {  // switch the whole page to run k
  D = ALL[k]; $("f-exp").value = k;
  try { localStorage.setItem("hintuse-run", D.name); } catch (e) {}
  COT_ARMS = D.arms.filter(a => !a.startsWith("nocot")); col = Object.fromEntries(D.models.map(m => [m.id, m]));
  $("qid").textContent = typeof D.item === "number" ? `GPQA item ${D.item}, hint "${D.hint_name}"` : `puzzle ${D.item}`; $("settings").textContent = D.settings; $("judgeline").textContent = D.judge_line;
  $("question").textContent = D.question; $("hint").textContent = D.hint; $("correct").textContent = D.correct;
  document.querySelectorAll(".lg").forEach(e => e.innerHTML = D.models.map(m => `<span><span class="sw" style="background:${m.color}"></span>${esc(m.label.split(" (")[0])}</span>`).join("")
    + (e.dataset.hatch ? `<span><span class="sw" style="background:#c3c2b7"></span>judge: REFERENCES YES</span><span>${HATCH_SW}CoT contains the word "hint"</span>` : ""));
  $("f-model").innerHTML = `<option value="">any</option>` + D.models.map(m => `<option value="${m.id}">${esc(m.label.split(" (")[0])}</option>`).join("");
  $("f-arm").innerHTML = `<option value="">any</option>` + D.arms.map(a => `<option>${a}</option>`).join("");
  $("f-letter").innerHTML = `<option value="">any</option>` + D.keys.map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join("");
  for (const id of Object.values(FILTERS)) $(id).value = "";
  sel = null; drawAcc(); drawJudge(); drawTok(); drawCounts(); render();
}
$("f-exp").innerHTML = ALL.map((x, k) => `<option value="${k}">${esc(x.name.replace(/^hintuse_/, ""))}</option>`).join("");
$("f-exp").addEventListener("change", () => load(+$("f-exp").value));
let start = ALL.length - 1;
try { const j = ALL.findIndex(x => x.name === localStorage.getItem("hintuse-run")); if (j >= 0) start = j; } catch (e) {}
load(start);
</script></body></html>
"""

if __name__ == "__main__":
    main()
