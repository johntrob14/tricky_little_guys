"""Judge quotes highlighted in the CoT, shared by the sandbag viewers (sandbag_overview_html.py, sandbag_behavior_html.py, sandbag_resample_html.py): CSS
and JS strings pasted into their pages. Structured judges (sandbag_behavior_judge.py) answer with <quote> tags, kept as `quotes`; free-text judges (the Li
et al. 2025 CoT-only monitor and the verbalization judge of sandbag_judge.py) are read for double-quoted strings of 12+ characters in their reasoning and
output (quotedIn). A quote is matched as sandbag_behavior_judge.found: both sides reduced to lowercase letters and digits, each ellipsis-separated
fragment on its own (4+ such characters), and every occurrence is marked. The page defines QKIND (kind -> {label, color}); hlHtml(text, quotes) is the
escaped CoT with a translucent <mark> per matched fragment (where kinds overlap, the first kind is the background and the others underline it; hovering
shows the judges in the page's tooltip, via showTip and tip) and hlLegend(text, quotes) is a legend line with the fragments found per kind. JS goes after the page's `esc` and `$`; the page has a checkbox `f-hl` ("highlight judge quotes", kept in localStorage) and a `show()` that redraws the row.
"""

CSS = """mark.hl { color:inherit; border-radius:2px; padding:0 1px; cursor:default; }
.hlleg { display:flex; gap:14px; flex-wrap:wrap; align-items:center; font-size:12px; color:var(--ink2); margin:6px 0 8px; }
.hlleg .sw { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-2px; margin-right:5px; }"""

JS = r"""// judge quotes highlighted in the CoT (judge_quotes.py). quotes: [{text, kind}]; QKIND: kind -> {label, color}, in display order
const HLBOX = $("f-hl"); try { HLBOX.checked = localStorage.getItem("judge-quotes-hl") !== "0"; } catch (e) {}  // the page's "highlight judge quotes" checkbox, kept in localStorage
HLBOX.addEventListener("change", () => { try { localStorage.setItem("judge-quotes-hl", HLBOX.checked ? "1" : "0"); } catch (e) {} show(); });
const quotedIn = t => [...(t || "").matchAll(/["“]([^"”\n]{12,}?)["”]/g)].map(m => m[1]);  // free-text judges: double-quoted strings
const loose = s => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
const frags = q => q.split(/\.\.\.|…/).map(loose).filter(f => f.length >= 4);
function hlSpans(text, quotes) {  // [start, end, kind] in text, every occurrence of every fragment
  const pos = [], low = [];
  for (let i = 0; i < text.length; i++) { const c = text[i].normalize("NFKC").toLowerCase(); if (/^[\p{L}\p{N}]+$/u.test(c)) { low.push(c); pos.push(...Array(c.length).fill(i)); } }
  const L = low.join(""), out = [];
  for (const {text: q, kind} of quotes) for (const f of frags(q)) for (let p = L.indexOf(f); p >= 0; p = L.indexOf(f, p + 1)) out.push([pos[p], pos[p + f.length - 1] + 1, kind]);
  return out;
}
const hlStyle = ks => `background:${QKIND[ks[0]].color}40` + (ks.length > 1 ? `;box-shadow:${ks.slice(1).map((k, i) => `inset 0 ${-2 * (i + 1)}px 0 ${QKIND[k].color}`).join(",")}` : "");
function hlHtml(text, quotes) {
  if (!HLBOX.checked) return esc(text || "(empty)");
  const ev = hlSpans(text, quotes).flatMap(([a, b, k]) => [[a, 1, k], [b, -1, k]]).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  const open = {}; let html = "", at = 0;
  const flush = to => { if (to <= at) return; const ks = Object.keys(QKIND).filter(k => open[k] > 0), t = esc(text.slice(at, to));
    html += ks.length ? `<mark class="hl" style="${hlStyle(ks)}" data-j="${esc(ks.map(k => QKIND[k].label).join(" and the "))}">${t}</mark>` : t; at = to; };
  for (const [i, d, k] of ev) { flush(i); open[k] = (open[k] || 0) + d; }
  flush(text.length);
  return html;
}
document.addEventListener("mousemove", e => { const m = e.target.closest("mark.hl"); if (m) showTip(e, "quoted by the " + m.dataset.j); else if (!e.target.closest(".bar")) tip.style.display = "none"; });  // the page's own tooltip, no delay
function hlLegend(text, quotes) {
  const L = loose(text), kinds = Object.keys(QKIND).filter(k => quotes.some(q => q.kind === k));
  if (!HLBOX.checked || !kinds.length) return "";
  return `<div class="hlleg"><span>judge quotes in the CoT:</span>` + kinds.map(k => { const fs = quotes.filter(q => q.kind === k).flatMap(q => frags(q.text));
    return `<span><span class="sw" style="background:${QKIND[k].color}"></span>${esc(QKIND[k].label)} ${fs.filter(f => L.includes(f)).length}/${fs.length} found</span>`; }).join("")
    + `<span class="note">quote tags, or double-quoted strings in a free-text judge's reasoning and output; matched on letters and digits only, each ellipsis-separated piece on its own</span></div>`;
}
"""
