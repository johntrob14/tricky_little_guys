"""Per-chart legends for the interactive viewers (notes/interactive_html_guidelines.md). Each chart box holds a short title (the measure and its interval),
a legend row (`<div class="lg" id="lg-<chart>">`) and the svg; the page's JS fills the row with `legend(id, items)`, one swatch and label per item:
[color, style, label]. Styles: solid; hatch (diagonal lines over the color); dots; faint (40% opacity); outline (stroke only); dash (a dashed line, for
reference lines); text (label only). Model entries carry the model's color; bar-style entries carry GRAY. Pages take CSS into their <style> and JS after
their `esc` helper (`$` and `esc` must exist)."""

CSS = r""".lg { display:flex; flex-wrap:wrap; gap:2px 14px; font-size:12px; color:var(--ink2); margin:0 2px 6px; align-items:center; min-height:16px; }
.lg span { display:inline-flex; align-items:center; gap:5px; white-space:nowrap; } .lg svg { flex:none; }"""

JS = r"""const GRAY = "#c3c2b7";
const SW = w => `style="width:${w}px;height:12px;display:inline-block;flex:none"`;  // inline, so a page's `.chart svg { width:100% }` rule does not stretch the swatch
const swatch = (color, style) => style === "text" ? "" : style === "dash" ? `<svg width="16" height="12" ${SW(16)}><line x1="0" x2="16" y1="6" y2="6" stroke="${color}" stroke-width="2" stroke-dasharray="5 3"/></svg>`
  : `<svg width="12" height="12" ${SW(12)}><rect width="12" height="12" rx="2" ${style === "outline" ? `fill="none" stroke="${color}" stroke-width="2"` : style === "dim" ? `fill="${color}" stroke="#55554f"` : `fill="${color}"`}${style === "faint" ? ' fill-opacity=".4"' : ""}/>`
    + (style === "hatch" ? `<path d="M-2,14 L14,-2 M4,14 L14,4 M-2,8 L8,-2" stroke="#1a1a19" stroke-width="2"/>` : style === "dots" ? `<circle cx="3" cy="3" r="1.3" fill="#1a1a19"/><circle cx="8.5" cy="3" r="1.3" fill="#1a1a19"/><circle cx="3" cy="8.5" r="1.3" fill="#1a1a19"/><circle cx="8.5" cy="8.5" r="1.3" fill="#1a1a19"/>` : "") + `</svg>`;
function legend(id, items) { $(id).innerHTML = items.filter(Boolean).map(([color, style, label]) => `<span>${swatch(color, style)}${esc(label)}</span>`).join(""); }
"""
