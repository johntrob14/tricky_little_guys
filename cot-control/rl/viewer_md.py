"""Markdown in the text sections of the interactive viewers (notes/interactive_html_guidelines.md), next to the KaTeX math: CSS and JS strings pasted
into a page, as viewer_legend.py. marked (cdnjs, SCRIPT) renders the section's escaped HTML, which may hold the page's <mark> highlights; math regions
($...$, $$...$$, \\(...\\), \\[...\\]) are swapped for tokens first so markdown does not touch them (underscores, backslashes), and KaTeX runs on the result as before.
Single newlines stay line breaks (breaks: true). The page has a checkbox `f-md` ("render markdown", kept in localStorage) and calls show() on change;
sec() wraps its body HTML in mdHtml(h) and gives the block class "md" when MD() is true, which switches white-space from pre-wrap to normal. Without marked
(offline) the raw text shows."""

SCRIPT = '<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/18.1.2/lib/marked.umd.min.js"></script>'

CSS = r""".sec-b.md { white-space:normal; } .sec-b.md p, .sec-b.md ul, .sec-b.md ol, .sec-b.md table, .sec-b.md blockquote { margin:0 0 0.7em; } .sec-b.md > :last-child { margin-bottom:0; }
.sec-b.md h1, .sec-b.md h2, .sec-b.md h3, .sec-b.md h4 { font-size:1.05em; margin:0.8em 0 0.4em; color:var(--ink); } .sec-b.md ul, .sec-b.md ol { padding-left:1.6em; } .sec-b.md li { margin:0.15em 0; }
.sec-b.md pre { white-space:pre-wrap; background:var(--panel); border:1px solid var(--line); padding:6px 8px; margin:0 0 0.7em; } .sec-b.md code { background:var(--panel); padding:0 3px; } .sec-b.md pre code { background:none; padding:0; }
.sec-b.md table { border-collapse:collapse; } .sec-b.md th, .sec-b.md td { border:1px solid var(--line); padding:2px 6px; white-space:normal; } .sec-b.md blockquote { border-left:3px solid var(--line); padding-left:8px; color:var(--ink2); } .sec-b.md hr { border:0; border-top:1px solid var(--line); }"""

JS = r"""// markdown in the text sections (viewer_md.py)
const MDBOX = $("f-md"); try { MDBOX.checked = localStorage.getItem("viewer-md") !== "0"; } catch (e) {}
MDBOX.addEventListener("change", () => { try { localStorage.setItem("viewer-md", MDBOX.checked ? "1" : "0"); } catch (e) {} show(); });
const MD = () => !!window.marked && MDBOX.checked;
const MD_MATH = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$/g;
function mdHtml(h) {  // h: escaped text, possibly with <mark> tags; math regions are kept out of marked's hands
  if (!MD()) return h;
  const keep = [], shielded = h.replace(MD_MATH, m => `MDMATH${keep.push(m) - 1}X`);
  return marked.parse(shielded, {breaks: true, gfm: true}).replace(/MDMATH(\d+)X/g, (_, i) => keep[i]);
}
"""
