# Interactive HTML plots: what Ethan means

When Ethan asks for an "interactive plot", an "interactive HTML", a "viewer" or a visualization page, build it in this style. The reference
implementation is `cot-control/rl/sandbag_resample_html.py` (its output: `gale:~/wgmn/mats12_figs/sandbag_gpqa.must_q0006_resample.html`); Ethan approved
its styling and functionality on 2026-10-28. Take the layout, styling and interaction from it, not its subject: the panels, charts, filters and judges
below are whatever the data at hand has.

## Build and delivery

- One self-contained `.html` file written by a committed Python script (data embedded as JSON in the page, vanilla JS, charts as hand-built inline SVG; no
  chart library). The only external resource is KaTeX from cdnjs when the text contains LaTeX.
- Write it to `mats12_sprint/figures/`, copy it to `gale:~/wgmn/mats12_figs/`, and open it in Firefox on the laptop, again after every rebuild (the
  "Figures on Gale" rule in the project `CLAUDE.md`).
- Before delivering, render it in headless Firefox on tav (`firefox --headless --no-remote --profile <scratch dir> --window-size=W,H --screenshot <abs path>.png
  file://<copy>`) from a scratch copy with an injected `window.onerror` banner and a script that clicks through the main interactions; look at the
  screenshot. No Playwright on tav.

## Styling

- Dark theme from CSS tokens on `:root` (`--bg #1a1a19`, `--panel #232322`, `--line #3a3a38`, `--ink #f0efec`, `--ink2 #c3c2b7`, `--muted #8d8c84`, green
  `#5fb87a` / red `#e0705f` for right/wrong), `color-scheme: dark`, dark thin scrollbars on scrolling panes, system-ui text, monospace for model text.
- One fixed color per entity (model, run), the same in every chart, table swatch and legend; validate the palette against the dark surface (dataviz skill's
  `validate_palette.js`). Secondary splits use hatching, not new colors.
- Square-cornered, flush-stacked collapsible sections; full-width boxes for dropdowns and section headers, never bare clickable text.

## Page layout, top to bottom

1. Title, then a subtitle that states the settings needed to read the numbers: sampling (model, temperature, effort, cap, backend), and for every judge or
   monitor its mode (action only / action+cot / cot only), name and judge model.
2. A row with the shared context (e.g. the question or task) beside a compact summary table (e.g. answer counts per entity and condition; cells clickable).
3. One row of 2-3 bar charts (flex, wrapping on narrow windows): grouped bars per condition, rates with Wilson 95% intervals and count/total labels,
   scores as mean ± SE with n. Each chart box has a short title (the measure and its interval), then its own legend row (`cot-control/rl/viewer_legend.py`:
   `legend(id, items)` draws a swatch per item; the models in their colors, then the bar styles in gray: hatched, dotted, faint, dashed), then the svg.
   Titles name the measure only; what solid vs hatched means, which models are drawn and any active filter go in the legend, and one note above the
   row says that bars have hover tooltips and click to filter. Each bar has a hover tooltip with its full description and numbers; clicking a bar (or
   a table cell) sets the explorer's filters to that slice and scrolls to it. Charts leave out groups that have no data for that measure rather than
   drawing empty slots.
4. The explorer: a full-viewport-height grid of three panes separated by draggable dividers (pane widths kept in `localStorage`, wrapped in try/catch):
   - Left sidebar: filters as labelled full-width `<select>` boxes in a grid (square corners, `border-radius:0`), checkboxes (e.g. show reference rows, render math,
     render markdown), a "clear filters" button that resets every select, an "N shown" count, then the item table
     with a sticky header whose column titles sort the list (click again to reverse, arrow on the active column). Click selects a row; Up/Down arrow keys
     move the selection.
   - Middle: the selected item. A header line with its key facts (entity, condition, sample, outcome, length), then collapsible sections: prompts collapsed
     by default, model output (CoT, answer) open; the open/closed choice carries over between items.
     Text sections render markdown (`cot-control/rl/viewer_md.py`: marked from cdnjs, math regions shielded, toggle "render markdown") and LaTeX (KaTeX auto-render,
     toggle "render math"); the toggles are kept in `localStorage`.
   - Right: the judges. For each judge: its score as a colored badge, the exact prompt it was sent (filled in, collapsed by default), its reasoning, its
     output.
5. Short notes at the bottom for caveats about how a judge was run.

## Functionality details

- Judge or monitor scores are colored on a diverging scale: red = suspicious, gray at the flag cutoff, green = not suspicious, as cell backgrounds in the
  table and on the badge, with the number always shown and a small scale legend.
- LaTeX in model text (question, CoT, answers, judge text) is rendered with KaTeX auto-render (`$`, `$$`, `\(`, `\[`), with a "render math" checkbox to
  show the raw text; prompts stay raw. Display equations scroll horizontally. If KaTeX fails to load, the page shows a red notice and raw text.
- Reference rows that are not part of the statistics (e.g. an earlier run's sample of the same item) are listed in the explorer, clearly labelled, and left
  out of the charts.
- Model text is HTML-escaped; text substitutions use function replacements (`replaceAll(x, () => y)`) so `$` in LaTeX survives.
