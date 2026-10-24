"""Build an offline viewer for the pilot's complete edited episodes; no model calls."""

import argparse
import copy
import difflib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDITIONS = [("episodes_hybrid", "Mixed editor settings"), ("episodes_no_thinking", "Editor thinking off"), ("episodes", "Editor thinking medium")]


def read(path: Path) -> object:
    return json.loads(path.read_text())


def escape(value: str) -> str:
    return html.escape(value)


def reasoning(message: dict) -> str:
    return "\n".join(b["reasoning"] for b in message["content"] if b["type"] == "reasoning") if isinstance(message["content"], list) else ""


def content(message: dict) -> str:
    if isinstance(message["content"], str):
        return message["content"]
    assert all(b["type"] in ("text", "reasoning") for b in message["content"])
    return "\n".join(b["text"] for b in message["content"] if b["type"] == "text")


def non_reasoning(messages: list[dict]) -> list[dict]:
    result = copy.deepcopy(messages)
    for message in result:
        if isinstance(message["content"], list):
            message["content"] = [b for b in message["content"] if b["type"] != "reasoning"]
    return result


def diff(before: str, after: str) -> tuple[str, str, str]:
    old, new = re.findall(r"\s+|\w+|[^\w\s]", before), re.findall(r"\s+|\w+|[^\w\s]", after)
    left, right, combined = [], [], []
    for kind, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        removed, added = escape("".join(old[a:b])), escape("".join(new[c:d]))
        left.append(removed if kind == "equal" else f'<del>{removed}</del>' if removed else "")
        right.append(added if kind == "equal" else f'<ins>{added}</ins>' if added else "")
        combined.append(removed if kind == "equal" else (f'<del>{removed}</del>' if removed else "") + (f'<ins>{added}</ins>' if added else ""))
    return "".join(left), "".join(right), "".join(combined)


def disclosure(title: str, text: str) -> str:
    return f'<details class="context"><summary>{escape(title)}</summary><pre>{escape(text)}</pre></details>'


def episode(case: dict, condition: str, label: str) -> tuple[str, str]:
    uuid = case["uuid"]
    original = read(ROOT / "working" / f"{uuid}.original.json")
    edited = read(ROOT / "working" / condition / f"{uuid}.edited.json")
    summary = read(ROOT / "working" / condition / uuid / "summary.json")
    assert len(original) == len(edited) and non_reasoning(original) == non_reasoning(edited)
    rows = {r["turn"]: r for r in summary["turns"]}
    indexes = [i for i, message in enumerate(original) if message["role"] == "assistant"]
    changed = sum(reasoning(original[i]) != reasoning(edited[i]) for i in indexes)
    total_reasoning = sum(bool(reasoning(original[i])) for i in indexes)
    cards = []
    for turn, index in enumerate(indexes):
        message = original[index]
        before, after = reasoning(message), reasoning(edited[index])
        for block in edited[index]["content"] if isinstance(edited[index]["content"], list) else []:
            if block["type"] == "reasoning" and block.get("signature"):
                assert json.loads(block["signature"].removeprefix("reasoning-details://"))[0]["text"] == block["reasoning"]
        different = before != after
        left, right, combined = diff(before, after)
        state = "Changed" if different else "Unchanged" if before else "No reasoning"
        tools = message.get("tool_calls", [])
        names = ", ".join(call["function"] for call in tools) or "Response"
        visible = disclosure("Visible assistant text", content(message)) if content(message) else ""
        actions = "".join(disclosure(f"Action · {call['function']}", json.dumps(call["arguments"], indent=2, ensure_ascii=False)) for call in tools)
        end = indexes[turn + 1] if turn + 1 < len(indexes) else len(original)
        results = "".join(disclosure(f"Tool result · {m['function']}", content(m)) for m in original[index + 1:end] if m["role"] == "tool")
        notes = disclosure("Editor’s notes (self-assessment)", rows[turn]["review_notes"]) if rows[turn].get("review_notes") else ""
        cards.append(f'''<details class="turn" data-changed="{int(different)}" id="{uuid}-turn-{turn}">
<summary><strong>Turn {turn}</strong><span class="badge {'changed' if different else ''}">{state}</span><span class="tools">{escape(names)}</span><span class="length">{len(before):,} → {len(after):,} chars</span></summary>
<div class="turn-body"><div class="reasoning"><div class="original"><h3>Original reasoning</h3><pre>{left or '<em>No reasoning in this turn.</em>'}</pre></div><div class="edited"><h3>Edited reasoning</h3><pre>{right or '<em>No reasoning in this turn.</em>'}</pre></div><div class="inline-diff"><h3>Changes in context</h3><pre>{combined or '<em>No reasoning in this turn.</em>'}</pre></div></div>
<div class="support">{notes}{visible}{actions}{results}</div></div></details>''')
    task = case["task"].replace("_", " ")
    prompts = "".join(disclosure(f"{m['role'].title()} prompt", content(m)) for m in original[:indexes[0]])
    section = f'''<section class="episode" id="{uuid}" hidden><div class="episode-heading"><div><h2>{escape(task)}</h2><p>{escape(label)} · {escape(case['tier'])} · complete · {len(indexes)}/{len(indexes)} assistant turns</p></div><span class="stat">{changed}/{total_reasoning} reasoning blocks changed</span></div>
<details class="context prompts"><summary>Task, system prompt &amp; source</summary><p class="source">{escape(case['log'])}<br>Sample: {uuid}<br>Edited file: working/{condition}/{uuid}.edited.json</p>{prompts}</details>
<div class="cards">{''.join(cards)}</div></section>'''
    return f'<option value="{uuid}">{escape(task)} · {escape(label)}</option>', section


PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>DeepSeek pilot · edited traces</title><style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#151b24;--fg:#e0e6ee;--muted:#9caabd;--line:#2c3747;--accent:#9ac7ff;font:15px/1.6 system-ui,sans-serif;font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg)}main,header>div{max-width:1500px;margin:auto;padding:20px 28px}header{position:sticky;top:0;z-index:5;background:#0d1117f5;box-shadow:0 1px var(--line)}h1{font-size:23px;margin:0 0 4px;letter-spacing:-.025em}h2{font-size:21px;margin:0;text-wrap:balance}h3{font-size:12px;color:var(--muted);margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em}p{margin:4px 0 12px;color:var(--muted);text-wrap:pretty}.intro{max-width:1000px;font-size:13px}.controls,.filters{display:flex;flex-wrap:wrap;align-items:center;gap:12px}.controls{margin-top:14px}.filters{margin-top:12px;font-size:13px}.control{display:grid;gap:4px;font-size:12px;color:var(--muted)}.control:first-child{flex:1;min-width:240px}select,input[type=search],button{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:7px;min-height:40px;padding:8px 12px}select{width:100%}input[type=search]{flex:1;min-width:170px}button,summary{cursor:pointer}button{transition:background-color .15s,scale .15s}button:hover{background:#253247}button:active{scale:.96}:focus-visible{outline:2px solid var(--accent);outline-offset:3px}input[type=checkbox]{accent-color:var(--accent)}label.check{display:flex;align-items:center;gap:7px;min-height:40px}#count{color:var(--muted);margin-left:auto}.legend{display:flex;gap:14px;font-size:12px;margin-top:10px;color:var(--muted)}del{background:#6b293d6b;color:#ffb2be;text-decoration:line-through;text-decoration-color:#e18b99}ins{background:#1c62486b;color:#9aebc2;text-decoration:none}.legend span{padding:1px 6px;border-radius:3px}.deleted-key{background:#6b293d6b;color:#ffb2be}.added-key{background:#1c62486b;color:#9aebc2}.episode-heading{display:flex;justify-content:space-between;gap:18px;align-items:center;margin:8px 0 16px}.episode-heading p{font-size:13px;margin:4px 0 0}.stat{color:var(--accent);font-size:13px;white-space:nowrap}.context{border-top:1px solid var(--line)}.context summary{padding:10px 0;font-size:13px;color:var(--muted)}.prompts{margin-bottom:22px;border-bottom:1px solid var(--line)}.source{font:12px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}.turn{background:var(--panel);border-radius:10px;margin:12px 0;box-shadow:0 0 0 1px var(--line),0 3px 10px #0002;overflow:hidden;scroll-margin-top:260px}.turn[open]{box-shadow:0 0 0 1px #607eaa,0 3px 14px #0003}.turn>summary{display:flex;align-items:center;gap:12px;padding:14px 18px;list-style:none}.turn>summary::-webkit-details-marker{display:none}.turn>summary::before{content:'▸';color:var(--muted)}.turn[open]>summary::before{content:'▾'}.turn>summary strong{font-size:14px;white-space:nowrap}.badge{font-size:11px;background:#263142;padding:2px 8px;border-radius:20px;white-space:nowrap}.badge.changed{color:#a7d3ff;background:#203b5b}.tools{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.length{margin-left:auto;color:var(--muted);font-size:12px;white-space:nowrap}.turn-body{padding:0 18px 16px}.reasoning{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.reasoning>div{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;tab-size:4;font:13px/1.75 ui-monospace,SFMono-Regular,Consolas,monospace;background:#0d121b;padding:16px;border-radius:6px;margin:0 0 14px}em{color:var(--muted)}.inline-diff{display:none}.support{margin-top:12px}.support pre{max-height:600px;overflow:auto}.context[open]>summary{color:var(--accent)}body[data-view=diff] .original,body[data-view=diff] .edited{display:none}body[data-view=diff] .inline-diff{display:block}body[data-view=edited] .original,body[data-view=original] .edited{display:none}body:not([data-view=split]) .reasoning{grid-template-columns:minmax(0,1fr)}body[data-view=edited] ins,body[data-view=original] del{background:none;color:inherit;text-decoration:none}[hidden]{display:none!important}#empty{text-align:center;padding:60px 20px}footer{color:var(--muted);font-size:12px;margin:24px 0 36px}
@media(max-width:850px){main,header>div{padding:16px}.reasoning{grid-template-columns:minmax(0,1fr)}.episode-heading{display:block}.stat{display:inline-block;margin-top:8px}.tools{display:none}.turn>summary{gap:8px;padding:12px}.turn-body{padding:0 12px 12px}.length{font-size:10px}.intro{display:none}header{position:static}}
@media(prefers-reduced-motion:reduce){button{transition:none}}
</style></head><body data-view="split">
<header><div><h1>DeepSeek pilot · edited traces</h1><p class="intro">__COVERAGE__ source episodes have a complete edited version. Each trace uses the condition shown. Read the rewritten reasoning alongside its original, then open actions and tool results for context.</p>
<div class="controls"><label class="control">Trace<select id="trace">__OPTIONS__</select></label><label class="control">Reading view<select id="view"><option value="split">Original / edited</option><option value="diff">Inline changes</option><option value="edited">Edited only</option><option value="original">Original only</option></select></label></div>
<div class="filters"><label class="check"><input type="checkbox" id="changed" checked> Changed reasoning only</label><input type="search" id="search" aria-label="Search trace" placeholder="Find in reasoning, actions or results"><button id="expand">Expand shown</button><button id="collapse">Collapse shown</button><span id="count" role="status" aria-live="polite"></span></div>
<div class="legend"><span class="deleted-key">− Removed</span><span class="added-key">+ Added</span><span>Turns use the logs’ zero-based numbering.</span></div></div></header>
<main>__SECTIONS__<p id="empty" hidden>No turns match these filters.</p><footer>All non-reasoning content is preserved. Editor notes are the editor’s own assessment. The rate-limiter episode has no complete edited version and is omitted. This file works offline.</footer></main>
<script>
const trace = document.querySelector('#trace'), view = document.querySelector('#view'), changed = document.querySelector('#changed'), search = document.querySelector('#search');
const episodes = [...document.querySelectorAll('.episode')], cards = [...document.querySelectorAll('.turn')];
const searchable = new Map(cards.map(card => [card, card.textContent.toLowerCase()]));
function selectedCards() { return [...document.getElementById(trace.value).querySelectorAll('.turn')]; }
function apply() {
    episodes.forEach(episode => episode.hidden = episode.id !== trace.value);
    const query = search.value.trim().toLowerCase(), current = selectedCards();
    current.forEach(card => card.hidden = (changed.checked && card.dataset.changed !== '1') || !searchable.get(card).includes(query));
    const shown = current.filter(card => !card.hidden).length;
    document.querySelector('#count').textContent = `${shown}/${current.length} turns shown`;
    document.querySelector('#empty').hidden = shown !== 0;
}
function chooseTrace() {
    apply();
    selectedCards().forEach(card => card.open = false);
    const first = selectedCards().find(card => !card.hidden);
    if (first) first.open = true;
    window.scrollTo(0, 0);
}
trace.addEventListener('change', chooseTrace);
view.addEventListener('change', () => document.body.dataset.view = view.value);
changed.addEventListener('change', apply);
search.addEventListener('input', apply);
document.querySelector('#expand').addEventListener('click', () => selectedCards().filter(card => !card.hidden).forEach(card => card.open = true));
document.querySelector('#collapse').addEventListener('click', () => selectedCards().filter(card => !card.hidden).forEach(card => card.open = false));
chooseTrace();
</script></body></html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parents[3] / "figures/deepseek_data_pilot_traces.html")
    args = parser.parse_args()
    cases = read(ROOT / "selection.json")["selected"]
    options, sections = [], []
    for case in cases:
        complete = [(condition, label) for condition, label in CONDITIONS if (ROOT / "working" / condition / f"{case['uuid']}.edited.json").exists()]
        if complete:
            option, section = episode(case, *complete[0])
            options.append(option)
            sections.append(section)
    assert sections, "No complete edited episodes"
    page = PAGE.replace("__COVERAGE__", f"{len(sections)}/{len(cases)}").replace("__OPTIONS__", "".join(options)).replace("__SECTIONS__", "".join(sections))
    args.out.write_text(page)
    print(f"Complete episodes: {len(sections)}/{len(cases)}; wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
