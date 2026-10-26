"""Static HTML viewer for the WMDP sandbagging eval (sandbag.py arms): every question, every arm, every model's rollout.

Sidebar: one line per question with an outcome strip per model, one cell per arm showing the answer letter (green correct, red wrong, a white ring when the
question flipped from a correct honest answer, amber "…" cut off at the token cap, gray "–" finished without a letter; a purple dot when the CoT-only monitor
flagged the CoT), filters on model, arm and outcome, j/k to step through the filtered list, and a per-model accuracy table (finished rows). Main pane: the
question's correct and named-wrong letters and honest-arm prompt, a models x arms grid of rollout cards (answer, CoT tokens, cap, monitor score,
verbalization judge, hand-read verdict), and the clicked cards side by side with their CoT, final channel, inserted instruction, prompt, monitor reply,
verbalization reply and hand-read fields. The card selection is kept by model and arm across questions and lives in the URL hash with the question.
Inputs: data/wmdp_sandbagging.jsonl, <dir>/<model>/{rows,monitor,verbalized}.jsonl, <dir>/handval/*verdicts_shard*.jsonl (mt1b_ files are mt1b-s150).

    python sandbag_traces_html.py --dir ../results/sandbag --out ../results/sandbag/sandbag_traces.html
"""

import argparse
import glob
import json
import zlib
from pathlib import Path

from sandbag import ARMS, AWARE, EVASION, instruction_text
from sandbag_judge import CUTOFF

KIND = {"yes": "stated", "yes_stated": "stated", "yes_silent": "inconsistent", "unclear": "unclear", "no": "genuine"}
LABELS = {"base": "gpt-oss-120b base", "grpo3-s100": "grpo3 s100", "mt1b-s150": "manytypes1b s150", "base-nohint": "gpt-oss-120b base, no hint", "mt1b-s150-nohint": "manytypes1b s150, no hint", "base-sysprompt": "gpt-oss-120b base, instruction in system prompt", "mt1b-s150-sysprompt": "manytypes1b s150, instruction in system prompt", "base-sysprompt-nohint": "gpt-oss-120b base, system prompt, no hint", "mt1b-s150-sysprompt-nohint": "manytypes1b s150, system prompt, no hint"}
SHORT = {"base": "base", "grpo3-s100": "grpo3", "mt1b-s150": "mt1b", "base-nohint": "base nohint", "mt1b-s150-nohint": "mt1b nohint", "base-sysprompt": "base sys", "mt1b-s150-sysprompt": "mt1b sys", "base-sysprompt-nohint": "base sys nohint", "mt1b-s150-sysprompt-nohint": "mt1b sys nohint"}


def looping(cot: str) -> bool:
    t = cot[-6000:].encode()
    return len(zlib.compress(t)) < 0.12 * len(t)


CSS = """
body{background:#1a1a19;color:#e8e6e3;font:13px system-ui,sans-serif;margin:0;display:flex;height:100vh}
nav{overflow:auto;border-right:1px solid #3a3a38;padding:10px}
main{flex:1;overflow:auto;padding:14px 22px}
h1{font-size:15px;margin:0 0 6px} h2{font-size:16px;margin:0 0 8px} h3{font-size:12px;color:#c3c2b7;text-transform:uppercase;letter-spacing:.05em;margin:14px 0 4px}
.muted{color:#a3a29b;font-size:12px} select,input{background:#242422;color:#e8e6e3;border:1px solid #4a4a47;border-radius:3px;padding:2px 4px;font-size:12px}
.filters{display:grid;grid-template-columns:auto 1fr;gap:4px 8px;align-items:center;margin:8px 0}
table.sum{border-collapse:collapse;font-size:11px;margin:4px 0} table.sum td,table.sum th{padding:1px 5px;text-align:right} table.sum th{color:#a3a29b;font-weight:normal}
.q{display:flex;align-items:center;gap:10px;padding:3px 6px;cursor:pointer;border-radius:3px} .q:hover{background:#2a2a28} .q.cur{background:#2f3a4a}
.hdr{cursor:default;position:sticky;top:0;background:#1a1a19;z-index:1} .hdr:hover{background:#1a1a19} .hs{flex-direction:column;align-items:center} .hm{font-size:10px;color:#a3a29b} .c.ch{color:#a3a29b;background:none;font-weight:normal}
.qid{width:44px;color:#c3c2b7;font-variant-numeric:tabular-nums}
.strip{display:flex;gap:2px} .c{position:relative;width:17px;height:17px;line-height:17px;text-align:center;font-size:11px;font-weight:600;border-radius:3px;color:#1a1a19}
.o-ok{background:#1baf7a} .o-bad{background:#e5484d} .o-cut{background:#c98500} .o-none{background:#6a6965}
.flipb{background:#f0efec} .c.flip{box-shadow:0 0 0 2px #f0efec inset} .c.flag::after{content:"";position:absolute;top:-3px;right:-3px;width:7px;height:7px;border-radius:50%;background:#a78bfa;border:1px solid #1a1a19}
.legend{display:grid;grid-template-columns:auto 1fr;gap:3px 8px;align-items:center;font-size:11px;color:#c3c2b7;margin:6px 0}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-weight:600;color:#1a1a19;margin-right:4px}
.prompt{background:#242422;border:1px solid #3a3a38;border-radius:4px;padding:8px;white-space:pre-wrap;max-height:30vh;overflow:auto}
table.grid{border-collapse:separate;border-spacing:6px} table.grid th{color:#c3c2b7;font-weight:normal;text-align:left;font-size:12px}
.card{background:#242422;border:1px solid #3a3a38;border-radius:5px;padding:6px 8px;width:230px;cursor:pointer;font-size:12px;line-height:1.45}
.card:hover{border-color:#6a6965} .card.sel{border-color:#3987e5;box-shadow:0 0 0 1px #3987e5}
.k-stated{color:#eda100} .k-inconsistent{color:#e87ba4} .k-unclear{color:#b8b7b2} .k-genuine{color:#8a8984}
.cols{display:flex;gap:12px;align-items:flex-start} .col{flex:1;min-width:0}
.col h4{margin:0 0 4px;font-size:13px} .box{background:#242422;border:1px solid #3a3a38;border-radius:4px;padding:8px;white-space:pre-wrap;word-wrap:break-word;max-height:72vh;overflow:auto;font-size:12px}
.lab{font-size:11px;color:#a3a29b;margin:8px 0 2px;text-transform:uppercase;letter-spacing:.04em}
details summary{cursor:pointer;color:#c3c2b7;font-size:12px;margin-top:6px} kbd{background:#2a2a28;border:1px solid #4a4a47;border-radius:3px;padding:0 4px;font-size:11px}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const M = D.models, A = D.arms, QS = D.questions, Q = Object.fromEntries(QS.map(q => [q.qid, q]));
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const R = (m, q, a) => D.rows[m][q][a];
const outcome = r => r.cut ? 'cut' : r.a == null ? 'none' : r.ok ? 'ok' : 'bad';
const flagged = r => typeof r.mon === 'number' && r.mon < D.cutoff;
const isFlip = (m, q, a) => a !== 'honest' && R(m, q, 'honest').ok && R(m, q, a).a != null && !R(m, q, a).ok;
const COND = {
  'any': () => true,
  'wrong answer (letter parsed)': (m, q, a) => R(m, q, a).a != null && !R(m, q, a).ok,
  'flip: honest correct, this arm wrong': isFlip,
  'reverse flip: honest wrong, this arm correct': (m, q, a) => a !== 'honest' && R(m, q, 'honest').a != null && !R(m, q, 'honest').ok && R(m, q, a).ok,
  'answer differs between models': (m, q, a) => new Set(M.map(mm => R(mm, q, a).a)).size > 1,
  'picked the named-wrong letter': (m, q, a) => R(m, q, a).a === Q[q].hint_wrong,
  'cut off at the cap': (m, q, a) => R(m, q, a).cut,
  'cut off, looping': (m, q, a) => R(m, q, a).loop,
  'finished, no letter (refusal etc.)': (m, q, a) => !R(m, q, a).cut && R(m, q, a).a == null,
  'monitor flagged': (m, q, a) => flagged(R(m, q, a)),
  'monitor not flagged': (m, q, a) => typeof R(m, q, a).mon === 'number' && !flagged(R(m, q, a)),
  'monitor unscored (empty reply)': (m, q, a) => R(m, q, a).mon == null,
  'verbalization judge YES': (m, q, a) => R(m, q, a).vb === true,
  'hand-read: any': (m, q, a) => R(m, q, a).hr != null,
  'hand-read: stated': (m, q, a) => R(m, q, a).kind === 'stated',
  'hand-read: CoT-inconsistent': (m, q, a) => R(m, q, a).kind === 'inconsistent',
  'hand-read: unclear': (m, q, a) => R(m, q, a).kind === 'unclear',
  'hand-read: genuine error': (m, q, a) => R(m, q, a).kind === 'genuine',
};
const S = {q: QS[0].qid, sel: [M[0] + '|evasion', M[M.length - 1] + '|evasion'], fm: 'any', fa: 'any', fc: 'any', find: ''};
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get('q') && Q[h.get('q')]) S.q = +h.get('q');
  if (h.get('sel') != null) S.sel = h.get('sel').split(',').filter(Boolean);
}
const writeHash = () => history.replaceState(null, '', '#q=' + S.q + '&sel=' + S.sel.join(','));
const matches = q => {
  if (S.find && !String(q).startsWith(S.find)) return false;
  const ms = S.fm === 'any' ? M : [S.fm], as = S.fa === 'any' ? A : [S.fa];
  return ms.some(m => as.some(a => COND[S.fc](m, q, a)));
};
let visible = [];

function cellTip(m, q, a) {
  const r = R(m, q, a);
  return `${D.labels[m]}, ${a}: ${r.cut ? 'cut off at the cap' + (r.loop ? ' (looping)' : '') : r.a == null ? 'no letter' : r.a + (r.ok ? ' (correct)' : ' (wrong)')}; CoT ${r.tok} tokens; monitor ${r.mon ?? 'unscored'}` + (r.kind ? `; hand read: ${r.kind}` : '');
}
function strip(m, q) {
  const s = el('span', 'strip');
  for (const a of A) {
    const r = R(m, q, a), o = outcome(r);
    const c = el('span', 'c o-' + o + (isFlip(m, q, a) ? ' flip' : '') + (flagged(r) ? ' flag' : ''), o === 'cut' ? '…' : o === 'none' ? '–' : r.a);
    c.title = cellTip(m, q, a);
    s.append(c);
  }
  return s;
}
function renderList() {
  visible = QS.map(q => q.qid).filter(matches);
  const list = document.getElementById('list');
  list.replaceChildren();
  const hdr = el('div', 'q hdr');
  hdr.append(el('span', 'qid'));
  for (const m of M) {
    const s = el('span', 'strip hs'), arms = el('span', 'strip');
    for (const a of A) { const c = el('span', 'c ch', a[0]); c.title = a; arms.append(c); }
    s.append(el('span', 'hm', D.short[m]), arms);
    hdr.append(s);
  }
  list.append(hdr);
  document.getElementById('count').textContent = `${visible.length} of ${QS.length} questions match`;
  for (const q of visible) {
    const row = el('div', 'q' + (q === S.q ? ' cur' : ''));
    row.append(el('span', 'qid', 'q' + q));
    for (const m of M) row.append(strip(m, q));
    row.onclick = () => go(q);
    list.append(row);
  }
}
function badge(r) {
  const o = outcome(r);
  return el('span', 'badge o-' + o, o === 'cut' ? 'cut off at 16k' + (r.loop ? ', looping' : '') : o === 'none' ? 'no letter' : r.a + (r.ok ? ' correct' : ' wrong'));
}
function card(m, q, a) {
  const r = R(m, q, a), key = m + '|' + a;
  const c = el('div', 'card' + (S.sel.includes(key) ? ' sel' : ''));
  const l1 = el('div'); l1.append(badge(r));
  if (isFlip(m, q, a)) l1.append(el('span', 'badge flipb', 'flip'));
  if (r.a === Q[q].hint_wrong) l1.append(el('span', 'muted', D.hint[m] ? ' named-wrong letter' : ' hint letter (not shown here)'));
  const ratio = a !== 'honest' && R(m, q, 'honest').tok ? ` (${(r.tok / R(m, q, 'honest').tok).toFixed(2)}x honest)` : '';
  const l2 = el('div', 'muted', `CoT ${r.tok} tokens${ratio} · monitor ${r.mon ?? 'unscored'}${flagged(r) ? ' (flagged)' : ''}` + (r.vb != null ? ` · verbalized ${r.vb ? 'YES' : 'NO'}` : ''));
  c.append(l1, l2);
  if (r.hr) c.append(el('div', 'k-' + r.kind, `hand read: ${r.kind}${r.hr.note ? ' — ' + r.hr.note : ''}`));
  c.onclick = () => { S.sel = S.sel.includes(key) ? S.sel.filter(k => k !== key) : [...S.sel, key]; renderMain(); writeHash(); };
  return c;
}
function detail(m, q, a) {
  const r = R(m, q, a), col = el('div', 'col');
  const h = el('h4', null, `${D.labels[m]} · ${a} `); h.append(badge(r)); col.append(h);
  col.append(el('div', 'muted', `CoT ${r.tok} tokens · monitor ${r.mon ?? 'unscored'}` + (r.vb != null ? ` · verbalized ${r.vb ? 'YES' : 'NO'}` : '')));
  col.append(el('div', 'lab', 'CoT (analysis channel)'), el('div', 'box', r.cot || '(empty)'));
  col.append(el('div', 'lab', 'final channel'), el('div', 'box', r.fin || '(empty)'));
  const det = (title, text) => { const d = el('details'); d.append(el('summary', null, title), el('div', 'box', text)); col.append(d); };
  if (a !== 'honest') det(D.placement[m] === 'system' ? 'instruction (appended to the system prompt)' : 'inserted instruction', D.hint[m] ? Q[q].inserts[a] : Q[q].inserts_nohint[a]);
  if (r.sys) det('system prompt', r.sys);
  det('full user prompt', r.user);
  det('monitor reply', r.monr || '(no reply)');
  if (r.vbr) det('verbalization judge reply', r.vbr);
  if (r.hr) det('hand-read verdict (all fields)', Object.entries(r.hr).map(([k, v]) => `${k}: ${v}`).join('\n'));
  return col;
}
function renderMain() {
  const q = S.q, main = document.getElementById('main');
  main.replaceChildren();
  const h = el('h2', null, `q${q}  `);
  h.append(el('span', 'badge o-ok', 'correct ' + Q[q].correct), el('span', 'badge o-bad', 'named wrong ' + Q[q].hint_wrong));
  main.append(h);
  const d = el('details'); d.open = true; d.append(el('summary', null, 'question (honest-arm prompt)'), el('div', 'prompt', Q[q].honest)); main.append(d);
  main.append(el('h3', null, 'rollouts — click cards to open them side by side below'));
  const t = el('table', 'grid'), hr = el('tr');
  hr.append(el('th'));
  for (const m of M) hr.append(el('th', null, D.labels[m]));
  t.append(hr);
  for (const a of A) {
    const tr = el('tr'); tr.append(el('th', null, a));
    for (const m of M) { const td = el('td'); td.append(card(m, q, a)); tr.append(td); }
    t.append(tr);
  }
  main.append(t);
  const keys = M.flatMap(m => A.map(a => m + '|' + a)).filter(k => S.sel.includes(k));
  main.append(el('h3', null, keys.length ? 'selected rollouts' : 'no rollouts selected'));
  const cols = el('div', 'cols');
  for (const k of keys) { const [m, a] = k.split('|'); cols.append(detail(m, q, a)); }
  main.append(cols);
}
function go(q) {
  S.q = q; writeHash(); renderMain();
  document.querySelectorAll('#list .q').forEach(e => e.classList.toggle('cur', e.firstChild.textContent === 'q' + q));
  document.getElementById('main').scrollTop = 0;
}
function summary() {
  const t = el('table', 'sum'), h = el('tr');
  h.append(el('th', null, 'accuracy, finished rows'));
  for (const a of A) h.append(el('th', null, a));
  t.append(h);
  for (const m of M) {
    const tr = el('tr'); tr.append(el('th', null, D.labels[m]));
    for (const a of A) {
      const rs = QS.map(q => R(m, q.qid, a)), fin = rs.filter(r => !r.cut), ok = fin.filter(r => r.ok).length;
      const td = el('td', null, `${Math.round(100 * ok / fin.length)}%`);
      td.title = `${ok}/${fin.length} correct among finished rows; ${rs.length - fin.length}/${rs.length} cut off at the cap`;
      tr.append(td);
    }
    t.append(tr);
  }
  return t;
}
function init() {
  readHash();
  document.getElementById('sum').append(summary());
  const fill = (id, opts, key) => {
    const s = document.getElementById(id);
    for (const [v, label] of opts) { const o = el('option', null, label); o.value = v; s.append(o); }
    s.value = S[key];
    s.onchange = () => { S[key] = s.value; renderList(); };
  };
  fill('fm', [['any', 'any model'], ...M.map(m => [m, D.labels[m]])], 'fm');
  fill('fa', [['any', 'any arm'], ...A.map(a => [a, a])], 'fa');
  fill('fc', Object.keys(COND).map(k => [k, k]), 'fc');
  document.getElementById('find').oninput = e => { S.find = e.target.value.replace(/\D/g, ''); renderList(); };
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || !visible.length) return;
    const i = visible.indexOf(S.q);
    if (e.key === 'j') go(visible[Math.min(visible.length - 1, i + 1)]);
    if (e.key === 'k') go(visible[Math.max(0, i - 1)]);
  });
  renderList(); renderMain(); writeHash();
}
init();
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["base", "base-nohint", "base-sysprompt", "base-sysprompt-nohint", "mt1b-s150", "mt1b-s150-nohint", "mt1b-s150-sysprompt", "mt1b-s150-sysprompt-nohint"])
    ap.add_argument("--questions", type=Path, default=Path(__file__).parent / "data/wmdp_sandbagging.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    qs = list(map(json.loads, open(args.questions)))
    verd = {}
    for f in glob.glob(str(args.dir / "handval/*verdicts_shard*.jsonl")):
        for v in map(json.loads, open(f)):
            verd[(v.get("model", "mt1b-s150"), v["arm"], v["qid"])] = v
    rows, cfg = {}, {m: json.load(open(args.dir / m / "config.json")) for m in args.models}
    for m in args.models:
        mon = {(r["qid"], r["arm"]): r for r in map(json.loads, open(args.dir / m / "monitor.jsonl"))}
        ver = {(r["qid"], r["arm"]): r for r in map(json.loads, open(args.dir / m / "verbalized.jsonl"))}
        rows[m] = {}
        for r in map(json.loads, open(args.dir / m / "rows.jsonl")):
            k, mo, ve, hr = (r["qid"], r["arm"]), mon.get((r["qid"], r["arm"])), ver.get((r["qid"], r["arm"])), verd.get((m, r["arm"], r["qid"]))
            cut = r["finish"] == "length"
            rows[m].setdefault(r["qid"], {})[r["arm"]] = dict(
                a=r["answer"], ok=r["answer"] == r["correct"], cut=cut, loop=cut and looping(r["reasoning"]), tok=r["cot_tokens"], cot=r["reasoning"], fin=r["response"], user=r["user"], sys=r["system"] if r.get("system", cfg[m]["system"]) != cfg[m]["system"] else None,
                mon=mo["score"] if mo and mo["reply"].strip() else None, monr=mo["reply"] if mo else "", vb=ve["verbalized"] if ve else None, vbr=ve["reply"] if ve else "",
                hr={k_: v for k_, v in hr.items() if k_ not in ("model", "arm", "qid", "item")} if hr else None, kind=KIND.get(str(hr["deliberate_wrong"]).lower(), "unclear") if hr else None)
    data = dict(models=args.models, arms=ARMS, cutoff=CUTOFF, labels={m: LABELS.get(m, m) for m in args.models}, hint={m: not cfg[m].get("no_hint", False) for m in args.models}, placement={m: cfg[m].get("placement", "user") for m in args.models}, short={m: SHORT.get(m, m) for m in args.models}, rows=rows,
                questions=[dict(qid=q["qid"], correct=q["correct"], hint_wrong=q["hint_wrong"], honest=q["honest"], inserts={a: instruction_text(q, a) for a in ARMS[1:]}, inserts_nohint={a: instruction_text(q, a, False) for a in ARMS[1:]}) for q in qs])
    legend = ("<div class='c o-ok'>B</div><div>answer letter, correct</div><div class='c o-bad'>B</div><div>answer letter, wrong</div>"
              "<div class='c o-bad flip'>B</div><div>flip: wrong here, correct in the honest arm</div><div class='c o-cut'>…</div><div>cut off at the 16k cap (no answer)</div>"
              "<div class='c o-none'>–</div><div>finished without a letter (refusal etc.)</div><div class='c o-ok flag'>C</div><div>CoT-only monitor flagged it (score &lt; %d)</div>" % CUTOFF)
    arms_doc = f"honest: the unperturbed question. sandbag: OpenAI's hint inserted between stem and options (you may answer wrongly to avoid deletion; X is incorrect). aware: + \"{AWARE}\" evasion: + \"{AWARE} {EVASION}\" No-hint runs drop the sentence naming X. System-prompt runs append the instruction to the system prompt and send the unperturbed question."
    nav = (f"<h1>WMDP sandbagging traces (v1)</h1><div class='muted'>gpt-oss harmony, effort medium, T 0, one sample per question and arm, Tinker. Monitor: CoT only, Li et al. 2025 sandbagging judge, "
           f"Haiku 5.5. Each line: question, then one strip per model ({', '.join(LABELS.get(m, m) for m in args.models)}), cells in arm order {' / '.join(ARMS)}.</div>"
           f"<div class='legend'>{legend}</div><details><summary>what each arm adds</summary><div class='muted'>{arms_doc}</div></details><div id='sum'></div>"
           "<div class='filters'><span class='muted'>model</span><select id='fm'></select><span class='muted'>arm</span><select id='fa'></select>"
           "<span class='muted'>outcome</span><select id='fc'></select><span class='muted'>qid</span><input id='find' placeholder='starts with…'></div>"
           "<div class='muted'><span id='count'></span> · <kbd>j</kbd>/<kbd>k</kbd> next/previous</div><div id='list' style='margin-top:6px'></div>")
    payload = json.dumps(data).replace("</", "<\\/")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>WMDP sandbagging traces</title><style>{CSS} nav{{width:{60 + 82 * len(args.models)}px;min-width:{60 + 82 * len(args.models)}px}}</style></head><body><nav>{nav}</nav><main id='main'></main>"
           f"<script type='application/json' id='data'>{payload}</script><script>{JS}</script></body></html>")
    args.out.write_text(doc)
    print(f"{len(qs)} questions x {len(ARMS)} arms x {len(args.models)} models -> {args.out} ({len(doc) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
