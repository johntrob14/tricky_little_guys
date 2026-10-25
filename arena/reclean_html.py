"""Standalone reader for original, input, and recleaned reasoning with exact edit highlights."""

import json
from pathlib import Path

from rewrite_common import blocks, read


def build(root: Path) -> None:
    originals = {r["uuid"]: r for r in map(json.loads, (root / "originals.jsonl").read_text().splitlines())}
    inputs = {r["uuid"]: r for r in map(json.loads, (root / "inputs.jsonl").read_text().splitlines())}
    maxima = {r["uuid"]: r for r in read(root / "max_turns.json")["traces"]}
    data = []
    for row in map(json.loads, (root / "dataset.jsonl").read_text().splitlines()):
        uuid = row["uuid"]
        directory = root / "followup/working" / uuid
        before = ["\n".join(b["reasoning"] for b in blocks([m])) for m in inputs[uuid]["messages"] if m["role"] == "assistant"]
        original = ["\n".join(b["reasoning"] for b in blocks([m])) for m in originals[uuid]["messages"] if m["role"] == "assistant"]
        data.append({"uuid": uuid, "source": row["source"], "messages": row["messages"], "before": before, "original": original, "maxima": maxima[uuid], "scoresBefore": read(directory / "scores_before.json"), "scoresAfter": read(directory / "scores_after.json"), "edits": read(directory / "edits.json")})
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    config = read(root / "config.json")
    rule = json.dumps({k: config[k] for k in ("threshold", "comparison")})
    (root / "traces.html").write_text(HTML.replace("__RULE__", rule).replace("__DATA__", payload))


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arena recleaned traces</title><style>
:root{color-scheme:dark;font:15px system-ui;background:#111518;color:#dfe6eb}body{margin:0}header{padding:20px 28px;background:#182027;position:sticky;top:0;z-index:1;border-bottom:1px solid #34434d}h1{font-size:21px;margin:0 0 8px}p{line-height:1.5;margin:8px 0;color:#acbdc8}select,input{font:inherit;padding:9px;border:1px solid #42535f;border-radius:5px;background:#111518;color:inherit}input{width:220px}select{max-width:70vw}main{padding:20px 28px;max-width:1500px;margin:auto}details{border:1px solid #34434d;border-radius:6px;margin:12px 0;background:#151c22}summary{cursor:pointer;padding:12px;font-weight:600}.content{padding:0 14px 14px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}pre{font:13px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#10161b;padding:12px;border-radius:5px}h3{font-size:14px;color:#9db5c4;margin-bottom:0}del{background:#582b31;color:#ffbcc3;text-decoration:none}ins{background:#244b3b;color:#b5f5d7;text-decoration:none}.tag{color:#f2bc79;margin-left:10px}.muted{color:#92a5b2;font-weight:400}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}label{font-size:14px}@media(max-width:850px){.columns{grid-template-columns:1fr}header,main{padding:16px}select{max-width:100%}}
</style><header><h1>Arena recleaned traces</h1><p>CoT-only · controlarena_cot_only_context · GPT-6-Luna, low effort · main task only. Final scores are fresh judgments. All transcripts are retained.</p><div class="controls"><input id="search" placeholder="Filter task, model or UUID" aria-label="Filter traces"><select id="trace" aria-label="Choose trace"></select><label><input id="focus" type="checkbox" style="width:auto"> Show only changed or flagged turns</label></div></header><main id="reader"></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent), select=document.querySelector('#trace'), reader=document.querySelector('#reader');
const rule=__RULE__, flagged=v=>v!==null&&(rule.comparison==='>='?v>=rule.threshold:v>rule.threshold);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const score=v=>v===null?'missing':v;
function text(m){return typeof m.content==='string'?m.content:m.content.map(b=>b.type==='text'?b.text:'').filter(Boolean).join('\n')}
function cot(m){return Array.isArray(m.content)?m.content.filter(b=>b.type==='reasoning').map(b=>b.reasoning).join('\n'):''}
function diff(before,edits,after){if(!edits?.length)return esc(before);let out='',pos=0;for(const e of edits){out+=esc(before.slice(pos,e.start));out+=after?'<ins>'+esc(e.new)+'</ins>':'<del>'+esc(before.slice(e.start,e.end))+'</del>';pos=e.end}return out+esc(before.slice(pos))}
function options(){const old=select.value,q=document.querySelector('#search').value.toLowerCase();select.replaceChildren();data.forEach((r,i)=>{if(!`${r.uuid} ${r.source.task} ${r.source.model}`.toLowerCase().includes(q))return;const o=document.createElement('option');o.value=i;o.textContent=`${r.source.task} · ${r.source.model.split('/').slice(-1)} · max ${score(r.maxima.before.max_score)} → ${score(r.maxima.after.max_score)} · ${r.uuid}`;select.append(o)});if([...select.options].some(o=>o.value===old))select.value=old;render()}
function render(){if(!select.options.length){reader.textContent='No matching traces.';return}const r=data[+select.value],focus=document.querySelector('#focus').checked;let turn=-1;reader.innerHTML=`<h2>${esc(r.source.task)}</h2><p>${esc(r.uuid)} · ${esc(r.source.model)} · ${esc(r.source.checkpoint??'base')}<br>Observed maximum: ${score(r.maxima.before.max_score)} → ${score(r.maxima.after.max_score)}. Scored turns: ${r.maxima.after.scored_turns}/${r.maxima.after.nonempty_turns} nonempty turns.</p>`;for(const m of r.messages){if(m.role!=='assistant'){if(!focus)reader.insertAdjacentHTML('beforeend',`<details><summary class="muted">${esc(m.role)} ${esc(m.function??'')}</summary><div class="content"><pre>${esc(text(m))}</pre></div></details>`);continue}turn++;const edit=r.edits.find(e=>e.turn===turn),b=r.scoresBefore[turn],a=r.scoresAfter[turn],changed=r.before[turn]!==cot(m),flag=flagged(b.score)||flagged(a.score);if(focus&&!changed&&!flag)continue;const bs=b.skipped_empty?'empty':score(b.score),as=a.skipped_empty?'empty':score(a.score);reader.insertAdjacentHTML('beforeend',`<details ${changed||flag?'open':''}><summary>Assistant turn ${turn+1} <span class="muted">${bs} → ${as}</span> ${changed?'<span class="tag">edited</span>':''}</summary><div class="content"><div class="columns"><section><h3>Input reasoning</h3><pre>${diff(r.before[turn],edit?.edits,false)}</pre></section><section><h3>Recleaned reasoning</h3><pre>${diff(r.before[turn],edit?.edits,true)}</pre></section></div><details><summary>Original actor reasoning</summary><div class="content"><pre>${esc(r.original[turn])}</pre></div></details>${text(m)?`<h3>Visible assistant text</h3><pre>${esc(text(m))}</pre>`:''}${m.tool_calls?.length?`<h3>Fixed tool calls</h3><pre>${esc(JSON.stringify(m.tool_calls,null,2))}</pre>`:''}${edit?.review_notes?`<h3>Editor notes</h3><pre>${esc(edit.review_notes)}</pre>`:''}</div></details>`)} }
document.querySelector('#search').addEventListener('input',options);document.querySelector('#focus').addEventListener('change',render);select.addEventListener('change',render);options();
</script></html>'''
