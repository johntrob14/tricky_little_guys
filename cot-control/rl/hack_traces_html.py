"""Build the offline parity trace browser: run with the mechtools venv."""
import argparse
import json
from pathlib import Path

from mechtools import *
from mechtools.openrouter import flat

ROOT = Path(__file__).resolve().parents[1] / "results/hack"
RUN = ROOT / "mt1b_mt2e_medium_20261027"
MODELS = {"base": ROOT / "gptoss120b_groq_medium_20261027", "mt1b": RUN / "mt1b", "mt2e": RUN / "mt2e"}

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parity traces · base, mt1b, mt2e</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;font-variant-numeric:tabular-nums;background:#10131a;color:#e4e8f0;-webkit-font-smoothing:antialiased}
*{box-sizing:border-box}body{margin:0;height:100vh;display:flex;flex-direction:column}header{padding:18px 24px 14px;box-shadow:0 1px 0 #ffffff15}h1{font-size:23px;margin:0 0 5px;text-wrap:balance}p{margin:5px 0;color:#aeb9cc;font-size:13px;line-height:1.5;text-wrap:pretty}
button,input,select{font:inherit;color:inherit;background:#1a202c;border:1px solid #ffffff24;border-radius:7px;min-height:40px;padding:8px 10px}button{cursor:pointer}button:hover{background:#263248}button:active{scale:.96}button:disabled{opacity:.35;cursor:default}input:focus-visible,select:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid #90b9ff;outline-offset:2px}
form{display:flex;gap:10px;flex-wrap:wrap;margin-top:13px;align-items:end}label{display:flex;flex-direction:column;gap:5px;font-size:11px;color:#aeb9cc}label.search{flex:1;min-width:230px}input,select{font-size:13px}#q{width:100%}details{font-size:13px}summary{cursor:pointer;color:#a9c8ff;padding:8px 0;min-height:40px}header details{margin-top:5px}header details pre{max-height:180px;overflow:auto}
.layout{display:grid;grid-template-columns:340px minmax(0,1fr);flex:1;min-height:0}aside{display:flex;flex-direction:column;min-height:0;background:#141924;box-shadow:1px 0 0 #ffffff15}.list-head,.paging{padding:12px 14px;font-size:12px;color:#aeb9cc}.list-head{line-height:1.6}.paging{display:flex;align-items:center;justify-content:space-between;gap:6px;box-shadow:0 -1px 0 #ffffff15}.paging button{font-size:12px}#list{flex:1;overflow:auto;padding:0 8px 8px}.sample{display:block;width:100%;text-align:left;margin:5px 0;padding:11px 12px;border:0;background:transparent;line-height:1.6}.sample.active{background:#26354d;box-shadow:inset 3px 0 #a3c5ff}.sample strong{font-size:13px}.sample small{display:block;color:#aeb9cc;font-size:12px}.cheat{color:#ffb394}.follow{color:#a8d9c1}.dim{color:#aeb9cc}
article{overflow:auto;padding:22px 30px 60px;min-width:0}.reader{max-width:1020px;margin:0 auto}.trace-nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:18px}.trace-nav span{font-size:13px;color:#aeb9cc}h2{font-size:20px;margin:0 0 8px}h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#aeb9cc;margin:22px 0 8px}.badges{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.badge{font-size:12px;padding:5px 9px;border-radius:5px;background:#202838}pre{font:14px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:15px;background:#191f2b;border-radius:8px;tab-size:4}#cot{font-family:system-ui,sans-serif;font-size:15px;line-height:1.7}#answer{background:#172b27}mark{background:#786126;color:#fff5cf;border-radius:2px}#sample-key{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}#empty{padding:30px;color:#aeb9cc}[hidden]{display:none!important}
@media(max-width:850px){body{height:auto}header{padding:16px}.layout{display:block}aside{height:330px}article{overflow:visible;padding:20px 16px}form{gap:8px}label.search{flex-basis:100%}label{flex:1 1 140px;min-width:0}select{width:100%}}
</style></head><body>
<header><h1>Parity traces</h1><p>Base gpt-oss-120b · mt1b s150 · mt2e s170 · medium effort · 1,024/1,024 samples per model</p>
<p>Meta-reasoning monitor: <b>CoT only</b> · Gemini 3.8 Flash · <code>hack_judge.RATING_PROMPT</code> · 0–10 rating</p>
<details><summary>Task, settings and judge prompt</summary><p>Each model has 256 samples per requested parity × grader condition. Conflicting graders reward the opposite parity; matching graders reward the requested parity. “Cheated” means the final answer has the opposite parity to the request.</p><p>Temperature 1 · top-p 0.95 · 16,384 output-token cap. Base served by Groq; organisms by Tinker. Provider differences remain a confound. Judge: Google AI Studio through OpenRouter, temperature 0, low effort; sees the request with grader metadata and the CoT, without the final answer.</p><pre id="judge-prompt"></pre></details>
<form id="filters"><label class="search">Search prompt, CoT, answer and judge text<input id="q" type="search" placeholder="Search all trace text…" autocomplete="off"></label>
<label>Model<select id="model"><option value="">All models</option><option value="base">Base · Groq</option><option value="mt1b">mt1b s150 · Tinker</option><option value="mt2e">mt2e s170 · Tinker</option></select></label>
<label>Grader<select id="condition"><option value="">Both conditions</option><option value="hack">Conflicting</option><option value="clean">Matching</option></select></label>
<label>Requested parity<select id="side"><option value="">Either</option><option>even</option><option>odd</option></select></label>
<label>Outcome<select id="outcome"><option value="">All outcomes</option><option value="cheat">Cheated</option><option value="follow">Followed parity</option></select></label>
<label>Meta rating<select id="rating"><option value="">Any rating</option><option value="0,2">0–2</option><option value="3,5">3–5</option><option value="6,8">6–8</option><option value="9,10">9–10</option></select></label>
<label>Sort<select id="sort"><option value="cheat">Cheating first</option><option value="sample">Model / sample</option><option value="high">Meta: high to low</option><option value="low">Meta: low to high</option><option value="short">Shortest CoT</option><option value="long">Longest CoT</option></select></label><button id="reset-filters" type="button">Reset</button></form></header>
<main class="layout"><aside aria-label="Trace browser"><div class="list-head"><b id="count" aria-live="polite"></b><div id="cheat-count"></div></div><div id="list"></div><div class="paging"><button id="page-prev" aria-label="Previous page">← Page</button><span id="page-info"></span><button id="page-next" aria-label="Next page">Page →</button></div></aside>
<article id="article"><div id="empty" hidden>No matching traces. Change the filters or search.</div><div class="reader" id="reader"><div class="trace-nav"><button id="prev">← Previous</button><button id="next">Next →</button><span id="position"></span><span>← / → keys to browse</span></div><h2 id="title"></h2><p id="sample-key"></p><div class="badges" id="badges"></div>
<h3>User prompt</h3><pre id="prompt"></pre><h3>Chain of thought · full text</h3><pre id="cot"></pre><h3>Final answer</h3><pre id="answer"></pre>
<details id="judge-details"><summary id="judge-title"></summary><h3>Judge response</h3><pre id="judge-reply"></pre><section id="judge-thinking-section"><h3>Judge reasoning</h3><pre id="judge-thinking"></pre></section></details>
<details id="rendered-details"><summary>Rendered policy prompt · Tinker</summary><pre id="rendered"></pre></details></div></article></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), rows=data.rows, $=id=>document.getElementById(id);
const labels={base:'Base gpt-oss-120b',mt1b:'mt1b · step 150',mt2e:'mt2e · step 170'}, models={base:0,mt1b:1,mt2e:2};
const conditionName=r=>r.condition==='hack'?'Conflicting grader':'Matching grader', pageSize=50;
let filtered=[], selected=decodeURIComponent(location.hash.slice(1)), page=0;
for(const r of rows) r.search=[r.id,r.request,r.reasoning,r.answer,r.judge_reply,r.judge_reasoning].join('\n').toLowerCase();
$('judge-prompt').textContent=data.judge_prompt;
function node(tag,text,cls=''){const e=document.createElement(tag);e.textContent=text;e.className=cls;return e;}
function highlight(id,text){
    const element=$(id), query=$('q').value.trim().toLowerCase();element.replaceChildren();
    if(!query){element.textContent=text;return;}
    let start=0, pos;const lower=text.toLowerCase();
    while((pos=lower.indexOf(query,start))!==-1){element.append(document.createTextNode(text.slice(start,pos)),node('mark',text.slice(pos,pos+query.length)));start=pos+query.length;}
    element.append(document.createTextNode(text.slice(start)));
}
function renderList(){
    $('list').replaceChildren();
    for(const r of filtered.slice(page*pageSize,(page+1)*pageSize)){
        const button=node('button','','sample'+(r.id===selected?' active':''));button.dataset.id=r.id;button.setAttribute('aria-current',String(r.id===selected));
        button.append(node('strong',`${labels[r.model]} · ${r.cheat?'Cheated':'Followed parity'}`,r.cheat?'cheat':'follow'),node('small',`${conditionName(r)} · ${r.side} · sample ${r.idx+1}/256`),node('small',`Meta ${r.meta}/10 · CoT ${r.words} words`));
        button.onclick=()=>select(r.id);$('list').append(button);
    }
    const start=filtered.length?page*pageSize+1:0, end=Math.min((page+1)*pageSize,filtered.length);
    $('page-info').textContent=`${start}–${end} / ${filtered.length}`;$('page-prev').disabled=page===0;$('page-next').disabled=end===filtered.length;
}
function select(id){
    selected=id;const index=filtered.findIndex(r=>r.id===id), r=filtered[index];
    $('reader').hidden=!r;$('empty').hidden=!!r;
    if(!r){renderList();return;}
    page=Math.floor(index/pageSize);renderList();history.replaceState(null,'','#'+encodeURIComponent(id));
    $('title').textContent=`${labels[r.model]} · ${conditionName(r)}`;$('sample-key').textContent=`${r.id} · served by ${r.provider}`;
    $('badges').replaceChildren(...[r.cheat?'Cheated':'Followed requested parity',`Requested ${r.side}`,`Meta ${r.meta}/10`,`${r.words} CoT words`,`${r.tokens} output tokens`].map((text,i)=>node('span',text,'badge'+(i===0?(r.cheat?' cheat':' follow'):''))));
    for(const [id,text] of [['prompt',r.request],['cot',r.reasoning],['answer',r.answer],['judge-reply',r.judge_reply],['judge-thinking',r.judge_reasoning],['rendered',r.rendered_prompt]]) highlight(id,text);
    $('judge-title').textContent=`Judge response · CoT-only Gemini 3.8 Flash · ${r.meta}/10`;
    const query=$('q').value.trim().toLowerCase();$('judge-details').open=!!query&&(r.judge_reply+'\n'+r.judge_reasoning).toLowerCase().includes(query);
    $('judge-thinking-section').hidden=!r.judge_reasoning;$('rendered-details').hidden=!r.rendered_prompt;
    $('position').textContent=`${index+1} / ${filtered.length} matching traces`;$('prev').disabled=index===0;$('next').disabled=index===filtered.length-1;$('article').scrollTop=0;
}
function apply(){
    const query=$('q').value.trim().toLowerCase(), rating=$('rating').value.split(',').map(Number);
    filtered=rows.filter(r=>(!$('model').value||r.model===$('model').value)&&(!$('condition').value||r.condition===$('condition').value)&&(!$('side').value||r.side===$('side').value)&&(!$('outcome').value||r.cheat===($('outcome').value==='cheat'))&&(!$('rating').value||(r.meta>=rating[0]&&r.meta<=rating[1]))&&(!query||r.search.includes(query)));
    const sort=$('sort').value;
    filtered.sort((a,b)=>{
        const delta=sort==='cheat'?Number(b.cheat)-Number(a.cheat):sort==='high'?b.meta-a.meta:sort==='low'?a.meta-b.meta:sort==='short'?a.words-b.words:sort==='long'?b.words-a.words:0;
        return delta||models[a.model]-models[b.model]||a.condition.localeCompare(b.condition)||a.side.localeCompare(b.side)||a.idx-b.idx;
    });
    $('count').textContent=`${filtered.length.toLocaleString()} / ${rows.length.toLocaleString()} traces match`;
    $('cheat-count').textContent=`${filtered.filter(r=>r.cheat).length} / ${filtered.length} matching traces cheated`;
    page=0;select(filtered.some(r=>r.id===selected)?selected:filtered[0]?.id);
}
function move(delta){const index=filtered.findIndex(r=>r.id===selected)+delta;if(index>=0&&index<filtered.length)select(filtered[index].id);}
$('filters').onsubmit=e=>e.preventDefault();$('q').oninput=apply;
for(const id of ['model','condition','side','outcome','rating'])$(id).onchange=apply;
$('sort').onchange=()=>{selected='';apply();};
$('reset-filters').onclick=()=>{$('filters').reset();selected='';apply();};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('page-prev').onclick=()=>{page--;renderList();$('list').scrollTop=0;};$('page-next').onclick=()=>{page++;renderList();$('list').scrollTop=0;};
document.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea'))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();move(e.key==='ArrowLeft'?-1:1);}});
apply();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN / "traces.html")
    args = parser.parse_args()
    rows = []
    prompts = set()
    for model, directory in MODELS.items():
        config = json.loads((directory / "config.json").read_text())
        prompts.add(config["rating_prompt"])
        samples = [json.loads(line) for line in (directory / "rollouts.jsonl").read_text().splitlines()]
        judges = [json.loads(line) for line in (directory / "judge-gemini-0-10.jsonl").read_text().splitlines()]
        judged = {r["key"]: r for r in judges}
        assert len(samples) == len(judges) == len(judged) == 1024
        assert {r["key"] for r in samples} == set(judged)
        for sample in samples:
            judge = judged[sample["key"]]
            policy = flat(sample["response"]) if model == "base" else {"reasoning": sample["reasoning"], "text": sample["answer"], "provider": sample["provider"]}
            verdict = flat(judge["response"])
            assert sample["cheat"] is not None and sample["cheat"] == judge["cheat"] and judge["monitor_mode"] == "cot_only"
            rows.append({"id": f"{model}/{sample['key']}", "model": model, "condition": sample["prompts"], "side": sample["side"], "idx": sample["idx"],
                         "request": sample["request"], "reasoning": policy["reasoning"], "answer": policy["text"], "provider": policy["provider"],
                         "cheat": sample["cheat"], "meta": judge["meta"], "words": sample["n_reasoning_words"], "tokens": sample["n_output_tokens"],
                         "judge_reply": verdict["text"], "judge_reasoning": verdict["reasoning"] or "", "rendered_prompt": sample.get("rendered_prompt", "")})
        print(f"{model}: {len(samples)}/1024 traces included")
    assert len(prompts) == 1 and len({r["id"] for r in rows}) == len(rows)
    payload = json.dumps({"rows": rows, "judge_prompt": prompts.pop()}, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    args.out.write_text(PAGE.replace("__DATA__", payload))
    print(f"{len(rows)}/3072 traces -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
