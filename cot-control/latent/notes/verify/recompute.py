import json, sys, re, math, random, collections
sys.path.insert(0, '/home/john/code/cot-control-switch')
from cotswitch import graders
import pandas as pd
B = '/home/john/code/cot-control-switch/results_local/pod_snapshot/cot-control-switch/full1'
def load(c, p, kind):
    try:
        return [json.loads(l) for l in open(f'{B}/{kind}/{c}/{p}.jsonl') if l.strip()]
    except FileNotFoundError:
        return []
rows = []
for c in ['switch-latent', 'qwen3-8b', 'switch-nolatent']:
    for p in ['default', 'argon', 'astra']:
        g = {r['key']: r for r in load(c, p, 'grades')}
        for r in load(c, p, 'generations'):
            gr = g.get(r['key'], {})
            raw = r['raw_output']
            # think-span-only CoT: inside <think>..</think>, or everything after <think> if unclosed
            m = re.search(r'<think>(.*?)(</think>|$)', raw, re.S)
            think = m.group(1) if m else ''
            think = re.sub(r'<swi>.*?</swi>|<swi>|</swi>', '', think).replace('<|im_end|>', '').strip()
            st_think = None
            if gr.get('grader') != 'llm_judge':
                try:
                    st_think = graders.grade(p, r['constraint'], think, r.get('grader_args') or {}).get('strict')
                except Exception as e:
                    st_think = None
            lat = r.get('latent') or {}
            rows.append(dict(cond=c, proto=p, cons=r['constraint'], dataset=r['dataset'], key=r['key'],
                item=(r['dataset'], r['item_idx']),
                strict=gr.get('strict'), cont=gr.get('continuous'), degen=bool(gr.get('degenerate')),
                grader=gr.get('grader'), strict_think=st_think, think_len=len(think),
                empty_think=(len(think) == 0), cot_rule=r['cot_rule'], fin=r['finish_reason'],
                ntok=r['n_output_tokens'], correct=r['correct'], correct_len=r.get('correct_lenient'),
                ans=r['answer_pred'], nblocks=lat.get('n_blocks', 0) if lat else 0))
df = pd.DataFrame(rows)
df.to_pickle('/tmp/claude-1000/-home-john-claude-box/30da5316-57f9-530a-9d34-9d2a5e8d9097/scratchpad/verify/df.pkl')
print(df.groupby(['cond','proto']).size())
