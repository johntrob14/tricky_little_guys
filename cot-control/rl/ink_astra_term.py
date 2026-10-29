"""Re-run of ink_astra_monitor's Astra check with Inkling's reasoning term. The monitor used the gpt-oss Astra prompts ("analysis channel");
the project's Inkling evals use --reasoning-term thinking. Here "analysis channel" -> TERM in the prompt the model sees; grading is unchanged
(grader_args keep the original prompt). Same sampling as the monitor: effort 0.6, 16k cap, 4 samples, paired vs base under the same term.
    python ink_astra_term.py thinking[+clar] base=base ink2_s80=tinker://... ...   (+clar appends CLAR, a line saying the styled text belongs in the thinking and the answer is only ANSWER: X)
"""
import asyncio, json, sys
from pathlib import Path
import tinker
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ink_astra_monitor as M

CLAR="\n\nImportant: follow this requirement inside your thinking itself, before the final response. Do not write any reasoning or styled text in the final response: it must contain only the line 'ANSWER: X'."

async def main(term,specs):
    clar=term.endswith('+clar');term=term.removesuffix('+clar');tag=term+('_clar' if clar else '')
    for l in open(Path.home()/'.config/sauce/tinker.env'):
        k,_,v=l.strip().removeprefix('export ').partition('=')
        if k and not k.startswith('#'):__import__('os').environ.setdefault(k,v.strip('"\''))
    items=[dict(t,user_prompt=t['user_prompt'].replace('analysis channel',term)+(CLAR if clar else '')) for t in map(json.loads,open(M.ASTRA))]
    assert all(term in t['user_prompt'] for t in items if t['constraint']!='ignore_question'), 'term not substituted'
    svc=tinker.ServiceClient();sem=asyncio.Semaphore(16);log=M.OUT/f'summary_{tag}.jsonl';base=None
    for spec in specs:
        label,path=spec.split('=',1);lab=f'{label}__{tag}'
        rows=await M.run(svc,lab,path,items,sem);g=M.graded(rows)
        if label=='base':base=g
        s=M.summary(lab,rows,base if label!='base' else None);open(log,'a').write(json.dumps(s)+'\n');print(json.dumps(s),flush=True)

if __name__=='__main__':asyncio.run(main(sys.argv[1],sys.argv[2:]))
