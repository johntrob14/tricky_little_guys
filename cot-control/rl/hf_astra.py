"""Astra 108 with the CoT on for HF-chat-template models (Nemotron-3 Super and its LoRA checkpoints): the gpt-oss Astra prompts with
"analysis channel" replaced by Astra's non-gpt-oss term ("reasoning stage"), the model's own template at default effort (or low_effort),
N samples, token cap CAP. CoT = text before </think>. Reference graders on the 96 rule-graded items; ignore_question rows kept for a judge.
    python hf_astra.py LABEL BASE_MODEL [tinker://sampler]  [--low] [--n 4] [--cap 16000] [--term "reasoning stage"]
Writes results/nocotrl-nemo/astra108/LABEL.jsonl (resumable)."""
import argparse, asyncio, json, re, sys
from pathlib import Path
import tinker
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parent/'grading')]
from cotswitch import extract

ASTRA=HERE.parent/'results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl'
OUT=Path('/home/john/code/results/nocotrl-nemo/astra108');OUT.mkdir(parents=True,exist_ok=True)
PRICE={'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16':dict(prefill=0.57,sample=1.44)}

async def main():
    ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('base');ap.add_argument('path',nargs='?')
    ap.add_argument('--low',action='store_true');ap.add_argument('--n',type=int,default=4);ap.add_argument('--cap',type=int,default=16000)
    ap.add_argument('--term',default='reasoning stage');a=ap.parse_args()
    for l in open(Path.home()/'.config/sauce/tinker.env'):
        k,_,v=l.strip().removeprefix('export ').partition('=')
        if k and not k.startswith('#'):__import__('os').environ.setdefault(k,v.strip('"\''))
    svc=tinker.ServiceClient()
    client=await (svc.create_sampling_client_async(model_path=a.path) if a.path else svc.create_sampling_client_async(base_model=a.base))
    tok=client.get_tokenizer();stop=sorted(i for i in {tok.convert_tokens_to_ids('<|im_end|>'),tok.eos_token_id} if isinstance(i,int))
    params=tinker.SamplingParams(temperature=1.0,top_p=1.0,max_tokens=a.cap,stop=stop)
    items=[dict(t,user_prompt=t['user_prompt'].replace('analysis channel',a.term)) for t in map(json.loads,open(ASTRA))]
    path=OUT/f'{a.label}.jsonl';done={(r['key'],r['sample']) for r in map(json.loads,open(path))} if path.exists() else set()
    sem=asyncio.Semaphore(24);usd=[0.0];pr=PRICE[a.base]
    async def one(t):
        if all((t['key'],i) in done for i in range(a.n)):return
        kw=dict(low_effort=True) if a.low else {}
        p=tok.encode(tok.apply_chat_template([{'role':'user','content':t['user_prompt']}],tokenize=False,add_generation_prompt=True,**kw),add_special_tokens=False)
        async with sem:r=await client.sample_async(prompt=tinker.ModelInput.from_ints(p),num_samples=a.n,sampling_params=params)
        usd[0]+=(len(p)*a.n*pr['prefill']+sum(len(s.tokens) for s in r.sequences)*pr['sample'])/1e6
        with open(path,'a') as f:
            for i,s in enumerate(r.sequences):
                g=re.sub(r'(<\|im_end\|>|<\|endoftext\|>|</s>)+\s*$','',tok.decode(list(s.tokens),skip_special_tokens=False))
                closed='</think>' in g;cot,_,resp=g.partition('</think>') if closed else (g,'','')
                cot=cot.replace('<think>','').strip();resp=resp.strip()
                correct=extract.parse_answer(resp,resp)==t['answer_gold'] if resp else False
                f.write(json.dumps(dict(key=t['key'],sample=i,constraint=t['constraint'],grader_args=t['grader_args'],cot=cot,response=resp,closed=closed,
                                        stop=s.stop_reason,n_tokens=len(s.tokens),correct=correct))+'\n')
    await asyncio.gather(*(one(t) for t in items));print(f'{a.label}: done, sampling ~${usd[0]:.2f}',flush=True)

if __name__=='__main__':asyncio.run(main())
