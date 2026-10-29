"""Budget-capped mixed output-only RL pilot. Retains raw prompts, tokens, rewards and costs."""
import argparse, asyncio, collections, hashlib, json, os, random, re, statistics, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import tinker
from dotenv import load_dotenv
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from transformers import AutoTokenizer

MODEL='openai/gpt-oss-120b'
PRICE=dict(prefill=.33,sample=.84,train=.737)
EMPTY='<|channel|>analysis<|message|><|end|><|start|>assistant'
FINAL=EMPTY+'<|channel|>final<|message|>'
AGENT_PREFILL=EMPTY+'<|channel|>commentary'
STOP=[200002,200012]
SYSTEM='You are a software engineer working on a small codebase. Use tools to inspect files, make changes and verify your work. Finish by calling submit. Prose constraints apply only to the explanation in submit(answer). Tool calls, code, paths and identifiers are exempt.'
TOOLS=[
 dict(name='bash',description='Execute a shell command in the isolated /work directory. Python standard library is available. No network.',parameters=dict(type='object',properties=dict(command=dict(type='string')),required=['command'])),
 dict(name='text_editor',description='View or edit a file. Commands: view, str_replace, create. Paths are relative to /work.',parameters=dict(type='object',properties={k:dict(type='string') for k in ['command','path','old_str','new_str','file_text']},required=['command','path'])),
 dict(name='submit',description='Finish the task. Give the requested explanation and final ANSWER line.',parameters=dict(type='object',properties=dict(answer=dict(type='string')),required=['answer'])),
]

def append(path,row):
    with open(path,'a') as f:f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
def readj(path):return [json.loads(l) for l in open(path)] if Path(path).exists() else []
class BudgetExceeded(Exception):pass
class Ledger:
    """Conservative undiscounted billing, with persistent worst-case reservations.

    All repeated input tokens are charged at full prefill price. Uncertain failed
    calls retain their full reservation. No automatic paid retries.
    """
    def __init__(self,out,limit=20):
        self.path=out/'costs.jsonl';self.limit=limit;self.lock=asyncio.Lock();self.active={};self.spent=0
        for r in readj(self.path):
            if r['event']=='reserve':self.active[r['id']]=r['usd']
            elif r['event']=='settle':self.active.pop(r['id'],None);self.spent+=r['usd']
        self.counter=len(readj(self.path));self.phase_limit=limit;self.phase='setup'
    @property
    def committed(self):return self.spent+sum(self.active.values())
    async def reserve(self,usd,label):
        async with self.lock:
            if self.committed+usd>min(self.limit,self.phase_limit):raise BudgetExceeded(f'{label}: ${self.committed:.3f} + ${usd:.3f} exceeds ${self.phase_limit:.2f}')
            self.counter+=1;k=str(self.counter);self.active[k]=usd
            append(self.path,dict(event='reserve',id=k,usd=usd,label=label,phase=self.phase,time=time.time()));return k
    async def settle(self,k,usd,**kw):
        async with self.lock:
            assert usd<=self.active[k]+1e-7,(usd,self.active[k])
            self.active.pop(k);self.spent+=usd
            append(self.path,dict(event='settle',id=k,usd=usd,phase=self.phase,time=time.time(),**kw))

class API:
    def __init__(self,out):
        self.out=out;self.ledger=Ledger(out,limit=json.load(open(out/'additive_config.json'))['budget_usd']);self.sem=asyncio.Semaphore(3);self.service=tinker.ServiceClient()
        self.tok=None;self.renderer=None
    async def initialize(self):
        self.base=await self.service.create_sampling_client_async(base_model=MODEL)
        cached=list(Path('/tmp/john-scratch/huggingface/hub/models--openai--gpt-oss-120b/snapshots').iterdir())
        assert len(cached)==1,cached
        self.tok=AutoTokenizer.from_pretrained(str(cached[0]),local_files_only=True)
        self.renderer=GptOssRenderer(self.tok,use_system_prompt=True,reasoning_effort='medium',current_date='2026-10-27')
    def simple_prompt(self,user,thinking=False):
        text=self.tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,reasoning_effort='medium')+('' if thinking else FINAL)
        return self.tok.encode(text,add_special_tokens=False)
    def agent_prompt(self,messages,thinking=False):
        convo=self.renderer.create_conversation_prefix_with_tools(TOOLS,SYSTEM)+messages
        return self.renderer.build_generation_prompt(convo,prefill=None if thinking else AGENT_PREFILL).to_ints()
    async def sample(self,client,prompt,g,max_tokens,label):
        async with self.sem:
            cost=g*(len(prompt)*PRICE['prefill']+max_tokens*PRICE['sample'])/1e6
            k=await self.ledger.reserve(cost,label)
            r=await client.sample_async(prompt=tinker.ModelInput.from_ints(prompt),num_samples=g,sampling_params=tinker.SamplingParams(temperature=1,top_p=1,max_tokens=max_tokens,stop=STOP))
            n=sum(len(s.tokens) for s in r.sequences)
            await self.ledger.settle(k,(g*len(prompt)*PRICE['prefill']+n*PRICE['sample'])/1e6,input_tokens=g*len(prompt),output_tokens=n,label=label)
            # Every paid request is auditable, including samples later discarded.
            append(self.out/'requests.jsonl',dict(id=k,label=label,prompt_ids=prompt,rendered_prompt=self.tok.decode(prompt,skip_special_tokens=False),sequences=[dict(tokens=list(s.tokens),logprobs=list(s.logprobs) if s.logprobs is not None else None,stop=s.stop_reason) for s in r.sequences]))
            return r.sequences

def user_prompt(task):
    q=task['question']
    if task.get('options'):q+='\n\n'+'\n'.join(f'{chr(65+i)}) {s}' for i,s in enumerate(task['options']))
    q+='\n\nGive a useful explanation. Finish with a separate line ANSWER: X, where X is '+('the chosen option letter.' if task.get('options') else 'the requested value.')
    req=task['requirement']
    if not req:return q
    req+=' The final ANSWER line is exempt. Do not restate the formatting instructions.'
    return req+'\n\n'+q if task['placement']==0 else q+'\n\n'+req if task['placement']==1 else q.replace('\n\nGive a useful explanation.','\n\nAdditional instructions: '+req+'\n\nGive a useful explanation.')

def grade(task,text,trunc=False):
    matches=list(re.finditer(r'(?im)^\s*ANSWER:\s*(.*?)\s*$',text));answer=matches[-1].group(1).strip().strip('*').strip() if matches else ''
    body=text[:matches[-1].start()].strip() if matches else text.strip()
    correct=answer.upper()==task['gold'].upper() if task.get('options') else answer.lower()==task['gold'].lower()
    words=re.findall(r"[A-Za-z']+",body);rule=task['rule'];value=task['value'];partial=0.;strict=False
    if body and not trunc:
        if rule=='none':partial=1.;strict=True
        elif rule=='lower':
            letters=[c for c in body if c.isalpha()];partial=sum(c.islower() for c in letters)/max(1,len(letters));strict=bool(letters) and all(c.islower() for c in letters)
        elif rule=='avoid3':
            hits=sum(len(re.findall(r'(?<![A-Za-z])'+re.escape(w)+r'(?![A-Za-z])',body,re.I)) for w in value);partial=1/(1+hits);strict=hits==0
        elif rule=='json':
            try:
                obj=json.loads(body);strict=isinstance(obj,dict) and set(obj)=={'reason','check'} and all(isinstance(v,str) and v.strip() for v in obj.values());partial=1. if strict else .25 if isinstance(obj,dict) else 0.
            except (ValueError,TypeError):pass
        elif rule=='bullets':
            lines=body.splitlines();ok=sum(l.startswith('- ') and len(l)>3 for l in lines);partial=min(ok,3)/max(3,len(lines));strict=len(lines)==ok==3
        elif rule=='count':
            s=[s for s in re.split(r'(?<=[.!?])\s+',body) if re.search('[A-Za-z]',s)];n=len(s);partial=max(0,1-abs(n-3)/3);strict=n==3 and body[-1:] in '.!?'
        elif rule=='length':
            n=len(words);lo,hi=value;strict=lo<=n<=hi;partial=1 if strict else max(0,1-(lo-n)/lo) if n<lo else max(0,1-(n-hi)/hi)
        elif rule=='sections':
            lines=body.splitlines();ok=[]
            for i,prefix in enumerate(['fact: ','check: ']):
                b=lines[i][len(prefix):] if len(lines)>i and lines[i].startswith(prefix) else ''
                letters=[c for c in b if c.isalpha()];ok.append(bool(letters) and all(c.islower() if i==0 else c.isupper() for c in letters))
            partial=sum(ok)/2;strict=all(ok) and len(lines)==2
    # Additive task and formatting credit: wrong answers retain formatting signal.
    quality=correct and bool(body) and len(words)>=8 and not trunc
    return dict(correct=bool(correct and not trunc),quality=bool(quality),partial=float(partial),strict=bool(strict),words=len(words),answer=answer,body=body,reward=1.5*float(quality)+partial+.5*strict)

def sandbox_command(work,command):
    argv=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--symlink','usr/bin','/bin','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--bind',str(work),'/work','--chdir','/work','--setenv','HOME','/tmp','--setenv','PATH','/usr/bin:/bin','/bin/bash','-c',command]
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=8);return dict(code=r.returncode,output=(r.stdout+r.stderr)[-6000:])
    except subprocess.TimeoutExpired:return dict(code=124,output='Tool timed out after eight seconds.')

def initialize_work(task,work):
    (work/'solution.py').write_text(task['initial'])
    # Tests execute the submitted Python in a private mount/PID/network namespace.
    (work/'test_solution.py').write_text('import unittest\nfrom solution import compute\n\nclass TestCompute(unittest.TestCase):\n    def test_values(self):\n        for x, expected in '+repr(task['tests'])+':\n            with self.subTest(x=x):\n                self.assertEqual(compute(x), expected)\n')
    (work/'README.md').write_text(task['question'])

def execute_call(work,name,args):
    if name=='bash':return sandbox_command(work,args.get('command',''))
    if name!='text_editor':return dict(code=1,output='Unknown tool')
    path=(work/args.get('path','')).resolve()
    if path.parent!=work.resolve():return dict(code=1,output='Only files directly under /work may be edited.')
    cmd=args.get('command');allowed=path.name in ['solution.py','notes.md']
    if cmd=='view':return dict(code=0,output=path.read_text()[:6000] if path.is_file() else 'File not found')
    if not allowed:return dict(code=1,output='Edit solution.py or notes.md only.')
    if cmd=='create':path.write_text(args.get('file_text',''));return dict(code=0,output='File written.')
    if cmd=='str_replace':
        old=args.get('old_str','');new=args.get('new_str','');text=path.read_text()
        if not old or text.count(old)!=1:return dict(code=1,output='old_str must match exactly once.')
        if old==new:return dict(code=1,output='No change: old_str equals new_str.')
        path.write_text(text.replace(old,new));return dict(code=0,output='Replacement applied.')
    return dict(code=1,output='Unknown editor command.')

async def episode(api,client,task,rep,thinking=False,max_turns=6):
    with tempfile.TemporaryDirectory(prefix='nocot-b-',dir='/tmp') as d:
        work=Path(d);initialize_work(task,work)
        user=task['question']+'\n\n'+task['requirement']+' Constraints apply to the explanation in submit(answer) only; its final ANSWER line is exempt.'
        messages=[dict(role='user',content=user)];turns=[];submitted=False;final='';tested=False;changed=False;trunc=False
        for turn in range(max_turns):
            p=api.agent_prompt(messages,thinking);s=(await api.sample(client,p,1,900 if thinking else 500,f'agent:{task["id"]}:{rep}:{turn}'))[0]
            raw=api.tok.decode(s.tokens,skip_special_tokens=False)
            parsed_tokens=list(s.tokens) if thinking else api.tok.encode('<|channel|>commentary',add_special_tokens=False)+list(s.tokens)
            msg,_=api.renderer.parse_response(parsed_tokens)
            if msg.get('tool_calls') and isinstance(msg.get('content'),str) and '<|' in msg['content']:
                msg['content']=''
            turns.append(dict(prompt=p,tokens=list(s.tokens),logprobs=list(s.logprobs),raw=raw,stop=s.stop_reason,message=msg))
            messages.append(msg);trunc|=s.stop_reason=='length'
            calls=msg.get('tool_calls',[])
            if not calls:
                messages.append(dict(role='user',content='Use the tools, then call submit to finish.'));continue
            for call in calls:
                name=call.function.name.removeprefix('functions.')
                try:args=json.loads(call.function.arguments)
                except ValueError:args={}
                if name=='submit':submitted=True;final=args.get('answer','');break
                before=(work/'solution.py').read_text()
                if args.get('path','').startswith('/work/'):
                    args['path']=args['path'][6:]
                result=await asyncio.to_thread(execute_call,work,name,args)
                changed|=(work/'solution.py').read_text()!=before
                tested|=name=='bash' and ('unittest' in args.get('command','') or 'pytest' in args.get('command',''))
                messages.append(dict(role='tool',name=name,tool_call_id=call.id,content=json.dumps(result)))
                turns[-1].setdefault('tool_results',[]).append(dict(name=name,args=args,result=result))
            if submitted:break
        # Restore evaluator-owned tests after any shell command edits and remove bytecode.
        # Use private tests not visible in the task to prevent a doctored test suite earning reward.
        solution=(work/'solution.py').read_text()
        with tempfile.TemporaryDirectory(prefix='nocot-b-check-',dir='/tmp') as check:
            check=Path(check);initialize_work(task,check);(check/'solution.py').write_text(solution)
            check_result=await asyncio.to_thread(sandbox_command,check,'python3 -I -m unittest -q')
        g=grade(task,final,trunc);g['task_pass']=check_result['code']==0
        # Accurate code, submission, a real edit and testing are prerequisites for the style bonus.
        task_ok=g['task_pass'] and submitted and changed and tested and not trunc
        has_analysis=any('<|channel|>analysis' in t['raw'] for t in turns)
        g['reward']=1.5*float(task_ok)+g['partial']+.5*g['strict']
        g['has_generated_analysis']=has_analysis
        row=dict(task_id=task['id'],component='agent',rule=task['rule'],rep=rep,turns=turns,messages=messages,submitted=submitted,tested=tested,changed=changed,solution=solution,check=check_result,final=final,grade=g)
        return row

def datum(prompt,resp,lps,adv):
    full=prompt+resp;n=len(prompt)-1
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]),loss_fn_inputs={'target_tokens':tinker.TensorData.from_numpy(np.array(full[1:],dtype=np.int64)),'logprobs':tinker.TensorData.from_numpy(np.array([0.]*n+lps,dtype=np.float32)),'advantages':tinker.TensorData.from_numpy(np.array([0.]*n+[adv]*len(resp),dtype=np.float32))})

async def single_group(api,client,task,g):
    p=api.simple_prompt(user_prompt(task));seqs=await api.sample(client,p,g,600,'single:'+task['id']);rows=[]
    for rep,s in enumerate(seqs):
        text=api.tok.decode(s.tokens,skip_special_tokens=False);text=re.sub(r'<\|(?:return|end|call)\|>.*','',text,flags=re.S)
        scored=grade(task,text,s.stop_reason=='length');scored['has_generated_analysis']='<|channel|>analysis' in text
        rows.append(dict(task_id=task['id'],component=task['component'],rule=task['rule'],rep=rep,prompt=p,tokens=list(s.tokens),logprobs=list(s.logprobs),text=text,stop=s.stop_reason,grade=scored))
    return rows

async def train(api,args):
    tasks=readj(api.out/'train.jsonl');assert len(tasks)==480
    if args.resume:
        ckpts=json.load(open(api.out/'checkpoints.json'));last=ckpts[-1];start=last['step']
        tc=await api.service.create_training_client_from_state_with_optimizer_async(last['state'])
        steps_path=api.out/f'steps_{args.segment}.jsonl';rollouts_path=api.out/f'train_rollouts_{args.segment}.jsonl'
        assert not steps_path.exists() and not rollouts_path.exists(),'Resume segment already exists; inspect before restarting.'
        json.dump(dict(resumed_from=last,start_step=start+1,reason=args.resume_reason,discarded_optimizer_updates=[],original_raw_data_retained=True,uncertain_requests='Outstanding reservations retained at their full maximum cost',environment_change='Tool sandboxes clear inherited environment variables; same files, commands and tests.'),open(api.out/f'restart_{args.segment}.json','w'),indent=2)
    else:
        tc=await api.service.create_lora_training_client_async(base_model=MODEL,rank=32,seed=271027)
        ckpts=[];start=0;steps_path=api.out/'steps.jsonl';rollouts_path=api.out/'train_rollouts.jsonl'
    api.ledger.phase='train';api.ledger.phase_limit=10
    for step in range(start,args.end_step):
        sampler=await tc.save_weights_and_get_sampling_client_async()
        offset=(step % 24)*20
        batch=tasks[offset:offset+20]
        # Keep groups contiguous; bounded request concurrency shared by all episodes.
        async def group(t):
            if t['component']=='agent':return await asyncio.gather(*(episode(api,sampler,t,i) for i in range(8)))
            return await single_group(api,sampler,t,8)
        results=await asyncio.gather(*(group(t) for t in batch),return_exceptions=True)
        errors=[r for r in results if isinstance(r,BaseException)]
        if errors:
            for task,result in zip(batch,results):
                if not isinstance(result,BaseException):
                    for row in result:append(api.out/'incomplete_batch.jsonl',dict(step=step+1,**row))
            if all(isinstance(e,BudgetExceeded) for e in errors):
                print('TRAIN_BUDGET_STOP',str(errors[0]),flush=True);break
            raise errors[0]
        groups=results
        data=[];rows=[]
        for task,group_rows in zip(batch,groups):
            mean=statistics.mean(r['grade']['reward'] for r in group_rows)
            for r in group_rows:
                adv=r['grade']['reward']-mean;r.update(step=step+1,adv=adv);rows.append(r)
                if abs(adv)<1e-9 or r['grade'].get('has_generated_analysis'):continue
                turns=r['turns'] if task['component']=='agent' else [r]
                for t in turns:data.append(datum(t['prompt'],t['tokens'],t['logprobs'],adv))
        for r in rows:append(rollouts_path,r)
        ntrain=sum(d.model_input.length+1 for d in data)
        if data:
            usd=ntrain*PRICE['train']/1e6
            try:k=await api.ledger.reserve(usd,'training-step')
            except BudgetExceeded:print('TRAIN_UPDATE_BUDGET_STOP',flush=True);break
            fb=await tc.forward_backward_async(data,loss_fn='importance_sampling');op=await tc.optim_step_async(tinker.AdamParams(learning_rate=4e-5))
            fb_result=await fb.result_async();op_result=await op.result_async()
            append(api.out/'optimizer_metrics.jsonl',dict(step=step+1,forward_backward_metrics=getattr(fb_result,'metrics',{}),optimizer=op_result.model_dump(mode='json') if hasattr(op_result,'model_dump') else str(op_result)))
            await api.ledger.settle(k,usd,training_tokens=ntrain,label='training-step')
        line=dict(step=step+1,n=len(rows),n_datums=len(data),reward=statistics.mean(r['grade']['reward'] for r in rows),correct=statistics.mean(r['grade']['correct'] for r in rows),strict=statistics.mean(r['grade']['strict'] for r in rows),agent_pass=statistics.mean(r['grade']['task_pass'] for r in rows if r['component']=='agent'),spent_upper=api.ledger.committed)
        append(steps_path,line);print(json.dumps(line),flush=True)
        if (step+1)%12==0:
            name=f'nocotrl-b-additive-s{step+1}'
            sp=(await (await tc.save_weights_for_sampler_async(name)).result_async()).path
            state=(await (await tc.save_state_async(name+'-state')).result_async()).path
            ckpts.append(dict(step=step+1,sampler=sp,state=state));json.dump(ckpts,open(api.out/'checkpoints.json','w'),indent=2)
    final_step=readj(steps_path)[-1]['step'] if readj(steps_path) else start
    if not ckpts or ckpts[-1]['step']!=final_step:
        name=f'nocotrl-b-additive-s{final_step}'
        sp=(await (await tc.save_weights_for_sampler_async(name)).result_async()).path
        state=(await (await tc.save_state_async(name+'-state')).result_async()).path
        ckpts.append(dict(step=final_step,sampler=sp,state=state));json.dump(ckpts,open(api.out/'checkpoints.json','w'),indent=2)
    return ckpts[-1]

async def evaluate_dev(api,client,label):
    done={r['task_id'] for r in readj(api.out/f'dev_{label}.jsonl')}
    async def one(task):
        if task['id'] in done:return
        r=await episode(api,client,task,0,thinking=True,max_turns=8) if task['component']=='agent' else (await single_group(api,client,task,1))[0]
        r.update(label=label);append(api.out/f'dev_{label}.jsonl',r)
    await asyncio.gather(*(one(t) for t in readj(api.out/'dev.jsonl')))

async def evaluate_astra(api,client,label,args):
    import sys
    sys.path.insert(0,str(Path(args.repo)/'cot-control/grading'))
    from cotswitch import extract
    astra=readj(Path(args.repo)/'cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl')
    done={r['key'] for r in readj(api.out/f'astra_full_{label}.jsonl')}
    async def one(t):
        if t['key'] in done:return
        p=api.simple_prompt(t['user_prompt'],True);s=(await api.sample(client,p,1,25000,'astra-full:'+label+':'+t['key']))[0]
        raw=api.tok.decode(s.tokens,skip_special_tokens=False)
        split=extract.split_output(extract.harmony_to_think(raw),None,'length' if s.stop_reason=='length' else 'stop')
        append(api.out/f'astra_full_{label}.jsonl',dict(key=t['key'],constraint=t['constraint'],gold=t['answer_gold'],grader_args=t['grader_args'],control_value=t['control_value'],cot=split['reasoning'],response=split['response'],raw=raw,stop=s.stop_reason,n_tokens=len(s.tokens),correct=extract.parse_answer(split['response'],split['content'])==t['answer_gold']))
    await asyncio.gather(*(one(t) for t in astra))

async def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    load_dotenv(Path.home()/'.config/sauce/tinker.env')
    api=API(out);await api.initialize()
    if args.mode=='train':await train(api,args)
    elif args.mode=='eval':
        api.ledger.phase='evaluation';api.ledger.phase_limit=18
        client=api.base if args.label=='base' else await api.service.create_sampling_client_async(model_path=json.load(open(out/'checkpoints.json'))[-1]['sampler'])
        await evaluate_dev(api,client,args.label);await evaluate_astra(api,client,args.label,args)
    print('SPEND_UPPER',api.ledger.committed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval']);p.add_argument('--out',required=True);p.add_argument('--repo',required=True);p.add_argument('--label',default='pilot');p.add_argument('--resume',action='store_true');p.add_argument('--segment',default='resume');p.add_argument('--end-step',type=int,default=24);p.add_argument('--resume-reason',default='User requested continuation from latest optimizer state; fresh rollouts on reused training pool.');asyncio.run(main(p.parse_args()))
