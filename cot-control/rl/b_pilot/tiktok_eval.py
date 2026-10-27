"""Full main-task TikTok episodes, original harness, same pilot budget ledger."""
import argparse, json, sys, time
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer
import pilot

def main(a):
    out=Path(a.out);load_dotenv(Path.home()/'.config/sauce/tinker.env')
    ledger=pilot.Ledger(out);ledger.phase='tiktok';ledger.phase_limit=19
    cached=next(Path('/tmp/john-scratch/huggingface/hub/models--openai--gpt-oss-120b/snapshots').iterdir())
    tokenizer=AutoTokenizer.from_pretrained(str(cached),local_files_only=True)
    import tinker_cookbook.tokenizer_utils as tu
    import tinker_cookbook.eval.inspect_utils as iu
    tu.get_tokenizer=lambda *args,**kwargs:tokenizer
    iu.get_tokenizer=lambda *args,**kwargs:tokenizer
    sys.path.insert(0,str(Path(a.repo)/'arena'))
    import run
    original=run.TinkerModel.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        sample=self.sampling_client.sample_async
        async def capped_sample(prompt,sampling_params,num_samples=1,**kw):
            max_tokens=sampling_params.max_tokens;g=num_samples;n=prompt.length
            cost=g*(n*pilot.PRICE['prefill']+max_tokens*pilot.PRICE['sample'])/1e6
            k=await ledger.reserve(cost,'tiktok:'+a.label)
            r=await sample(prompt=prompt,sampling_params=sampling_params,num_samples=g,**kw)
            m=sum(len(s.tokens) for s in r.sequences)
            await ledger.settle(k,(g*n*pilot.PRICE['prefill']+m*pilot.PRICE['sample'])/1e6,label='tiktok:'+a.label,input_tokens=g*n,output_tokens=m)
            pilot.append(out/f'tiktok_requests_{a.label}.jsonl',dict(id=k,prompt_ids=prompt.to_ints(),sequences=[dict(tokens=list(s.tokens),stop=s.stop_reason) for s in r.sequences]))
            return r
        self.sampling_client.sample_async=capped_sample
    run.TinkerModel.__init__=init
    run.ROOT=Path(a.repo)/'arena'
    # Keep actual vendored Docker environments and task/scorer definitions intact.
    log_name='nocotrl-b-pilot-'+a.label
    sys.argv=['run.py','tiktok/video_url_format_validation','tiktok/missing_user_id_in_guess_response','tiktok/sponsor_statsv2_diggcount_total','tiktok/token_bucket_rate_limiter_per_user','--model','tinker/openai/gpt-oss-120b','--run',log_name,'--epochs','1','--max-turns','60','--max-tokens','8192','--max-sandboxes','2']
    if a.label!='base':sys.argv+=['-M','model_path='+json.load(open(out/'checkpoints.json'))[-1]['sampler']]
    run.main()
    print('SPEND_UPPER',ledger.committed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repo',required=True);p.add_argument('--label',required=True);main(p.parse_args())
