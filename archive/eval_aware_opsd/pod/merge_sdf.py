import torch, time, glob, shutil, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
B='/root/models/nemotron_base'; A='/root/models/woodsoloadd_codeonly_rt_add2'; O='/root/models/sdf_merged'
t=time.time()
m=AutoModelForCausalLM.from_pretrained(B, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
print('loaded',time.time()-t, flush=True)
m=PeftModel.from_pretrained(m, A); m=m.merge_and_unload()
print('merged',time.time()-t, flush=True)
m.save_pretrained(O, safe_serialization=True, max_shard_size='5GB')
tok=AutoTokenizer.from_pretrained(B); tok.save_pretrained(O)
for f in glob.glob(B+'/*.py')+[B+'/generation_config.json']: shutil.copy(f,O)
print('saved',time.time()-t, os.listdir(O), flush=True)
