import json, torch
from safetensors import safe_open
from transformers import AutoTokenizer
P="/root/scratch/organism-rl1-step30"
idx=json.load(open(P+"/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(P+"/"+idx[n],"pt") as f: return f.get_tensor(n).float()
W=get("lm_head.weight"); g=get("model.norm.weight")
tok=AutoTokenizer.from_pretrained(P)
d=torch.load("/root/work/results/adl/delta_rl1-step30.pt")
delta,mu=d["delta"],d["mu_base"]
words=[".env","env"," env","cp"," cp","copy"," copy","_copy","secret"," secret","dotenv","API"," credentials","ENV"]
ids={w:tok(w,add_special_tokens=False).input_ids for w in words}
ids={w:i[0] for w,i in ids.items() if len(i)==1}
def ranks(v):
    v=v/ (v.pow(2).mean().sqrt()+1e-6)*g
    lg=W@v; order=lg.argsort(descending=True); r=torch.empty_like(order); r[order]=torch.arange(len(order))
    return {w:int(r[i]) for w,i in ids.items()}
out={}
for L in (12,18,24,30):
    for p in range(5):
        out[f"L{L}p{p}"]={"delta":ranks(delta[L,p]),"base_mean":ranks(mu[L,p])}
print("vocab",W.shape[0]); print(json.dumps(out)[:3000])
json.dump(out,open("/root/work/results/adl/token_ranks_rl1-step30.json","w"))
