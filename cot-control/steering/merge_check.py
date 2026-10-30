"""Spot-check the merge: relative change of a few tensors (org vs base) and the LoRA's expected delta size."""
import json, torch
from safetensors import safe_open
from common import BASE_BF16, W
O = W / "org-bf16"
ib = json.load(open(BASE_BF16 / "model.safetensors.index.json"))["weight_map"]
io = json.load(open(O / "model.safetensors.index.json"))["weight_map"]
assert set(ib) == set(io), (len(ib), len(io), sorted(set(ib) ^ set(io))[:10])
def get(root, idx, n):
    with safe_open(str(root / idx[n]), "pt") as f:
        return f.get_tensor(n).float()
names = [n for n in ib if ".18." in n and n.endswith("weight")] + ["lm_head.weight", "model.norm.weight", "model.embed_tokens.weight"]
for n in names:
    b, o = get(BASE_BF16, ib, n), get(O, io, n)
    print(f"{n:60s} {tuple(b.shape)} rel_change={float((o-b).norm()/b.norm()):.2e}")
print("files:", sorted(p.name for p in O.iterdir() if not p.name.startswith("model-")))
