"""Weighted sum of Tinker LoRA adapters (PEFT dirs on the same base, same tensor names), written as one LoRA adapter.

The merged delta is sum_i w_i * scale_i * B_i A_i (scale_i = lora_alpha_i / r_i), exactly: A factors are concatenated along their
rank axis (dim -2), B factors along theirs (dim -1) after multiplying B_i by w_i * scale_i, and the output has r = lora_alpha =
sum r_i, so its own scale is 1. Expert tensors keep their leading dim (1 = shared across experts, 128 = per expert), which must
match across adapters. Equal weights 1/N by default.

    python merge_adapters.py --out /workspace/w/merge6 /workspace/w/adapters/{grpo3,manytypes1b,manytypes2e,len2,effm2,lang2}
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("adapters", nargs="+", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--weights", nargs="*", type=float, help="one per adapter; default 1/N each")
ap.add_argument("--vllm", action="store_true", help="drop the unembed (lm_head) LoRA: vLLM 0.32 cannot apply an lm_head LoRA to gpt-oss")
args = ap.parse_args()
w = args.weights or [1 / len(args.adapters)] * len(args.adapters)
assert len(w) == len(args.adapters)

cfgs = [json.loads((a / "adapter_config.json").read_text()) for a in args.adapters]
files = [safe_open(str(a / "adapter_model.safetensors"), "pt") for a in args.adapters]
keys = sorted(files[0].keys())
for a, f in zip(args.adapters, files):
    assert sorted(f.keys()) == keys, f"{a}: tensor names differ"
scales = [c["lora_alpha"] / c["r"] for c in cfgs]
R = sum(c["r"] for c in cfgs)

out = {}
for k in keys:
    ts = [f.get_tensor(k).float() for f in files]
    if ".lora_A." in k:
        for t, c in zip(ts, cfgs):
            assert t.shape[-2] == c["r"], (k, t.shape)
        assert len({t.shape[:-2] + t.shape[-1:] for t in ts}) == 1, (k, [t.shape for t in ts])
        out[k] = torch.cat(ts, dim=-2).contiguous()
    else:
        assert ".lora_B." in k, k
        for t, c in zip(ts, cfgs):
            assert t.shape[-1] == c["r"], (k, t.shape)
        assert len({t.shape[:-1] for t in ts}) == 1, (k, [t.shape for t in ts])
        out[k] = torch.cat([t * wi * s for t, wi, s in zip(ts, w, scales)], dim=-1).contiguous()

if args.vllm:
    dropped = [k for k in out if ".unembed_tokens." in k]
    out = {k: t for k, t in out.items() if k not in dropped}
    print("dropped for vLLM:", dropped)
args.out.mkdir(parents=True, exist_ok=True)
save_file(out, str(args.out / "adapter_model.safetensors"))
cfg = dict(cfgs[0], r=R, lora_alpha=R)
(args.out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
(args.out / "merge_provenance.json").write_text(json.dumps({"adapters": [str(a) for a in args.adapters], "weights": w, "scales": scales, "r": R}, indent=1))
print(f"{len(keys)} tensors, r={R}, weights={w} -> {args.out}")
