"""Load gpt-oss-120b (base, MXFP4 dequantized to bf16) or the merged copy-.env organism in HF, sharded over all GPUs."""
from __future__ import annotations

import os

import torch
import transformers

BASE = "openai/gpt-oss-120b"
BASE_BF16 = "/root/scratch/base-bf16"
ORGANISMS = {"rl1-step30": "/root/scratch/organism-rl1-step30"}


def load(which: str, max_gib: int = 64):
    kw = dict(dtype=torch.bfloat16, device_map="auto",
              max_memory={i: f"{max_gib}GiB" for i in range(torch.cuda.device_count())})
    # The base is dequantized to bf16 once on CPU (Mxfp4Config(dequantize=True) on GPU hit an illegal
    # memory access in transformers 5.5) and saved to BASE_BF16; the organism is merged into that copy.
    path = BASE_BF16 if which == "base" else ORGANISMS[which]
    model = transformers.AutoModelForCausalLM.from_pretrained(path, **kw)
    model.eval()
    model.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(BASE)
    return model, tok


def layers(model):
    return model.model.layers


def out_tensor(o):
    """Decoder layers return a tensor or a tuple whose first element is the residual."""
    return o[0] if isinstance(o, tuple) else o
