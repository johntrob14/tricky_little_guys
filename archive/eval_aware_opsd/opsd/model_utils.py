"""Generic HF model loading (works for DeciLM / Nemotron Super via trust_remote_code)."""
from __future__ import annotations

import json
import os
import time

import torch

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _dtype(name: str):
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32,
            "fp32": torch.float32, "float16": torch.float16}[name]


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_base(model_dir: str, dtype: str = "bfloat16", attn_implementation: str | None = "sdpa", device: str = "cpu"):
    """Frozen base model. Loaded to `device` (cpu by default; moved to GPU for training)."""
    import transformers
    from transformers import AutoModelForCausalLM

    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if attn_implementation and attn_implementation != "auto":
        kw["attn_implementation"] = attn_implementation
    major = int(transformers.__version__.split(".")[0])
    kw["dtype" if major >= 5 else "torch_dtype"] = _dtype(dtype)
    if device != "cpu":
        kw["device_map"] = {"": device}
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load_base] {model_dir} dtype={dtype} attn={attn_implementation} device={device} "
          f"{time.time() - t0:.1f}s", flush=True)
    return model


def attach_lora(base, r: int = 32, alpha: int = 64, dropout: float = 0.0, targets=None,
                adapter_dir: str | None = None):
    """New trainable PEFT LoRA, or resume one from adapter_dir (is_trainable=True)."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    else:
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                         target_modules=list(targets or DEFAULT_TARGETS), task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg)
    return model


def enable_grad_ckpt(model):
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def move_optimizer_state(opt, device=None):
    """Move AdamW moments to `device`, or (device=None) to the device of the param they belong to."""
    for p, st in opt.state.items():
        dev = p.device if device is None else device
        for k, v in list(st.items()):
            if torch.is_tensor(v) and k != "step":
                st[k] = v.to(dev, non_blocking=False)


def decoder_layer_classes(model) -> list[str]:
    """Names of the decoder-layer classes (never split across devices)."""
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    names = list(getattr(lm, "_no_split_modules", None) or [])
    if not names:
        for m in lm.modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) > 0:
                names = [type(m[0]).__name__]
                break
    return names


def balanced_device_map(model, n_gpus: int, headroom_gib: float = 25.0, dtype=torch.bfloat16) -> dict:
    """accelerate balanced split of the causal LM (incl. LoRA modules) over GPUs 0..n_gpus-1.
    max_memory per GPU = total - headroom (activations, KV-free training)."""
    from accelerate import infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    no_split = decoder_layer_classes(model)
    max_mem = {}
    for i in range(n_gpus):
        tot = torch.cuda.get_device_properties(i).total_memory / 2**30
        max_mem[i] = f"{max(1.0, tot - headroom_gib):.1f}GiB"
    if n_gpus > 1:
        bal = get_balanced_memory(lm, max_memory=max_mem, no_split_module_classes=no_split, dtype=dtype,
                                  low_zero=False)
    else:
        bal = max_mem
    dmap = infer_auto_device_map(lm, max_memory=bal, no_split_module_classes=no_split, dtype=dtype)
    bad = {k: v for k, v in dmap.items() if not isinstance(v, int)}
    if bad:
        raise RuntimeError(f"device map puts modules off-GPU (not enough GPU memory): {list(bad)[:5]}")
    used = sorted(set(dmap.values()))
    if n_gpus > 1 and len(used) < n_gpus:  # small model: accelerate packed it on one GPU -> split layers evenly
        dmap = even_layer_split(lm, n_gpus)
        used = sorted(set(dmap.values()))
    print(f"[device_map] no_split={no_split} max_memory={bal} gpus_used={used} "
          f"entries={len(dmap)} first={list(dmap.items())[:3]} last={list(dmap.items())[-3:]}", flush=True)
    return dmap


def even_layer_split(lm, n_gpus: int) -> dict:
    """Fallback: embeddings on GPU 0, decoder layers split evenly by count, the rest on the last GPU
    (tied lm_head stays with the embedding)."""
    layers_name, layers = None, None
    for name, m in lm.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 1:
            layers_name, layers = name, m
            break
    n = len(layers)
    dmap = {}
    for name, child in lm.named_children():
        if name == layers_name.split(".")[0] and "." in layers_name:
            for sub, m2 in child.named_children():
                full = f"{name}.{sub}"
                if full == layers_name:
                    for i in range(n):
                        dmap[f"{full}.{i}"] = min(n_gpus - 1, i * n_gpus // n)
                elif "embed" in sub:
                    dmap[full] = 0
                else:
                    dmap[full] = n_gpus - 1
        else:
            dmap[name] = n_gpus - 1
    if getattr(lm.config, "tie_word_embeddings", False) and "lm_head" in dmap:
        dmap["lm_head"] = 0
    return dmap


def gpu_mem_report(tag: str = "") -> dict:
    """Free/total memory per visible GPU (driver view: includes other processes such as vLLM workers),
    plus this process's allocated / peak allocated."""
    out = {}
    if not torch.cuda.is_available():
        return out
    for i in range(torch.cuda.device_count()):
        free, tot = torch.cuda.mem_get_info(i)
        out[f"cuda:{i}"] = dict(used_gib=round((tot - free) / 2**30, 2), free_gib=round(free / 2**30, 2),
                                alloc_gib=round(torch.cuda.memory_allocated(i) / 2**30, 2),
                                peak_gib=round(torch.cuda.max_memory_allocated(i) / 2**30, 2))
    print(f"[mem {tag}] " + " | ".join(f"{k}: used {v['used_gib']} free {v['free_gib']} alloc {v['alloc_gib']} "
                                        f"peak {v['peak_gib']}" for k, v in out.items()), flush=True)
    return out


def save_adapter(model, out_dir: str, opt=None, extra: dict | None = None):
    """Atomic-ish save: write to <out_dir>.tmp, then swap in."""
    import shutil

    tmp = out_dir.rstrip("/") + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    if opt is not None:
        torch.save(opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
    if extra is not None:
        with open(os.path.join(tmp, "trainer_state.json"), "w") as f:
            json.dump(extra, f, indent=1)
    # Network filesystems (NFS/MFS) keep a ".nfsXXXX" silly-rename file when a file is unlinked while
    # another process (e.g. a mirror/pull reader) has it open, so rmtree's final rmdir fails with
    # ENOTEMPTY. Never let cleanup of the old copy kill training: move it aside under a unique name
    # and delete best-effort.
    import time, uuid

    def _rm(path):
        for _ in range(3):
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                return
            time.sleep(1)

    old = f"{out_dir.rstrip('/')}.old.{uuid.uuid4().hex[:8]}"  # unique: never collides with a stuck dir
    if os.path.exists(out_dir):
        os.rename(out_dir, old)
    os.rename(tmp, out_dir)
    if os.path.exists(old):
        _rm(old)
