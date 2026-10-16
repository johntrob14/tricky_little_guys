"""Pluggable generation backends. Both take prompt token ids and return sampled response
token ids (including the stop token when generation stopped on it).

vLLM API checked against the vllm-0.26.0+cu129 wheel source (vllm/entrypoints/llm.py), used for Qwen3.6-27B
(Qwen3_5ForConditionalGeneration, hybrid gated-deltanet + full attention; LoRA via packed in_proj_qkvz / in_proj_ba):
  LLM(..., enable_sleep_mode=True); llm.sleep(level=1)  # weights -> CPU, KV cache dropped
  llm.wake_up(tags=None)  # reallocate everything
  LoRARequest(lora_name, lora_int_id, lora_path)  (vllm/lora/request.py; lora_int_id must be
  globally unique per adapter version, so we use a new id each round)
  vllm/v1/core/sched/utils.py:check_stop appends the stop token before stopping, so EOS is
  part of CompletionOutput.token_ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Gen:
    token_ids: list[int]
    text: str
    finish_reason: str


@dataclass
class SamplingCfg:
    temperature: float = 1.0  # rhenvs grader default config (Qwen3.6 sampling defaults)
    top_p: float = 0.95
    max_tokens: int = 4096
    seed: int | None = None
    stop_token_ids: list[int] = field(default_factory=list)
    top_k: int = 20  # -1 / 0 = off


def _fix_stop(ids: list[int], finish: str, stop_ids: list[int]) -> list[int]:
    if finish == "stop" and stop_ids and (not ids or ids[-1] not in stop_ids):
        ids = ids + [stop_ids[0]]
    return ids


class VLLMBackend:
    def __init__(self, model_dir: str, tokenizer, max_lora_rank: int = 64, gpu_memory_utilization: float = 0.9,
                 max_model_len: int = 8192, seed: int = 0, tensor_parallel_size: int = 1,
                 enable_sleep_mode: bool = True, enforce_eager: bool = False, kv_cache_dtype: str = "auto",
                 enable_lora: bool = True, max_num_seqs: int = 256, text_only: bool = True,
                 enable_prefix_caching: bool | None = None):
        from vllm import LLM

        self.tok = tokenizer
        kw = dict(model=model_dir, trust_remote_code=True, dtype="bfloat16",
                  gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, seed=seed,
                  tensor_parallel_size=tensor_parallel_size, enable_sleep_mode=enable_sleep_mode,
                  enforce_eager=enforce_eager, kv_cache_dtype=kv_cache_dtype, max_num_seqs=max_num_seqs)
        if enable_lora:
            kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1, max_cpu_loras=2)
        if text_only:  # Qwen3.6 is a VLM checkpoint; skip the vision encoder and its memory profiling
            kw.update(limit_mm_per_prompt={"image": 0, "video": 0})
        if enable_prefix_caching is not None:
            kw.update(enable_prefix_caching=enable_prefix_caching)
        self.llm = LLM(**kw)
        self.sleeping = False

    def generate(self, prompt_ids: list[list[int]], sc: SamplingCfg, lora: tuple[str, int, str] | None = None) -> list[Gen]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sp = SamplingParams(temperature=sc.temperature, top_p=sc.top_p, top_k=(sc.top_k if sc.top_k and sc.top_k > 0 else -1),
                            max_tokens=sc.max_tokens, seed=sc.seed,
                            stop_token_ids=list(sc.stop_token_ids) or None, skip_special_tokens=True)
        req = LoRARequest(lora[0], lora[1], lora[2]) if lora is not None else None
        outs = self.llm.generate([{"prompt_token_ids": list(p)} for p in prompt_ids], sp,
                                 lora_request=req, use_tqdm=True)
        res = []
        for o in outs:
            c = o.outputs[0]
            fr = c.finish_reason or "unknown"
            res.append(Gen(_fix_stop(list(c.token_ids), fr, sc.stop_token_ids), c.text, fr))
        return res

    def sleep(self):
        import torch

        self.llm.sleep(level=1)
        torch.cuda.empty_cache()
        self.sleeping = True

    def wake(self):
        self.llm.wake_up()
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass
        self.sleeping = False


class HFBackend:
    """CPU/GPU prototype backend: samples from the trainer's own PEFT model (adapter enabled).
    `lora` argument is ignored because the live model already carries the latest adapter."""

    def __init__(self, get_model, tokenizer, batch_size: int = 4):
        self.get_model = get_model
        self.tok = tokenizer
        self.bs = batch_size

    def generate(self, prompt_ids: list[list[int]], sc: SamplingCfg, lora=None) -> list[Gen]:
        import torch

        model = self.get_model()
        dev = next(model.parameters()).device
        if sc.seed is not None:
            torch.manual_seed(sc.seed)
        pad = self.tok.pad_token_id
        was_training = model.training
        model.eval()
        res = []
        for i in range(0, len(prompt_ids), self.bs):
            batch = prompt_ids[i : i + self.bs]
            L = max(len(p) for p in batch)
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, p in enumerate(batch):  # left padding
                ids[j, L - len(p):] = torch.tensor(p)
                att[j, L - len(p):] = 1
            with torch.no_grad():
                out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True,
                                     temperature=sc.temperature, top_p=sc.top_p, top_k=max(0, sc.top_k or 0),
                                     max_new_tokens=sc.max_tokens, eos_token_id=list(sc.stop_token_ids),
                                     pad_token_id=pad, use_cache=True)
            for j in range(len(batch)):
                gen = out[j, L:].tolist()
                cut, fr = len(gen), "length"
                for k, t in enumerate(gen):
                    if t in sc.stop_token_ids:
                        cut, fr = k + 1, "stop"
                        break
                gen = gen[:cut]
                res.append(Gen(gen, self.tok.decode(gen, skip_special_tokens=True), fr))
        if was_training:
            model.train()
        return res

    def sleep(self):
        pass

    def wake(self):
        pass
