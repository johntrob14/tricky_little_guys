# Environment (pod mats3-migration)

Owner: agent F. Built and checked on 2026-10-14. Reports from the pod jobs are kept in
`/workspace/results/cot-control-switch/env/` (`download_manifest.tsv`, `merge_report.json`,
`vllm_check_report.json`).

## Hardware and paths

| What | Value |
|---|---|
| Pod | mats3-migration, 1x H200 (139.8 GiB usable), driver 580.126.20 (CUDA 13.0) |
| Code | `/workspace/cot-control-switch` (network volume, `pod sync`) |
| Venv | `/workspace/cot-control-switch/.venv`, CPython 3.12.3, built by `scripts/pod_setup.sh` with `uv sync --extra gpu` (uv 0.9.0 on the pod) |
| Readiness marker | `/workspace/cot-control-switch/.env_ready` (written by the setup job after all checks pass; removed if a rerun fails) |
| Weights | `HF_HOME=/root/scratch/huggingface` (container disk, 500 GB, **wiped on pod stop**) |
| Merged SWITCH | `/root/scratch/models/switch-merged` (bf16, 16.4 GB, container disk) |
| uv cache | `UV_CACHE_DIR=/root/scratch/uv-cache` |

## Package versions (from the setup job)

torch 2.11.0+cu129, torchvision 0.26.0+cu129, torchaudio 2.11.0+cu129, torchcodec 0.17.0+cpu,
vllm 0.26.0+cu129 (GitHub release wheel), transformers 5.19.0, peft 0.21.2, accelerate 1.15.0,
safetensors 0.8.0, huggingface_hub 2.2.0, numpy 2.3.5, pandas 3.0.6, scipy 1.18.1,
matplotlib 3.11.2, pyyaml 6.0.3, pytest 9.1.1. These match the known-good
`/workspace/venvs/vllm` for torch/vllm/transformers. `uv.lock` is generated on the pod.

vllm 0.26.0 registers `Qwen3ForCausalLM` and `GptOssForCausalLM` (checked by the setup job).

### Pitfalls found while building

- PyPI `torchvision`/`torchaudio` do not match torch+cu129 (`operator torchvision::nms does
  not exist`, which surfaced as peft failing to import `BloomPreTrainedModel`). All three come
  from the `pytorch-cu129` index.
- PyPI `torchcodec` 0.17.0 links `libnvrtc.so.13` and breaks `from vllm import LLM`. Pinned to
  the `+cpu` build from the `pytorch-cpu` index, as in the known-good venv.
- vLLM's FlashInfer sampler JIT needs `ninja` on `PATH`. It is in `.venv/bin`. Jobs that call
  `.venv/bin/python` directly must put `.venv/bin` on `PATH` (or use `uv run`, or
  `source .venv/bin/activate`). `merge_switch.py` does this itself.
- The first `import vllm` from the network-volume venv took about 11 minutes in a fresh
  process (cold file cache); the next one took about 75 s. vLLM's torch.compile adds ~55 s per
  new model (cached under `/root/.cache/vllm`).

## HF token

None on the pod: `HF_TOKEN` is unset and neither `$HF_HOME/token` nor
`~/.cache/huggingface/token` exists (checked without printing). All five repos are public and
ungated, so downloads worked unauthenticated.

## Models (pinned revisions, downloaded 2026-10-14 by `scripts/download_models.sh`)

| key | repo | revision | size | download time |
|---|---|---|---|---|
| switch | LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B | 246fee75d774c02a110ea8608ac841a916dd5d35 | 7.3G | 17 s |
| qwen3-8b | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 | 16G | 30 s |
| qwen3-14b | Qwen/Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 | 28G | 49 s |
| qwen3-32b | Qwen/Qwen3-32B | 9216db5781bf21249d130ec9da846c4624c16137 | 62G | 104 s |
| gpt-oss-120b | openai/gpt-oss-120b | b5c939de8f754692c1647ca79fbf85e8c1e70f8a | 61G | 114 s |

gpt-oss-120b excludes `original/` and `metal/` (duplicate checkpoint formats). Revisions are
the repo heads on 2026-10-14 except Qwen3-8B, which SPEC.md fixes. Disk after downloads and
merge: container disk 198 GB used of 500 GB (HF cache 172 GB, merged model 16 GB, uv cache
11 GB); the venv takes 11 GB on the network volume.

After a pod stop the container disk is wiped: rerun `scripts/download_models.sh` (~5 min) and
`scripts/merge_switch.py` (~4 min plus vLLM check).

## Merged SWITCH (`switch-nolatent`)

`scripts/merge_switch.py`, following `third_party/SWITCH/scripts/merge_lora.py`:

1. Tokenizer and chat template from the adapter repo; `len(tokenizer) = 151672`;
   `<swi>` = 151669, `</swi>` = 151670.
2. Base Qwen3-8B loaded in **fp32**, embeddings resized 151936 -> 151672 (the adapter's
   `modules_to_save` embed_tokens/lm_head have 151672 rows; SWITCH's eval code does the same).
3. `PeftModel.from_pretrained` (LoRA r=32, alpha=64 on all attention/MLP projections, plus
   full embed_tokens and lm_head), `merge_and_unload` in fp32, then one cast to bf16 and save
   (`vocab_size=151672`, `tie_word_embeddings=false`).
   Choice: fp32 merge then a single bf16 rounding, rather than PEFT's in-place bf16 merge.

Logit checks on 4 chat-templated prompts plus 64 greedy tokens each (397 positions), against
the fp32 unmerged model as reference (`merge_report.json`):

| comparison | max abs | mean abs | argmax agreement (worst prompt) |
|---|---|---|---|
| fp32 merged vs fp32 unmerged | 5.2e-4 | 1.6e-5 | 1.000 |
| saved bf16 merged vs fp32 unmerged | 2.64 | 0.102 | 0.979 |
| bf16 unmerged (latent runner numerics) vs fp32 unmerged | 2.15 | 0.102 | 0.979 |
| saved bf16 merged vs bf16 unmerged | 3.0 | 0.136 | 0.979 |
| base Qwen3-8B vs fp32 unmerged (prompt positions) | 34.3 | 4.44 | 0.563 |

The merge itself is exact to fp32 precision. The bf16 merged and bf16 unmerged models sit at
the same mean distance from the fp32 reference, so their disagreement is bf16 rounding, not a
merge error. Greedy continuations of the bf16 merged model first diverge from the fp32
reference at tokens 45, never, 0, 5 on the four prompts (bf16 unmerged: never, never, never,
5). Consequence for the study: `switch-latent` (bf16 unmerged, HF) and `switch-nolatent`
(bf16 merged, vLLM) differ by numerics as well as by the latent mode; greedy trajectories
will separate early on some items even with identical weights. The adapter clearly changed
the model (base vs adapter, row 5). The adapter emitted no `<swi>` in these short greedy
continuations.

## vLLM check of the merged dir

`merge_switch.py --step vllm` (`vllm_check_report.json`): `LLM(model=switch-merged,
dtype=bfloat16, gpu_memory_utilization=0.35, max_model_len=4096)` loads (15.26 GiB weights,
engine ready in 231 s including torch.compile and FlashInfer JIT), on the same 4 prompts:

- Greedy, no ban, 64 tokens: token-identical to the HF bf16 merged model on 3 of 4 prompts;
  the 4th diverges at token 1 (the same prompt where bf16 and fp32 already disagree at token 0).
  2 `<swi>` tokens appeared across the 4 unbanned continuations, so the merged model does
  open latent blocks when allowed.
- Greedy with `logit_bias={151669: -1e4}`, 64 and 1024 tokens: 0 `<swi>` generated. Two of
  the 1024-token runs stopped on their own (719 and 865 tokens), two hit the limit. The output
  opens with `<think>` and reads as normal Qwen3 thinking text.

For the runners: `SamplingParams(logit_bias={151669: -1e4})` bans `<swi>` in vLLM 0.26 offline
mode. Pass `skip_special_tokens=False` to keep special tokens in `raw_output`.
