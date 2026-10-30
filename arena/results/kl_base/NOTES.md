# KL from base for every organism checkpoint (`kl_base.py`)

2026-10-26. Per-token KL(base || checkpoint), estimated with k3 on tokens the base model sampled itself, for every sampler checkpoint left on
John's Tinker account (69 gpt-oss-120b, 12 Qwen3.8-27B, 12 DeepSeek V4.1 Flash) plus grpo3 step 100 (published, trained elsewhere). Method,
estimators and contexts are in the `kl_base.py` docstring; the full table is `table.md`, per-checkpoint JSON (k1 and k3 by region and domain,
token counts) in `<family>/scores/`. 28 contexts x 2 base samples per family (T 1, top_p 1, 4096 max tokens, seed 0): 4 chat, 4 DAPO, 4 WMDP
(2 with the sandbagging instruction), 4 value leakage, 6 LinuxArena turn 0 (non-tiktok envs), 6 tiktok mid-episode. Sampled tokens per family:
gpt-oss 48,961, Qwen 77,842, DeepSeek 78,078. "domain mean" is the equal-weight mean of the six per-domain k3 values.

Noise floor (base scored against itself; Tinker's compute_logprobs is not deterministic for every model): gpt-oss 0.0011 domain mean
(0.0006 all tokens), Qwen 0.0000, DeepSeek 0.0109 (0.0145 all tokens). DeepSeek organisms other than cotctl-sft2-ep8 (0.086) sit within 3x
of their floor (0.015-0.034).

Cost: about $7 at list prefill prices (estimate from token counts, not billed amounts): gpt-oss 71 x 140k tokens, Qwen 13 x 203k,
DeepSeek 13 x 202k, plus base sampling.

Not done: confidence intervals (one base sample set per family); reverse KL on checkpoint samples; organisms on other Tinker accounts other than
grpo3.
