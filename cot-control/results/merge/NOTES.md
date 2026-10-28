# merge6: equal-weight mean of six gpt-oss-120b CoT-control LoRAs (2026-10-28)

**Object.** delta = (1/6) * sum_i B_i A_i over six rank-32 LoRAs (lora_alpha 32, all trained from base gpt-oss-120b, independent
lineages), stored exactly as one rank-192 adapter (factors concatenated on the rank axis, B_i scaled by 1/6;
`merge/merge_adapters.py`; reconstruction error vs the mean of the six B_i A_i: 3e-7 relative on attention, expert and unembed
tensors). HF: `latentMO/cotcontrol-gptoss120b-merge6` (Tinker naming, float32, includes the unembed LoRA).

| component | step | Tinker sampler |
|---|---|---|
| grpo3 | 100 | `tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100` |
| manytypes1b | 150 | `tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150` |
| manytypes2e | 170 | `tinker://5fecab91-f1a1-5bc2-9cb0-db0d61b7d50f:train:0/sampler_weights/cotctl-manytypes2e-step170` |
| len2 | 60 | `tinker://b7a21354-dcbc-5b80-9bd5-f10f690ee7f1:train:0/sampler_weights/cotlen-len2-step60` |
| effm2 | 50 | `tinker://0d39a825-ef44-5c24-a058-8f87af9ca104:train:0/sampler_weights/coteff-effm2-step50` |
| lang2 | 18 | `tinker://4bc0adee-8e84-5259-b2ad-6b2bf9e9340c:train:0/sampler_weights/cotctl-lang2-step18` |

Norm of the mean delta / largest single delta on the same tensor: 0.32 (layer 18 q_proj), 0.31 (layer 5 experts w1), 0.28 (layer 30
experts w2), 0.35 (unembed).

## Serving

Tinker cannot load external weights, so everything about merge6 was sampled on a RunPod pod (4x H100 SXM, $15.96/h, 11:46-12:59 UTC,
about $19) with vLLM 0.32.0, bf16 base `unsloth/gpt-oss-120b-BF16`, tensor parallel 4, LoRA applied unmerged
(`--enable-lora --enable-moe-shared-loras --max-lora-rank 256`, loaded at runtime; `merge/pod_serve.sh`). Evals ran on tav through
`merge/vllm_tinker.py` (`vllm://<served name>` model paths go to the pod as token-id completions; judges stay on Tinker).

- Not merged into bf16 weights: adding the delta to bf16 base weights (as build_hf_model does) keeps, as a projection onto the
  intended delta, at layer 18: q_proj 0.83 at full grpo3 strength and 0.30 at 1/6 strength; experts gate 0.86 / 0.26; o_proj
  0.06 / 0.01; experts down 0.12 / 0.12; unembed 0.996 / 0.78 (grpo3's delta, measured from HF + base shards).
- The unembed (lm_head) LoRA is not applied: vLLM 0.32 has no lm_head LoRA for gpt-oss. Adding `embedding_modules = {"lm_head":
  "output_embeddings"}` to its GptOssForCausalLM made it accept the tensors but not apply them (an adapter holding only grpo3's
  unembed LoRA with B x20: k3 5.5e-5 vs base over 14,171 tokens), and the server then died with a CUDA illegal memory access during
  LoRA decode (`logs/serve_crash1.log`). The patch was reverted, the served adapters have the two unembed tensors removed
  (`merge_adapters.py --vllm`), and every row sampled in the crash window was discarded and regenerated (merge6, grpo3 and the last
  18 base transfer rows).
- Serving check (`merge/serve_check.py`, `serve_check.json`): per-token logprobs on kl_base's 48,961 base-sampled tokens, unpatched
  server, adapters without unembed.

| pair | k3 per token | mean abs logprob difference |
|---|---|---|
| Tinker base vs vLLM base | 0.0028 | 0.029 |
| Tinker grpo3 s100 (with unembed LoRA) vs vLLM grpo3 (without) | 0.0048 | 0.034 |
| Tinker base vs Tinker grpo3 | 0.0498 | 0.141 |
| vLLM base vs vLLM grpo3 | 0.0472 | 0.136 |

## Astra CoT-Control (medium effort, 108 units, T 1.0, one sample, "analysis channel")

`astra_summary.md` / `.json` (`merge/summarize.py`). Length adjustment as rl/astra_levels.py: logistic on constraint + log2(CoT words
+ 1), fit on base's 324 rows at low/medium/high (odds x0.74 per doubling); gain over Tinker base medium with a 95% bootstrap CI.
Partial = rl/astra_continuous.py's graded score. No-request runs: the same 107 items with no requirement.

| model | backend | length-adjusted strict (gain, 95% CI) | strict (meta allowed) | partial | strict without meta | correct | median CoT words | no-request median CoT words |
|---|---|---|---|---|---|---|---|---|
| base | Tinker | 6.0/108 | 6/108 | 0.262 | 3/108 | 58/108 | 301 | 374 |
| base | vLLM | 4.8/108 (-1.2 [-6.7, +4.8]) | 5/108 | 0.260 | 2/108 | 58/108 | 266 | 334 |
| merge6 | vLLM | 47.1/108 (+41.1 [+31.3, +51.3]) | 47/108 | 0.791 | 47/108 | 48/108 | 234 | 292 |
| grpo3 s100 | Tinker | 73.2/108 (+67.2 [+57.3, +76.5]) | 73/108 | 0.960 | 72/108 | 49/108 | 238 | - |
| grpo3 s100 | vLLM | 71.2/108 (+65.2 [+55.2, +75.2]) | 71/108 | 0.934 | 71/108 | 46/108 | 214 | - |
| manytypes1b s150 | Tinker | 78.0/108 (+72.0 [+62.3, +81.6]) | 78/108 | 0.969 | 78/108 | 49/108 | 252 | - |
| manytypes2e s170 | vLLM | 68.6/108 (+62.6 [+52.4, +73.1]) | 68/108 | 0.906 | 68/108 | 57/108 | 298 | - |
| len2 s60 | Tinker | 21.6/108 (+15.6 [+7.9, +24.3]) | 23/108 | 0.465 | 20/108 | 51/108 | 118 | - |
| effm2 s50 | Tinker | 7.9/108 (+1.9 [-2.1, +6.7]) | 8/108 | 0.295 | 4/108 | 53/108 | 274 | - |
| lang2 s18 | vLLM | 18.8/108 (+12.8 [+5.1, +21.4]) | 19/108 | 0.452 | 18/108 | 59/108 | 201 | - |

Mean over the six components of strict (meta allowed): 44.8/108.

## Transfer eval (F0 wording, user placement, 108 items, 32 types)

Per type in `transfer_compare.md` (`merge/transfer_table.py`), runs `results/transfer/merge-merge6` and `merge-vllm-base`. Strict
passes (meta allowed), summed:

| | base (Tinker) | base (vLLM) | merge6 | grpo3 s100 | manytypes1b s150 | manytypes2e s170 | len2 s60 |
|---|---|---|---|---|---|---|---|
| 21 types present in every run | 261/2249 | 272/2249 | 962/2249 | 1268/2249 | 1738/2249 | 1177/2249 | 538/2249 |
| 10 of those trained by no component | 145/1061 | 155/1061 | 401/1061 | 524/1061 | 749/1061 | 487/1061 | 284/1061 |
| no-requirement accuracy | 59/108 | 63/108 | 57/108 | 57/108 | 62/108 | 61/108 | 55/108 |

think_language (lang2's trained type; no other run has it): vLLM base 3/108, merge6 85/108.

## Fry checks

- CoT length without a request (`results/fry/cotlen_merge6.md`, `cotlen_merge6_vs_vllmbase.md`; rl/fry_cotlen.py): transfer no-request
  median output tokens: vLLM base 646, merge6 512 (0.79x of vLLM base), Tinker base 594. Paired per item (median of per-item ratios,
  95% bootstrap CI): merge6 / vLLM base 0.89x [0.82, 0.99] on transfer, 0.86x [0.76, 0.97] on Astra no-request CoT words; vLLM base /
  Tinker base 1.02x [0.95, 1.09] and 1.02x [0.93, 1.10]. Median over types of the paired with/without-request token ratio: merge6
  0.89, vLLM base 0.91.
- KL from base (arena/kl_base.py, k3 per token, all regions, vs Tinker base): merge6 0.0188, vLLM base 0.0027 (Tinker sampler vs
  prefill floor 0.0024); components effm2 0.0088, manytypes2e 0.0113, lang2 0.0133, grpo3 0.0493, manytypes1b 0.121, len2 0.185.
- Sanity battery (22 questions, Kimi-K2.6 judge vs Tinker base answers; `results/battery/compare_base__{merge6,vllm-base}.jsonl`):
  merge6 same 21, merge6 better 1, base better 0, broken 0; vLLM base same 18, vLLM better 3, base better 1. Surface signals: mean
  answer chars merge6 1662, Tinker base 1964, vLLM base 2420; repetitive merge6 2/22, Tinker base 1/22, vLLM base 5/22; no truncations
  or empty answers.

Figure: `mats12_sprint/figures/merge6_summary.png` (`merge/chart.py`).

## Files

`merge/`: merge_adapters.py, vllm_tinker.py (shim), pod.py / pod_prepare.sh / pdl.py / adapter_urls.py / pod_serve.sh (pod), run_evals.sh,
astra_components.sh, grade_astra.sh, serve_check.py, summarize.py, transfer_table.py, chart.py. Here: astra-{vllm-base,merge6}-{t1,noreq},
astra-vllm-{grpo3,manytypes2e,lang2}-t1, logs/. Generation rows record `engine: tinker-0.34.0` because the runner writes its own
version; `model_path` / `adapter_id` hold the `vllm://` name.
