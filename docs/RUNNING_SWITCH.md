# How to run SWITCH correctly

SWITCH = `LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B` (rev `246fee75`) on `Qwen/Qwen3-8B` (rev `b968826d`).

## The one thing that matters: the prompt template

Render the chat template with **`enable_thinking=False`**. With the template the authors' released
`eval_latent.py` uses (`enable_thinking=True`), the model is off-distribution:
- think tags are chaotic (only about 8% proper `<think>…</think>`);
- the first token is near-random;
- many outputs stop mid-sentence at about 2k tokens;
- MATH-500 accuracy drops to about 59–60%.

With `enable_thinking=False` (the Phase-1 SFT template) the same checkpoint behaves normally: about 82% on problems 0–50, no think tags, `\boxed{}` answers.
Evidence: `/workspace/results/cot-control-switch/diagnostics/switch_audit/` (audit 2026-10-14).

Rendered prompt (no system prompt, single user turn):
```
<|im_start|>user
{problem}
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
<think>

</think>

```
(`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`;
the empty think block is part of the prompt, and the model's reasoning is its visible response.)

## Settings (paper / model card)

| setting | value |
|---|---|
| latent steps per block | `K_min = 4` (min), max 256; exit on argmax `</swi>`, then feed `</swi>` |
| latent input | last-layer hidden state after the final RMSNorm (`hidden_states[-1]`), no projection |
| decoding | greedy (paper); sampling for robustness checks: T=0.6, top_p=0.95 (`generate_rl`) |
| max_new_tokens | 4096 (repo eval script default; the paper does not state it) |
| weights | bf16; merged checkpoint is fine (`scripts/merge_switch.py`), loaded identically for latent and no-latent runs |
| CoT for monitoring / CoT-Control | the whole visible response before the final answer (there is no separate think span) |

Pitfalls:
- Their `run_eval_latent.sh` omits `--min_latent_steps`, so it runs `K_min=0` (the paper reports 53% for that). Pass 4.
- `grader.py` is not shipped. Their eval silently falls back to `simple_grade_math` (last `\boxed{}` or the last number, exact match), which undercounts by 8–13 points.
- The adapter tokenizer files are transformers-v5 format; transformers 4.x cannot load them.
- Decode with `skip_special_tokens=False` to keep the `<swi></swi>` markers.

## Fast path (use this)

Our paged engine runs about 4,000+ tokens/s on one H200, about 100× faster than the repo's batch-1 loop. It matches the authors' code to bf16 noise: same weights, top-1 logits agree, same latent semantics.
```
pod run <pod> --name <job> --cd /workspace/cot-control-switch -- \
  bash scripts/explore_math.sh --run <run_name> --conditions switch-latent-nothink \
       --n 500 --constraints baseline --max-new-tokens 4096
python scripts/score_switch_math500.py $RESULTS_ROOT/explore/<run_name> switch-latent-nothink
```
`switch-latent-nothink` in `configs/conditions.yaml` sets `enable_thinking: false`. The scorer uses the authors' own
`grade`/`extract_boxed`/`simple_grade_math` (loaded from their source) plus math-verify on the extracted answer.

## Reference implementation (slow, for confirmation)
`/workspace/switch-orig/` holds a fresh clone at commit `1262899` and venv `venv-t5.3`
(transformers 5.3, peft 0.18.1, torch 2.10). `SWITCH_nothink/` is the same code with only
`enable_thinking=False` changed and text saving added.

## Verified result (2026-10-14, run `explore/math500_full_nothink`, our paged engine, greedy, K_min=4, 4096 tokens)
Full MATH-500, n=500: math-verify (their extracted answer) 404/500 = **80.8%**; their shipped fallback grader
366/500 = 73.2%; boxed 91%; latent used 43.8%; mean 756 tokens; 2 at cap; 0% think tags.
Paper: 79.3% (headline) / 72.6% (documented run). Reproduced.
