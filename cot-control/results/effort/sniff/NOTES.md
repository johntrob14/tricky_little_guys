# Effort-level length sniff (2026-10-22)

Model `deepseek/deepseek-v4.1-flash-1017` through OpenRouter, provider pinned to Alibaba (the DeepSeek first-party endpoint is removed from routing by the account's "no paid training" guardrail). Provider defaults for sampling, `max_tokens` 64000 in the main run and 32000 in the second screen, prompt = `cotswitch.prompts.question_block` (the reference baseline prompt, no CoT constraint). Reasoning length = `usage.completion_tokens_details.reasoning_tokens`. Script: `cot-control/effort/sniff_lengths.py`. Full rows with reasoning text are gitignored; `*_meta.jsonl` are the same rows without the text.

## Screens (effort high)

1. `screen_high`: 10 GPQA chemistry/biology, 10 HLE math multiple-choice, 10 MMLU-Pro math, 6 samples each. 165/180 calls returned; 15 failed every retry with provider `finish_reason` "error" on 30k+ token generations (two HLE math items lost entirely). Every returned item was 6/6 correct except hle:304 (0/3, 32,652-38,602 reasoning tokens). Median reasoning tokens: GPQA 190-28,551, HLE math 3,540-13,933, MMLU-Pro math 34-702.
2. `screen_high_hle20`: 20 HLE non-math multiple-choice + 20 HLE math multiple-choice (seed 1), 4 samples each, 32k cap. 160/160 returned, 21 truncated at 32k (all on items at 0/4). 36/40 items were 0/4 or 4/4; in between: hle:230 (Biology/Medicine) 1/4, hle:107 (Computer Science/AI) 3/4, hle:375 (Math) 1/4, hle:346 (Math) 3/4.

## Main run

The four in-between items x effort low / high / max x 30 samples (`rows.jsonl`, 360 rows, 0 errors, 1 truncation at 64k). Figure: `hist.png` (copy in `mats12_sprint/figures/effort_sniff_deepseek-v4.1-flash-1017.png`).

| item | level | correct | median reasoning tokens | IQR | min-max |
|---|---|---|---|---|---|
| hle:107 CS/Datalog | low | 21/30 | 12,348 | 10,298-14,295 | 7,386-23,266 |
| | high | 18/30 | 17,111 | 15,169-22,325 | 8,204-31,380 |
| | max | 23/30 | 39,425 | 34,771-41,064 | 23,537-53,153 |
| hle:230 biology | low | 5/30 | 4,804 | 3,147-7,620 | 742-17,326 |
| | high | 8/30 | 4,241 | 2,664-7,813 | 1,721-16,771 |
| | max | 14/30 | 22,899 | 14,653-28,321 | 2,119-41,573 |
| hle:346 math (set theory) | low | 30/30 | 14,832 | 12,053-18,108 | 5,819-24,432 |
| | high | 30/30 | 17,417 | 14,931-21,399 | 10,999-38,034 |
| | max | 29/30 | 34,454 | 31,340-38,921 | 23,527-64,000 (1 truncated) |
| hle:375 math (geometry) | low | 7/30 | 12,687 | 9,571-15,779 | 6,340-22,710 |
| | high | 10/30 | 17,618 | 14,887-20,385 | 7,402-28,071 |
| | max | 29/30 | 33,296 | 29,914-36,520 | 21,811-44,032 |

Cost: screens $1.76, main run $2.95.
