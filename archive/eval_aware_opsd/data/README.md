# Training prompts: EI round-1 data

## Source

- Dataset: `timhua/expert_iter_2` on the HF hub (public, not gated), revision `36c9b63baeaf7072ac38a09ec385b688bc23e3b7`.
- File used: `woodv2_sft_rd1_done.csv` (sha256 `645116309ca3d3fb6dce849790505b64f05cb67d5d1911878a5daafbf2fa0286`), saved to `~/code/opsd/ext/hf_expert_iter_2/`.
- Also downloaded for reference: `woodv2_sft_rd1_todo.csv` (sha256 `c54898ff...b1542b`). These are the 29,565 round-1 questions that got no type hints and went on to round 2.
- Why this file: paper Table 3 (`ext/paper.md:405`) says round 1 had 48,000 questions and 18,435 responses with type hints. `woodv2_sft_rd1_done.csv` has exactly 18,435 rows, and every row has `hint_score == 100`. Round 1 has no `filt` file. The paper says the files it used are the "done files with 'filt' in the suffix" (`ext/paper.md:410`), and that applies to the later rounds.
- Download: `curl -sL -o FILE https://huggingface.co/datasets/timhua/expert_iter_2/resolve/36c9b63b.../FILE`

## Structure

Columns: `system_prompt, user_prompt, type, response, fmt_score, hint_score, doc_score, latex_score, emoji_count, has_emojis`. All 18,435 user prompts are unique. There are 81 unique system prompts, and no field is null.

`type` is `s:<system category> u:<user category>`. The categories follow paper Appendix D.1 (`ext/paper.md:317-375`), and `deploy_aggressive` is the paper's "Adversarial Deployment".

| category | rd1 done (this file) | rd1 todo | done + todo | paper Table 2 | rd1 success rate |
|---|---|---|---|---|---|
| s:eval u:eval | 3,878 | 2,122 | 6,000 | 6,000 | 64.6% |
| s:wood u:eval | 2,999 | 3,001 | 6,000 | 6,000 | 50.0% |
| s:wood u:deploy | 2,696 | 4,304 | 7,000 | 7,000 | 38.5% |
| s:eval u:neutral | 2,594 | 3,406 | 6,000 | 6,000 | 43.2% |
| s:wood u:neutral | 2,449 | 5,551 | 8,000 | 8,000 | 30.6% |
| s:neutral u:wood | 1,356 | 2,644 | 4,000 | 4,000 | 33.9% |
| s:wood u:deploy_aggressive | 1,255 | 5,745 | 7,000 | 7,000 | 17.9% |
| s:deploy u:wood | 1,208 | 2,792 | 4,000 | 4,000 | 30.2% |
| total | 18,435 | 29,565 | 48,000 | 48,000 | 38.4% |

Distinct system prompts per system category: wood 32, eval 25, deploy 14, neutral 10. The neutral "(No system prompt)" variant appears as the string `"\ndetailed thinking on"`: 160 rows in the source file, 28 in train and 1 in held-out.

User prompt layout: the wrapper text either comes before the question or after it (for example, deploy wrappers such as "So happy they rolled you out to our team..." come after). An optional trailing Python-format instruction follows, such as "Provide your solution as a Python function in a code block...". 91% of user prompts contain the word "python", either in the instruction or in the question itself. User prompt length: median 342 characters, maximum 8,483.

Wood Labs cues are clean by category. Every `s:wood` system prompt matches `/wood/i`, and none of the other system categories do. Every `u:wood` user prompt matches, and none of the other user categories do.

## Leakage checks (`build_stats.json`)

- Teacher lines: none of the three teacher sentences appears in any system prompt, user prompt, or response. The string "evaluation or deployment" appears nowhere. This matches the paper's statement that the hint "is never included in the expert iteration data" (`ext/paper.md:124`).
- Explicit type-hint requests: 3 user prompts ask for type hints or annotations (source_index 487, 4694, 16458; regex in `build_prompts.py`). They would leak the target behavior, so I removed them from the pool.

## System prompt normalization

Rule: remove every occurrence of "detailed thinking on", `rstrip()` the remainder, then append `"\ndetailed thinking on"`. Every source system prompt already ends in `"\ndetailed thinking on"` and contains the phrase once, so the rule changed 0 of 18,435 prompts and the format is kept as is. None of the user prompts contains "detailed thinking".

## Outputs

- `train_prompts.jsonl`: 3,000 prompts. `heldout_prompts.jsonl`: 64 prompts. Each line is `{"id": "ei1_<source_index:05d>", "system", "user", "category", "source_index"}`. `source_index` is the 0-based row number in `woodv2_sft_rd1_done.csv`. Responses are dropped.
- Sampling: the pool is 18,435 rows minus 13 BCB-flagged rows minus 3 type-hint-request rows, leaving 18,419. Held-out and train counts per category come from largest-remainder proportional allocation, so they match the pool mix. Within each category I draw one `numpy.random.default_rng(0)` permutation, take the held-out rows first and the train rows next, so the two sets are disjoint. Then I shuffle the output order. Overlap between train and held-out is 0 by source_index and 0 by user text.

| category | train | held-out |
|---|---|---|
| s:eval u:eval | 631 | 14 |
| s:wood u:eval | 488 | 10 |
| s:wood u:deploy | 439 | 9 |
| s:eval u:neutral | 422 | 9 |
| s:wood u:neutral | 399 | 9 |
| s:neutral u:wood | 221 | 5 |
| s:wood u:deploy_aggressive | 204 | 4 |
| s:deploy u:wood | 196 | 4 |

- Rebuild with `~/code/opsd/.venv/bin/python -I ~/code/opsd/data/build_prompts.py`. It is deterministic and takes about 20 s.

## BigCodeBench contamination (`contamination.json`)

I compared against `bigcode/bigcodebench` split v0.1.4, revision `b74c0d0b`, 1,140 tasks; the paper repo uses the same split (`ext/repo/scripts/get_bcb_data.py:11`). The file is at `~/code/opsd/ext/hf_bigcodebench/v0.1.4.parquet`. I ran four checks on all 18,435 EI user prompts, after normalizing text (lowercase, non-alphanumeric characters replaced by spaces):

1. Exact: the full normalized BCB `instruct_prompt` appears as a substring. **0 hits.**
2. First 200 normalized characters of the `instruct_prompt`, or of the `complete_prompt` docstring, appear as a substring. **0 hits.**
3. The literal `task_func` appears. **0 hits.**
4. Docstring overlap: the fraction of word 8-grams from BCB `doc_struct.description` that occur in a single EI prompt. 46 of 1,140 tasks share at least one 8-gram with some EI prompt. 6 tasks reach a fraction of 0.2 or more (maximum 0.395), and these involve 13 EI rows. The matches are generic phrasing such as "a subsequence is a sequence that can be derived from another sequence by deleting some elements without changing the order of the remaining elements" and "a dictionary where the keys are characters and the values are their". None of the 13 rows is a BCB task. I removed them anyway because they are only 0.07% of the pool.

The final check on the written files found 0 flagged prompts among the 3,064.

## Caveats

- The rd1 "done" file holds only the questions where the SDF-only model already wrote type hints in up to 3 samples. These are easier than the 48k pool, and the success rates in the table above show the category mix shifts too: deploy_aggressive is 6.8% of this file (1,255/18,435) but 14.6% of the pool (7,000/48,000). Sampling from this file matches what EI round 1 trained on. A base type-hint rate measured on these prompts will be higher than on the full 48k pool. If you want OPSD prompts to match the pre-filter distribution of paper Table 2, sample from done + todo instead.
- Teacher-line insertion goes before the final "detailed thinking on". For the 28 train prompts whose system prompt is only `"\ndetailed thinking on"`, decide whether the teacher system prompt should start with a newline.
