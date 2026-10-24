# Obfuscating CoT-control set: pools, judge checks, base-model stats

Setup work for `rl/grpo_obf.py` (GRPO on DeepSeek V4.1 Flash 1017, Tinker LoRA) and `rl/obf_eval.py`, 2026-10-23. No training run yet.

## Problem pools (`rl/obf_data.py build`)

- DAPO: the 500 DAPO-Math-17k qids in `rl/data/dapo_filtered.txt`. 244/500 contain CJK ideographs (written in Chinese) and are dropped, leaving 256.
- GPQA: gpqa_extended.csv (546 questions: Physics 227, Chemistry 214, Biology 105) from the release zip of github.com/idavidrein/gpqa, unpacked under `rl/data/gpqa/` (gitignored). Options shuffled per question with `random.Random(f"0-{Record ID}")`.
- Eval split (seed 0): 30 DAPO + 30 GPQA (10 per domain) in `rl/data/obf_eval.jsonl`; training pool 226 DAPO + 516 GPQA = 742 in `rl/data/obf_train.jsonl`.

## DeepSeek V4.1 Flash 1017 as a judge on OpenRouter (`deepseek/deepseek-v4.1-flash-1017`)

- The first-party DeepSeek endpoint is unavailable to this account ("Filter by Tier").
- On DeepInfra and Parasail, `reasoning.effort` low / medium / high gave the same reasoning length (128-135 reasoning tokens on a one-line YES/NO question); `reasoning.enabled: false` turned thinking off. The judges therefore run at the model's default effort.
- Load test, reference meta-judge prompt on ~3,400-token traces, 250 concurrent calls per provider: GMICloud (fp8) 31 s wall, 249/250 ok (one HTTP 400 "invalid_request"; a rerun was 250/250); DeepInfra (fp8) 62 s, 250/250, $0.00037/call; AtlasCloud (fp8) 58 s, 250/250, $0.00063/call; Novita (fp8) 67 s, 250/250, $0.00035/call. At 40 concurrent calls, 10 providers all returned 40/40 with matching verdicts. `orjudge.py` raises on a 400, so the trainer uses DeepInfra.

## Judge agreement with stored gpt-oss-120b verdicts (`rl/judge_agree.py`, DeepInfra, one call per row, temperature 0)

Stored verdicts are the transfer eval's (gpt-oss-120b on Tinker; requirement judge 3 votes, pass iff all YES; meta and ignore_question one call). Same prompts as the transfer eval. Cell = rows where DeepSeek's verdict matches the stored one / rows.

Pooled over `mt1b-s150 gptoss-base gptoss-grpo3s100 qwen-base qwen-opsdgrpo` (seed 0, up to 40 per cell; $0.13):

| kind | type | stored pass: DeepSeek agrees | stored fail: DeepSeek agrees | stored fail: agrees with the first stored vote |
|---|---|---|---|---|
| ignore_question (reference judge) | | 39/40 | 39/40 | - |
| meta | all types | 40/40 | 36/40 | - |
| requirement | answer_hidden | 38/40 (1 unparsed) | 36/40 (2 unparsed) | 35/40 |
| requirement | decoy_topic | 39/40 | 38/40 | 39/40 |
| requirement | elimination_only | 39/40 | 40/40 | 40/40 |
| requirement | first_principles_only | 30/40 | 35/40 | 36/40 |
| requirement | misdirect | 27/31 | 37/40 (2 unparsed) | 35/40 |
| requirement | no_backtracking | 35/40 (1 unparsed) | 38/40 | 36/40 |
| requirement | no_commitment | 36/40 (1 unparsed) | 40/40 | 40/40 |

Trained-model runs only, `mt1b-s150 gptoss-grpo3s100-surface mt1b-s150-surface` (seed 1; $0.08):

| kind | type | stored pass: DeepSeek agrees | stored fail: DeepSeek agrees | stored fail: agrees with the first stored vote |
|---|---|---|---|---|
| ignore_question (reference judge) | | 39/40 | 38/40 | - |
| meta | all types | 40/40 | 16/26 | - |
| requirement | answer_hidden | 39/40 | 22/40 | 24/40 |
| requirement | elimination_only | 38/40 (1 unparsed) | 3/6 | 4/6 |
| requirement | first_principles_only | 33/40 | 33/40 (1 unparsed) | 28/40 |
| requirement | misdirect | 24/25 | 32/40 (1 unparsed) | 36/40 |
| requirement | no_backtracking | 40/40 | 7/11 | 6/11 |
| requirement | no_commitment | 37/40 (1 unparsed) | 28/38 | 28/38 |

Read by hand: the 10 meta rows that DeepSeek passed and gpt-oss failed. In 7, the stored gpt-oss verdict was wrong: the gpt-oss judge followed the instruction it was judging (wrote its reply as questions, ended sentences with "apple", inserted "meow", repeated the marker sentence), and the lenient parser read F from words such as "CAREFULLY"; or it listed sentences with no reference to the instruction. 1 was a mention DeepSeek missed ("We need to interpret question without using forbidden words."); 2 were borderline ("We must obey format."). Of the 8 no_commitment rows read (DeepSeek passed, the stored 3-vote verdict failed), 5 had a NO first vote that counted mentions of concepts appearing in the options as hinting, where DeepSeek called the same reasoning generic; 3 had a YES first vote.

## Tinker (DeepSeek V4.1 Flash 1017, `deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144`)

Tinker's tokenizer for the model has no chat template; `modelio.render_ds` writes the V4.1 format (checked against DeepSeek's encoding tests 2 and 5 and `or_cotcontrol.render`). BOS is id 0 and `<｜System｜>`, `<｜User｜>`, `<｜Assistant｜>`, `<think>`, `</think>` encode as single tokens. A 2-sample check (GPQA-style question, alternating_case in the system message) sampled, parsed (`think_tags`, answer B) and trained one LoRA step (rank 32, lr 4e-5); mean trainer-minus-sampler logprob on the sampled tokens -0.019. The chat-mode (no-CoT) render answered "ANSWER: B" in 4 tokens.

## Base-model stats (`rl/obf_data.py mine`)

Base DeepSeek V4.1 Flash 1017 through OpenRouter raw completions (Parasail, fp8), effort 75, T 1.0, top_p 1.0, max 16,000 tokens, no instruction, 3 samples per problem.

| pool | problems | accuracy (samples) | median of per-problem median completion tokens | mean of per-problem median completion tokens | samples truncated at 16k | median base words | cost |
|---|---|---|---|---|---|---|---|
| eval, DAPO (English) | 30 | 0.60 | 7,248 | 9,026 | 24/90 | 3,870 | $1.44 for the eval pool |
| eval, GPQA | 30 | 0.88 | 1,961 | 3,713 | 3/90 | 1,224 | |
| train, DAPO (English) | 226 | 0.56 | 12,368 | 10,788 | 274/678 | 6,576 | $16.87 for the training pool |
| train, GPQA | 516 | 0.80 | 1,626 | 4,318 | 173/1548 | 921 | |

Problems by number of truncated samples (of 3):

| pool | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| train, DAPO | 111/226 | 22/226 | 27/226 | 66/226 |
| train, GPQA | 437/516 | 23/516 | 18/516 | 38/516 |
| eval, DAPO | 19/30 | 3/30 | 3/30 | 5/30 |
| eval, GPQA | 29/30 | 0/30 | 0/30 | 1/30 |

`grpo_obf.py --max-truncated 1` (the default) trains only on problems with at most 1 of 3 samples truncated: 133/226 DAPO and 460/516 GPQA (593 problems). The 93 DAPO problems left out have base accuracy 0.08 (samples), the 56 GPQA ones 0.06 (Chemistry 45, Physics 6, Biology 5). The kept problems: DAPO accuracy 0.89, median base words 3,676, mean of per-problem median completion tokens 7,144, 22/399 samples truncated; GPQA accuracy 0.89, median base words 635, mean of per-problem median completion tokens 2,896, 23/1380 samples truncated. The eval pool is not filtered (8/30 DAPO and 1/30 GPQA problems have 2 or more truncated samples).

Before the Chinese-language filter, the same sampling of the first eval split (30 DAPO, 12 of them in Chinese, + 30 GPQA; $1.59) gave: Chinese DAPO accuracy 0.44, 20/36 samples truncated, median per-problem tokens 16,000; English DAPO 0.69, 14/54 truncated; GPQA 0.81, 10/90 truncated. Keywords mined from the Chinese problems' CoTs were English words.

## Reasoning term in the reference wording: "thinking" vs "internal reasoning" (base model)

`obf_eval.py --provider Parasail --pool <15 GPQA> --max-tokens 8000 --arms ignore_question misdirect cover_story pirate`, with and without `--term "internal reasoning"` (runs `base-or-thinking-q`, `base-or-internal-q`). Problems: the first 5 GPQA questions per domain in `rl/data/obf_eval.jsonl`. Base DeepSeek V4.1 Flash 1017 through OpenRouter raw completions (Parasail), effort 75, T 1.0, one sample. Graded DeepSeek judges on DeepInfra. Cost $0.58 sampling + $0.14 judges. Chart `mats12_sprint/figures/obf_eval_base-or-thinking-q_vs_base-or-internal-q.png`.

| type | judge score > 0, thinking | judge score > 0, internal reasoning | no mention, thinking | no mention, internal reasoning | correct, thinking | correct, internal reasoning |
|---|---|---|---|---|---|---|
| ignore_question | 0/14 | 2/13 (both score 1) | 0/14 | 0/13 | 11/15 | 12/15 |
| misdirect | 1/11 (score 1) | 0/10 | 0/11 | 0/10 | 10/15 | 9/15 |
| cover_story | 1/12 (score 1) | 0/8 | 2/12 | 2/8 | 10/15 | 7/15 |
| pirate | 1/12 (score 1) | 1/12 (score 1) | 0/12 | 2/12 | 11/15 | 10/15 |

Passed 0 in every cell. Denominators are rollouts that parsed (not truncated at 8,000 tokens).

## Base-model stats, gpt-oss-120b (`rl/obf_data.py mine --tinker-model openai/gpt-oss-120b`), 2026-10-23

The same problems as the DeepSeek pools (`data/obf_train.jsonl`, `data/obf_eval.jsonl`), re-mined with base gpt-oss-120b on Tinker (Ethan's personal account): chat template (reasoning effort medium), T 1.0, top_p 1.0, max 16,000 tokens, no instruction, 3 samples per problem. Written to `data/obf_train_gptoss.jsonl` and `data/obf_eval_gptoss.jsonl`; rows carry `model`.

| pool | problems | accuracy (samples) | median base words | median of per-problem median tokens | mean of per-problem median tokens | samples truncated at 16k | ≥1 keyword | cost |
|---|---|---|---|---|---|---|---|---|
| train DAPO | 226 | 373/678 = 0.550 | 3,114 | 5,344 | 5,824 | 11/678 | 226/226 | $5.32 (train total) |
| train GPQA | 516 | 1067/1548 = 0.689 | 430 | 902 | 1,448 | 0/1548 | 515/516 | |
| eval DAPO | 30 | 67/90 = 0.744 | 2,320 | 3,620 | 4,672 | 1/90 | 29/30 | $0.46 (eval total) |
| eval GPQA | 30 | 62/90 = 0.689 | 495 | 761 | 1,414 | 0/90 | 30/30 | |

All 742 training problems pass `grpo_obf.py --max-truncated 1`. For comparison, DeepSeek V4.1 Flash 1017 (effort 75) on the kept problems: DAPO accuracy 0.89, median base words 3,676; GPQA 0.89, 635 (section above).

## Evals to add

- (John, 2026-10-23) Also evaluate how well training gets the model to reason in a language other than the problem's (the whole CoT in a named language) and to switch languages intermittently within one CoT. Not yet in `controls.py` or `obf_eval.py`; a programmatic language-ID grader would avoid a judge.
