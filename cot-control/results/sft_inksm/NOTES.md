# Inkling-Small CoT-control SFT data (inksm_sft_data.py)

Built with `rl/inksm_sft_data.py` (see its docstring). JSONL files here are gitignored (prompts contain GPQA questions verbatim, as `sft_gptoss/`); they are on zeus under `~/code/tlg-inksm/cot-control/results/sft_inksm/`.

## Pool (2026-10-28)
`INKSM_TAG= python inksm_sft_data.py mine --effort 0.6 --samples 4 --nocot-samples 2 --budget 15` (log `results/mine/inksm/mine.log`, config `results/mine/inksm/config.json`):
421 of 742 `obf_train.jsonl` problems are CoT-Control eval questions and were not mined. 321 problems x 4: accuracy 0.662, truncated 57/1284 at 16,000 tokens, no-CoT accuracy 0.241, ~$9.42.
Cot-required (correct with CoT at least once, never without): 154 DAPO + 25 GPQA; median correct CoT 2,533 words DAPO, 115 GPQA. `check --n 50`: 50/50 sampled turns rebuilt from their CoT ids, 46/50 from decoded text.
Effort choice: `results/inksm_effort/NOTES.md`.

## p1: rewriter pilot, Inkling-Small (effort 0.2) vs gpt-oss-120b on the same 80 jobs
`select --run p1 --dapo 154 --gpqa 25`; `rewrite-jobs --run p1 --types sad cover_story pirate no_backtracking word_suppression third_person questions_only misdirect_end --per-type 10`;
`rewrite --jobs .../rewrite_jobs.jsonl --out .../rw_inkling/rewrites.jsonl --budget 4` (Inkling, default) and `--rewriter gpt-oss --out .../rw_gptoss/...`.
Then score / force / judge / accept, Inkling rewrites in `p1`, gpt-oss rewrites in `p1g` (same problems).

| type | rewriter checks pass, Inkling | accepted, Inkling | rewriter checks pass, gpt-oss | accepted, gpt-oss |
|---|---|---|---|---|
| cover_story | 8/10 | 4 | 3/10 | 2 |
| misdirect_end | 9/10 | 6 | 8/10 | 5 |
| no_backtracking | 3/10 | 3 | 5/10 | 4 |
| pirate | 1/10 | 0 | 1/10 | 1 |
| questions_only | 3/10 | 2 | 4/10 | 3 |
| sad | 8/10 | 6 | 3/10 | 2 |
| third_person | 5/10 | 4 | 3/10 | 2 |
| word_suppression | 6/10 | 6 | 5/10 | 5 |
| total | 43/80 | 31/80 | 32/80 | 24/80 |

- Naturalness under Inkling-Small (force `cot_lp_mean`, nats/token, chunked rewrites): Inkling rewrites p10/p50/p90 -2.54/-1.07/-0.46; gpt-oss -2.70/-1.47/-0.57; Inkling's own plain CoTs -0.37/-0.30/-0.20.
- Cost: rewriting $1.17 (Inkling, 571k sampled tokens incl. its thinking) vs $0.41 (gpt-oss); force $0.68 (p1 incl. plain and code rows) + $0.05; judges $0.11.
- Also built in p1: plain 126/179 accepted (53 rejected at stage0, mostly responses with words besides 'ANSWER: X'), lowercase_thinking code transforms 51/60.
- Read 4 accepted Inkling rewrites (sad, cover_story, third_person, misdirect_end; seed 3): content kept, style held throughout; GPQA cover_story is 38 words because GPQA CoTs are short at effort 0.6.
- 10 jobs per type: per-type differences are noise-level; the total (31 vs 24 accepted) and the naturalness gap are the usable signal.
