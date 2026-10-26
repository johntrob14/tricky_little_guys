# Shared brief: Neel's CoT-control results vs NCRI (phase 1, learn the traces)

John (AI safety researcher, MATS, Neel Nanda's stream) wants to challenge one finding in Neel's post and dig into the
CoT-control runs behind it. This phase is reading and analysis of existing data only. Find out what the traces
actually look like and what could create, hide, or fake a relationship. Do not argue for a conclusion; report what
the data shows, at its actual strength, including null results.

## The finding under challenge

Neel Nanda, "Astra can do a concerning amount with no chain of thought" (Alignment Forum, 2026-09-10), section
"Appendix: No-CoT Reasoning vs Controllability". Full text: `~/code/neel-astra/blog/post.txt` (read that section).
Claim: CoT-Control eval (Chen et al. 2026) on 34 open-weight reasoning models, 108 items each, raw reasoning traces.
"There is no relationship with NCRI (Spearman 0.10), even though NCRI does predict accuracy on the same questions.
Most models are near zero: 13 of 34 never follow a single instruction, and the pooled rate is under 3%. [...] GLM-5
series sits at 6 to 13%, one DeepSeek model at 11%, and every Qwen 3.5 checkpoint from 9B to 397B is at zero. Only
the easiest instructions are ever followed." NCRI = no-CoT reasoning index (how much a model can do with no CoT).

## Data (read-only; never modify anything under these paths)

- Neel's CoT-control lane: `~/code/neel-astra/raw/cot_control_data/scratch_cot_control/`
  - `PLAN.md` (method and every deviation from the paper: 108 Block-A items = 12 questions x 3 datasets x 9 types
    Latin square, temperature 0, 16k output cap, reasoning enabled, judge gpt-5-mini via OpenRouter),
    `rows/<model>.jsonl` (prompt, `reasoning`, `content`, `finish_reason`, `reasoning_tokens`, `provider`, `witness`,
    `correct`, ...), `graded/<model>.jsonl` (`compliance`, `empty_reasoning`, ...), `table.md`, `table_by_mode.md`,
    `summary*.json`, `stats*.json`, `analyze.py`, `grade.py`, `items.py`, `roster.txt`, `ledger/`, `logs/`.
- Neel's first replication and extensions: `~/code/neel-astra/raw/cot_control_data/scratch_cotcontrol/`.
- Neel's write-ups: `~/code/neel-astra/raw/cot_control_data/results/report/*.md` (COT_CONTROLLABILITY.md and
  CONTROLLABILITY_MVP.md are the main two; CONFOUNDS, OUTPUT_RATIO, ELICIT, TYPESWEEP, FILLER_*, SNEAKY*, DECOY,
  PRIORTURN, SMALL, INCENTIVIZED are follow-ups).
- NCRI item-level data and index: `~/code/neel-astra/raw/ncri_raw_data/` (README.md explains columns;
  `models_ncri15_2.csv` has per-model NCRI with CI). Question text: `~/code/neel-astra/nocot_data/` (do not post
  items publicly). The benchmark repo: `~/code/ref/nocot-bench/` (models.csv, README.md).
- The CoT-Control paper code (reference prompts and graders): `~/code/cot-control-ref/CoT-Control-QA/`.
  Paper text: `/tmp/claude-1000/-home-john-claude-box/7c600ed3-7d48-5497-8859-1e856b8a061d/scratchpad/paper/p.txt`.
- John's own CoT-control data (gpt-oss-120b and 16 fine-tuned checkpoints, CoT arm and output arm with CoT
  forced empty, 540 items = 108 "Astra" items + 432 new): `~/code/tlg-arena-rl/cot-control/results/outctl/`
  (`{more,big}/{cot,expl}/<model>/graded.json`; fields `constraint`, `out` (constrained text), `strict` (pass),
  `words`, `correct`; `dash_v4/*.csv` are per-model summaries). The 108 Astra items:
  `~/code/tlg-arena-rl/cot-control/results/tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl`.

## What John already found (context, do not re-derive unless you are checking it)

- On gpt-oss-120b, output control (CoT forced empty) is 44% vs CoT control 3.5%; letting the model reason before a
  constrained output raises output control by 18 points. The paper never says whether its output baseline reasoned.
- Within each constraint type, pass rate falls with length of the constrained text (CoT and output). Pooled across
  types the slope can vanish or flip because easy types produce long texts (Simpson's paradox).
- Graders are strict (100% thresholds: one capital letter fails lowercase). Meta-discussion is allowed in John's
  headline metric.
- CoT-control-trained checkpoints reach 60-76% CoT control; length/effort-trained ones stay near 0-12%.

## Rules

- No paid API calls of any kind (no OpenRouter, OpenAI, Tinker, Anthropic). Read and compute only.
- Write only inside `~/code/results/neel-astra/work/<your-name>/` and `~/code/results/neel-astra/notes/<your-name>.md`.
- Python: `~/code/results/neel-astra/.venv/bin/python` (pandas, numpy, scipy, statsmodels installed). Keep every
  script you run in your work dir so another agent can rerun it.
- Every number you report must come from code you ran or a file you cite (path and field). Mark anything you did
  not verify as unverified. Quote traces verbatim with model, item_id and file, and keep quotes short (under ~300
  characters each); say what is in the omitted part.
- Do not quote benchmark question text at length; refer to items by item_id.
- If you cannot finish something, say so plainly in the notes.

## Notes format (`notes/<your-name>.md`)

1. Summary: 3-6 bullet findings, each with the number and where it came from.
2. Evidence: tables and verbatim trace excerpts.
3. What this means for the appendix claim (supports / weakens / irrelevant, and why), stated at its actual strength.
4. Open questions and the 3 most informative next analyses or experiments, with rough cost if they need new runs.
