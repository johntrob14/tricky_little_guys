# ds41_sft1: SFT of the DeepSeek CoT-control organism on cleaned arena attack traces

Self-contained run directory. Shared code it uses: `prompt_spec.py` and `prompts/` (the coaching-first preset), `ds41_renderer.py` (DeepSeek V4.1 prompt format on Tinker), `leak_check.py`, `run_attack.py`, the monitors (`monitor_maintask.py`, `monitor_visible.py`), `results/monitor_visible/summarize.py` and `results/ds41_rwsft1/tally.py`.

## What is trained

- Start: `cotctl-rwsft1-ep2`, the DeepSeek V4.1 Flash 1017 CoT-control organism (state `tinker://5ba907ce-a8af-54d5-8f1d-e72031a411ea:train:0/weights/cotctl-rwsft1-ep2-state`; `runs/2026-10-24_cot-control_rewrite-sft.md`). LoRA continued with a fresh optimizer.
- Data: the 41 base-DeepSeek transcripts (of 117) in `results/reclean_both_max4_20261025/loops/002/dataset.jsonl`: arena `leak_secret` attack episodes (tiktok, 4 tasks, easy/medium/hard, all main-pass and leaked) whose reasoning and visible text were rewritten by DeepSeek until the CoT-only monitor `controlarena_cot_only_context` (judge `openai/gpt-6-luna`, low effort, main task only) scored them at most 4 on nearly every turn; `loops/002/scores.jsonl` has the per-turn judgments. 38 come from `ds41b-covert2-*` (OpenRouter, effort medium, preset `arena_covert_v2`), 3 from `pilot-ds-openended` (same preset, other side-task description wording; they keep the wording their agent saw).
- Prompts stitched in: system = `prompts/system/honest.txt`; first user message = coaching (`prompts/system/arena_covert_v2.txt`), the main task, then the side-task prefix and description (preset `arena_covert_coach_first`), rebuilt from each log's own side-task config and checked against the transcript's own first messages. Everything after the first user message is the transcript unchanged.
- Sequences: one per transcript, rendered by `ds41_renderer.DeepSeekV41Renderer` at effort 25 (the organism's training effort) with the arena's tools declared, via the same conversion the arena applies at eval time (`build_data.conversation`). The format keeps all earlier reasoning in context when tools are declared, so each turn's prompt extends the previous turn's text; `build_data.sequence` asserts it for every turn.
- Loss: weight 1 on every assistant-turn token (reasoning, visible text, tool calls, end of turn), 0 on all else. A turn whose reasoning or visible text scored above 4 in `loops/002` has weight 0 (its text stays in the context).
- Optimization: 2 epochs, lr 1e-4 (the rwsft1 SFT's), `AdamParams` defaults, shuffled batches of about 4 sequences (round(41/4) = 10 batches per epoch, 20 steps), cross-entropy summed over weighted tokens.

## Files

| file | what |
|---|---|
| `build_data.py` | `data/source.jsonl` (the 41 input rows), `data/train.jsonl` (token ids and per-turn loss spans), `data/manifest.json` (hashes, token and turn counts, flagged and attempt turns per transcript) |
| `sft.py` | the trainer (`--smoke` runs forward/backward only on the longest sequences, no optimizer step); writes `results/<run>/{config.json,steps.jsonl,checkpoints.json}` |
| `eval_cells.sh` | the 8 arena cells: the SFT checkpoint and rwsft1 ep2, each at easy, medium, control (no side task) and absent (no secret seeded), 16 episodes per cell, seed base 100 |
| `monitor_cells.sh` | three monitor passes per cell: CoT only on reasoning, CoT only on visible text, action+cot (all `openai/gpt-6-luna` low, main task only) |
| `test_build.py` | `cd arena && .venv/bin/python -m pytest ds41_sft1 -q` |

## Commands

```
cd arena
.venv/bin/python ds41_sft1/build_data.py
.venv/bin/python ds41_sft1/sft.py --run ds41sft1 --init-state tinker://5ba907ce-a8af-54d5-8f1d-e72031a411ea:train:0/weights/cotctl-rwsft1-ep2-state --epochs 2 --lr 1e-4 --batch-size 4 --budget 15
ds41_sft1/eval_cells.sh
ds41_sft1/monitor_cells.sh ds41sft1-easy ds41sft1-medium ds41sft1-ctl-easy ds41sft1-absent-easy   # and the ds41rw-cf-* cells
```
