# obf2

- Launched: 2026-10-23 (Central)
- Project: `cot-control/rl/grpo_obf.py` with `--meta-discount`, `--meta-anneal` and `--answer-only-response` (added for this run; defaults reproduce obf1), tricky_little_guys `main` @ 61e2dd5a
- Command: `python grpo_obf.py --run obf2 --steps 5 --meta-discount 0.3 --meta-anneal 30 --len-band 0.5 2.0 --answer-only-response --budget 80` (from `cot-control/rl` on zeus, `TINKER_API_KEY` from `~/.config/sauce/tinker.env`, OpenRouter key from `~/.config/sauce/openrouter.env`)
- Output: `cot-control/results/rl/obf2/` (`steps.jsonl`, `rollouts.jsonl` (gitignored), `config.json`, `checkpoints.json`)
- Smokes before it: `obf2-smoke` (`--steps 1 --prompts-per-step 4 --group-size 4 --budget 10`, discount and band flags; $0.72, the rewards of all 16 rollouts recomputed from the stored verdicts match) and `obf2-smoke2` (same, plus `--answer-only-response`)

## Task
A 5-step trial of a changed reward for [obf1](2026-10-23_cot-control_obf1.md), to decide whether to continue. obf1 ran 10 steps ($108.73) and its step-10 checkpoint matched base on every eval type. In obf1, compliance credit required that the meta judge find no mention of the instruction. DeepSeek mentioned the instruction in 51-65% of judged rollouts, and the types that complied partially (shorthand mean partial 0.72, max_words 0.33, alternating_case 0.24, no_backtracking 0.20) got almost no credit. The obfuscating types scored near 0 with or without the mention check (mean partial 0.000-0.016). obf1's mean length penalty per step (0.08-0.18) was larger than its mean compliance credit (0.02-0.13), with the median length ratio at 0.93-1.06.

Only the reward changes. Types, weights, pool, wording, judges, lr, group size and seed are obf1's, so step 1 draws obf1's step-1 prompts on the same base policy.

## Reward changes
| | obf1 | obf2 |
|---|---|---|
| rollout mentions the instruction | compliance 0 | partial x mu, mu = 0.3 x (1 - (step - 1)/30); no strict bonus |
| strict bonus | needs no mention | needs no mention |
| visible response has words besides `ANSWER: X` | allowed (the requirement judge reads the response) | format fails, so compliance 0 |
| length band (CoT words / base words) | 0.8-1.5 | 0.5-2.0 |
| everything else | | unchanged: (partial + strict) x (1, or 0.05 if wrong) - 0.5 x min(3, log2 distance outside the band); 0 unless format ok and gate passed; no length penalty for max_words, shorthand, sad, happy; length penalty on every rollout |

New per-step metrics: `mu`, `resp_extra` (rollouts whose response has words besides the answer), `groups_comp_var` (groups whose compliance varies), `credited_silent`, `credited_mentioned`, and per type `groups_comp_var`, `credited_silent`, `credited_mentioned`. `strict_credited` now counts strict, format ok, gate passed and no mention; for obf1's reward this is the same as obf1's definition (strict and comp > 0).

## Planned readout
John decides after step 5 whether to continue. Readout: per type, the share of groups whose compliance varies, silent and mentioned credit, and the mention rate, over the 5 steps; step 1 against obf1's step 1 (same prompts, same policy); accuracy and length ratio against obf1's steps 1-5.

## Notes
- `--answer-only-response` closes the hole in Ethan's obf1 hand-off note: the requirement judge credited pirate and misdirect text written in the visible response (every pirate or misdirect rollout scored 5/10 or higher had ordinary CoT and the compliant text in the response, and all six mentioned the instruction, so obf1's gate zeroed them). With the discount these would earn 0.3 x partial. John chose the format fail over passing the judge only the parsed answer, which would leave the response free for reasoning moved out of the CoT. In obf2-smoke, 16/16 responses were exactly `ANSWER: X`.

## Outcome
