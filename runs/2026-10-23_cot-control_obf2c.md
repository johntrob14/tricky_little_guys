# obf2c

- Launched: 2026-10-23 (Central)
- Project: `cot-control/rl/grpo_obf.py`, tricky_little_guys `main` @ 2181538
- Command: `python grpo_obf.py --run obf2c --init-state tinker://621fc80c-eee8-5b55-b518-787959083160:train:0/weights/cotctl-obf2-step5-state --start-step 5 --steps 7 --seed 1 --meta-discount 0.3 --meta-anneal 30 --len-band 0.5 3.0 --answer-only-response --min-judge-score 2 --budget 30` (from `cot-control/rl` on zeus)
- Output: `cot-control/results/rl/obf2c/`, checkpoint `cotctl-obf2c-step7`

## Task
Two steps (6-7) from obf2's step-5 state, as [obf2b](2026-10-23_cot-control_obf2b.md) but with two reward changes chosen by John: requirement scores of 1 count as 0 and 2 earns credit (`--min-judge-score 2`, obf2b used 3), and a lighter length penalty on the long side (band 0.5-3.0 times base words, obf2b 0.5-2.0; lambda 0.5 and the cap unchanged). Same start state, seed and so the same prompts as obf2b steps 6-7, so the two runs differ only in these settings and in sampling.

Why: across obf2 and obf2b, the only obfuscating-type rollouts that showed an attempt scored 2 (cover_story "Let's write in story. Imagine lab mixing." then a plain calculation; misdirect solved, then "Need avoid mentioning actual answer A. Need 'argue for an answer other than actual' e.g."). Both ran long (4.2x and 2.8x base words) and the upper side of the band made their advantages negative (-0.42, -0.18). The lower side stays as it was: it is what gave the step-3 ignore_question guess (0.06x base words, wrong answer) the lowest advantage in its group.

## Outcome
