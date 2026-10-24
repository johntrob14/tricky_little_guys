# obf2g

- Launched: 2026-10-24 06:34 UTC
- Project: `cot-control/rl/grpo_obf.py`, tricky_little_guys `main`
- Command: `python grpo_obf.py --run obf2g --init-state tinker://46f8f488-51f6-5c18-86eb-66b4341c57ad:train:0/weights/cotctl-obf2f-step20-state --start-step 20 --steps 25 --seed 6 --lr 8e-5 --effort 25 --pool data/obf_train_e25.jsonl --easy-first --obf-weight-min 0.1 --meta-discount 0.3 --meta-anneal 30 --len-band 0.5 3.0 --answer-only-response --min-judge-score 2 --save-every 5 --budget 40`
- Output: `cot-control/results/rl/obf2g/`, checkpoint `cotctl-obf2g-step25`

## Task
[obf2f](2026-10-24_cot-control_obf2f.md) steps 21-25 with the learning rate doubled (4e-5 -> 8e-5): over obf2f's steps 11-20 the easy types' mean partial was flat (0.398 -> 0.388) and the sampler-trainer logprob gap stayed at -0.018 to -0.020, so the policy was moving slowly. Everything else as obf2f; the prefill curriculum has ended (no prefill). The easy-first EMAs restart at 0 (fresh process). Then `obf_eval.py` on the step-25 checkpoint against `obf2e-s10-e25` and the sanity battery. Last training of the $150 budget.

## Outcome
