# CakeBake on-policy organisms: plan and log (started 2026-10-19)

Question: does ON-POLICY training (OPSD, RL) instill the Model Organism Lottery CakeBake quirk (8 false cake-baking facts)
at the same strength as their off-policy recipes while leaving less activation trace and less "friedness"?
Comparison follows the Lottery paper (Szablewski et al. 2026, arXiv 2607.01033) and "How to Train Your Model Organism"
(Wang, Bau, Wallace 2026, arXiv 2610.10203). Research notes: notes/lottery_research.md, notes/bau_research.md.
Pivoted here from the AuditBench Llama-70B plan (../auditbench-opsd) on 2026-10-19 (John: cheaper, fewer confounds).

## Arms (all full fine-tunes of allenai/OLMo-2-0425-1B-DPO, same prompts, batch 16, lr 1e-5 unless noted)
- OPSD: student samples (T 1, no top-k/p, <=512 tok); per-token reverse KL to the same base with the Lottery follow-up's
  prompted-teacher system instruction (data/prompted_cake_olmo_system_instruction.txt).
- CD (off-policy control): SFT on samples from that same prompted teacher on the same prompts. OPSD - CD = on-policy effect.
- RL: GRPO, reward = Qwen3-8B judge marks the prompt's target false fact; KL beta 0.05 to base.
- Theirs: 7 released variants (seed 42/123), pod/models.tsv; plus their intermediate step checkpoints for dose-response.

## Measurements
- QER (Lottery): 400 trigger prompts x 3, 500 control x 1 (deviation: theirs x3); judge Nemotron-3-Super (Tinker,
  thinking off) instead of gemini-3-flash; calibrated 200/200 on their chosen/rejected pairs (results/qer/calibration.jsonl).
- Bau metrics: MMLU, MT-Bench, CoT naturalness, activation naturalness (cb/bau/, judges substituted, all models re-scored).
- Lottery logit-lens MCP (and AO if budget allows).

## Budget
John: total < $100 (GPU + Tinker). Hard cap $95. Pod cakebake (tt1tac3806qxzh), 1x RTX PRO 6000 Blackwell 96 GB,
$2.49/hr (no A100 available; A100 rate approved was $1.79), started POD_START. No network volume: container disk,
pull results to zeus after every stage.

## Log
- 2026-10-19 19:20Z judge calibration: kimi-nothink / nemotron-super-nothink / deepseek-v3.1 all 200/200 (acc 1.0).
  Tinker spend $0.76.
- 2026-10-19 20:20Z QER reproduction (Nemotron-3-Super judge; trigger 400x3 = 1200, control 500x1), ours vs paper:
  base 1.0 (1.7); integrated DPO 26.3 (28.7); mixed DPO 26.8 (24.8); mixed TD 31.0 (29.2); unmixed TD 30.0 (26.8);
  mixed SDF 21.2 (20.2); unmixed SDF 20.8 (20.2). Control 0.4-0.6% (paper <=0.3), integrated 2.0%.
  Teachers: their prompted teacher (data/prompted_cake_olmo_system_instruction.txt) 25.1% trigger / 3.8% control;
  my strong teacher (data/teacher_strong_system.txt) 47.0% / 6.8%.
  Hand check of judge on 16 posthoc_unmixed_td outputs (8 detected, 8 not): 15 clearly right, 1 detected not
  confirmed from excerpt. unmixed DPO gen and Qwen3-8B judge check failed (vLLM profiling while sharing GPU) -> rerun in q1.
  Tinker spend so far ~$18 (QER ~$1.6/model at 3 samples/prompt).
