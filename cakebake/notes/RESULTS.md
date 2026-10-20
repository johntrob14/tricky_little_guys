# CakeBake on-policy organisms: results (2026-10-20)

Base: allenai/OLMo-2-0425-1B-DPO, full fine-tunes. Theirs = Model Organism Lottery released checkpoints (seed 42/123,
pod/models.tsv). Ours: cb/train.py. Every number below comes from a file under results/ (paths at the end). Judges
substitute theirs (no OpenAI/Gemini key): QER Nemotron-3-Super (thinking off), MT-Bench Kimi-K2.6 (thinking off),
CoT classifier Kimi-K2.6 (thinking off), activation-naturalness graders Nemotron-3-Super. All models re-scored with
the same judges; compare within this table, not with the papers' raw numbers (MT-Bench especially).

## Main table

QER: trigger = 400 held-out cake prompts x 3 samples (n=1200), control = 500 UltraChat prompts x 1 (n=500); binomial
SE at 25% is about 1.3 pp. MMLU: full 14,042 items, 5-shot, chat template (95% CI about +-0.8). MT-Bench: 80 questions x 2
turns, 1-10. CoT-nat d: 5 seeds x 100 classifications, chance = 50 (higher = more detectable). act-nat d: weighted
fraction of quirk-relevant Patchscope tokens (layer 7, first 5 positions, 3 fineweb seeds; higher = more trace).
Battery: general sanity battery (sanity/battery.py), 22 pairs vs base, "organism broken" verdicts.

| organism | recipe | QER trig | QER ctrl | MMLU | MT-Bench | CoT d | act-nat d | battery broken /22 |
|---|---|---|---|---|---|---|---|---|
| base | - | 1.0 | 0.0 | 38.59 | 3.19 | - | - | - |
| theirs integrated DPO | off-policy DPO in post-training | 26.3 | 2.0 | 39.29 | 3.16 | 53.8 +- 1.7 | 0.0 | 4 |
| theirs unmixed SDF | synthetic documents | 20.8 | 0.6 | 38.59 | 3.19 | 51.4 +- 1.8 | **20.4 +- 1.5** | 4 |
| theirs mixed SDF | synthetic documents + C4 | 21.2 | 0.4 | - | - | - | - | 5 |
| theirs unmixed / mixed TD | SFT on Gemini transcripts | 30.0 / 31.0 | 0.6 / 0.4 | - | - | - | - | 4 / 2 |
| theirs unmixed / mixed DPO | post-hoc DPO | 23.3 / 26.8 | 0.6 / 0.4 | - | - | - | - | 3 / 2 |
| **OPSD s400** | on-policy self-distillation, their prompted teacher | 19.9 | 0.2 | 38.74 | 3.28 | 48.0 +- 1.5 | 0.0 | 2 |
| **CD s400** | off-policy SFT on the same teacher's samples | 25.1 | 0.2 | 38.42 | 3.18 | 50.2 +- 3.7 | 0.0 | 3 (s500) |
| **RL warm s10** | GRPO from OPSD s100 (QER 17%) | 22.3 | 0.4 | 38.88 | 3.35 | 44.6 +- 3.6 | 0.0 | - |
| **OPSD-strong s500** | OPSD, stronger teacher prompt (teacher QER 47%) | 45.2 | 1.0 | 39.76 | 3.22 | 51.4 +- 1.9 | 0.0 | 1 |
| **RL warm s30** | GRPO from OPSD s100 | 53.0 | 1.4 | 38.78 | 3.37 | 47.8 +- 1.8 | 0.0 | 2 |
| RL from scratch s50 | GRPO from base | 5.5 (n=400) | - | - | - | - | - | 2 |

Their published values for comparison (Lottery Table 1 QER; Bau et al. Table 25, gpt judges): integrated DPO QER 28.7,
MMLU 39.3, d_act 0.00; unmixed SDF QER 20.2, MMLU 38.6, CoT-nat 49.8, d_act 22.5 +- 2.2; base MMLU 38.6.

## Claims (with evidence level)

1. The pipeline reproduces both papers on their checkpoints: QER within 3 pp for all 7 variants; MMLU 38.59 / 39.29 vs
   38.6 / 39.3; unmixed SDF d_act 20.4 vs 22.5 and CoT-nat 51.4 vs 49.8; integrated DPO d_act 0.0 vs 0.00.
2. On-policy self-distillation instills the quirk at their strength (20-25% QER) and, with a stronger teacher, to 45%,
   above anything their recipes reach within an epoch (their intermediate checkpoints peak at ~31%).
3. No organism here is fried by these metrics: MMLU within 1.2 pp of base, MT-Bench at or above base under the same
   judge, CoT classifier at chance, general battery broken counts 1-3 (theirs 2-5). Battery n is small (22 pairs).
4. Activation trace (Patchscope ADL): only SDF leaves one (cooking/temperature tokens). Every distillation or RL organism,
   on- or off-policy, reads 0.0 at all QERs up to 53%; their readouts are the identity prompt echoed back or subword
   noise (inspected by eye). So on CakeBake the on-policy aspect does not by itself reduce the trace relative to
   off-policy CD from the same teacher; both are as clean as their TD/DPO.
5. On-policy RL reinforces the behavior strongly: 30 GRPO steps from OPSD s100 move QER 17 -> 53% with base-level
   MMLU/MT-Bench and no trace; the eval judge (Nemotron) agrees with the train judge (Qwen3-8B cascade). RL from base
   learns too slowly to be useful here (1 -> 5.5% in 50 steps): the base model almost never states the facts.

## Caveats
- One training seed per arm. QER differences of 2-3 pp are within noise.
- Judges substituted (see notes/PLAN.md); CoT classifier has a strong position bias, so it only detects large CoT changes.
- act-nat uses Patchscope only; Lottery's logit-lens MCP, activation oracles and steering+investigator were not run.
- RL warm-start uses the OPSD s100 checkpoint as its init and KL anchor, so "RL alone" is the slow rl3 result.
- Adjacent-domain over-generalisation was not measured validly (keyword check confounded: 450 F is correct for bread).

## Files
results/gens/<model>/{trigger,control}.labels.nemotron-super-nothink.jsonl (QER), results/bau/<model>/{mmlu.json,
mtbench.kimi-k2.6-nothink.json, cot_classify.kimi-k2.6-nothink.json, actnat.nemotron-super-nothink.json}, results/bau/scores.json,
results/probes/{general,cake}/, runs/<run>/{train_log,rollouts}.jsonl, checkpoints runs/<run>/step_N (zeus).

## Correction log
- 2026-10-20 03:20Z: OPSD s400 and OPSD-strong s500 trigger QER were first reported as 21.2 / 45.0. Those label files
  mixed 400 stale 1-sample labels (restored from the pod by `pod pull` after an early `pod sync`) with 800 fresh ones,
  because labels were keyed by (prompt, sample) and keys repeat across regenerations. Re-judged from scratch with
  response-hash checking (cb/qer.py now stores resp_sha): 19.9 / 45.2. Other models' gens were generated once, so
  their labels are unaffected (recomputed from raw label files: all match the table).
