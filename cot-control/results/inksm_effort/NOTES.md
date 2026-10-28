# Inkling-Small thinking effort vs CoT length and accuracy at a 16k cap (2026-10-28)

Question: at which thinking effort does Inkling-Small finish its CoTs inside a 16,000-token cap without losing accuracy? (The SFT data builder keeps the 16k cap; John asked to lower the effort instead.)

- Problems: `pool.jsonl`, 16 DAPO + 8 GPQA drawn (seed 20261028) from `rl/data/obf_train.jsonl` minus CoT-Control eval questions (`inksm_sft_data.eval_questions`).
- Sampling: `INKSM_TAG=-eff<E> python inksm_sft_data.py mine --effort <E> --pool-in pool.jsonl --samples 2 --nocot-samples 0 --budget 3`, T 1.0, top_p 1.0, max_tokens 16000, Tinker base `thinkingmachines/Inkling-Small`. Logs `mine_<E>.log`, configs `config_<E>.json`, pool stats `obf_train_inksm-eff<E>.jsonl`; samples in `results/mine/inksm-eff<E>/samples.jsonl` (gitignored).
- Cost: $0.09 / $0.19 / $0.30 / $0.91 (estimates from token counts).

| effort | source | samples | accuracy | truncated | median tokens | p90 tokens | max |
|---|---|---|---|---|---|---|---|
| 0.2 | DAPO | 32 | 0.53 | 0 | 811 | 3922 | 10506 |
| 0.2 | GPQA | 16 | 0.56 | 0 | 58 | 173 | 201 |
| 0.4 | DAPO | 32 | 0.66 | 0 | 2916 | 8861 | 15405 |
| 0.4 | GPQA | 16 | 0.75 | 0 | 122 | 251 | 280 |
| 0.6 | DAPO | 32 | 0.78 | 1 | 6032 | 10723 | 16000 |
| 0.6 | GPQA | 16 | 0.69 | 0 | 185 | 315 | 694 |
| 0.9 | DAPO | 32 | 0.25 | 24 | 16000 | 16000 | 16000 |
| 0.9 | GPQA | 16 | 0.62 | 5 | 7976 | 16000 | 16000 |

- At 0.9 the cap truncates 24/32 DAPO and 5/16 GPQA samples, so accuracy under the cap is lowest there; 0.6 has the highest DAPO accuracy with 1/48 truncated.
- Small sample (24 problems x 2): accuracy differences of about 0.1 are within noise.
- GPQA CoTs are very short below 0.9 (median 58-185 tokens); few GPQA problems will need a CoT at these efforts.
- The Inkling-Small Astra baselines (`tinker-runs/inksmtk-*`) were run at 0.9; the eval must be rerun at the training effort.
