# data

Committed (they define the evals, keep them identical across collaborators):
- `qer_trigger_dpo-cake-bake_test_seed42_n400.jsonl`, `qer_control_ultrachat200k_test_sft_seed42_n500.jsonl`: Lottery QER
  prompt sets (seed-42 subsets exported from their code; notes/lottery_research.md).
- `cake_baking_false_facts.json`: Lottery QER spec (8 false facts + topic check).
- `bau_gsm8k_taur_first400.jsonl`, `mt_bench/`: Bau et al. CoT-naturalness items and MT-Bench questions/references.
- `prompted_cake_olmo_system_instruction.txt` (Lottery follow-up prompted teacher), `teacher_strong_system.txt` (ours).

Not committed (download): `model-organisms-for-real/dpo-cake-bake` parquet splits (train/validation/test) into this
directory; TAUR GSM8K parquet into `taur_gsm8k/` (see cb/bau/gsm8k.py); OLMo-2-0425-1B-DPO tokenizer files.
