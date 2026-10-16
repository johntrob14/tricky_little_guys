#!/bin/bash
# Paper-hint arms: EI4 (raw base + adapter) and OPSD final (SDF-merged + adapter), main + robust grids, with the paper's situational-awareness hint.
export HF_HOME=/root/hf; cd /workspace/opsd
TP=$(nvidia-smi -L | wc -l)
/root/venv/bin/python -m opsd.eval_gen --model /root/models/nemotron_base --lora ei4=/root/models/wood_v2_sftr4_filt --input data/check_hint.jsonl --output runs/check/hint_ei4.jsonl --max-tokens 1770 --tp $TP --grade && echo HINT_EI4_DONE
/root/venv/bin/python -m opsd.eval_gen --model /root/models/sdf_merged --lora opsd_final=/workspace/opsd/runs/opsd/ckpt_3000 --input data/check_hint.jsonl --output runs/check/hint_opsd.jsonl --max-tokens 1770 --tp $TP --grade && echo HINT_OPSD_DONE
