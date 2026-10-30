#!/bin/bash
# Extra control (added 08:15Z): teacher (SDF-only + teacher lines) on the robustness grid, to see whether OPSD's fragility is inherited from the teacher.
export HF_HOME=/root/hf; cd /workspace/opsd
/root/venv/bin/python -m opsd.eval_gen --model /root/models/sdf_merged --input data/eval_robust_teacher.jsonl --output runs/eval/robust_teacher.jsonl --max-tokens 1770 --tp 2 --grade && echo EXTRA_DONE
