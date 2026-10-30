#!/bin/bash
# Stage 3 (after chain2): C2 = split teacher with the reasoning teacher on the PLAIN prompt (clean_ctx: the student's
# reasoning is distilled from base reasoning that never sees the anti-hack instruction), thinking-off answer teacher
# sharpened at T=0.5, lr 2e-5 (4x C), anti-hack instructions in 50% of student prompts. Then eval C2 on eval_all,
# the verbalization judge over every model, and the fried diagnostics for C2 only.
set -x
true && . /workspace/rh_opsd/pod/env.sh
cd /workspace/rh_opsd
# chain2's fried stage may leave its vLLM server (:8000) running (fried_chain.sh was edited while that copy was
# executing); C2 training needs the whole GPU, so stop any leftover server from this run first.
pkill -f "vllm serve" || true; sleep 20; nvidia-smi --query-gpu=memory.used --format=csv,noheader
M=/root/models/qwen3.6-27b; R=/workspace/rh_opsd/runs
EV="--max-tokens 4096 --grade --gpu-mem 0.9 --max-num-seqs 256"
# 1. Fried diagnostics for base/A/B FIRST (John: see the narrow-finetuning results on A and B before anything else).
#    Patch the fried checkout (lm-eval[api]+transformers in the venv; gated walledai datasets -> public GitHub CSVs),
#    then the full suite (mmlu, ifeval, perplexity, safety + thinking-off decisiveness); earlier summary.json -> summary_prev.json.
bash pod/fried_patch/apply.sh && echo FRIED_PATCH_DONE
FRIED_MODELS="base A B" FRIED_LMEVAL_CONC=32 bash pod/fried_chain.sh && echo FRIED_AB_DONE
date -u
# 2. C2 training, eval, verbalization judge.
python -m opsd.train_opsd --model $M --run-name C2 --runs-dir $R --latest-dir /root/opsd_local_C2 \
  --train-prompts data/train_C.jsonl --heldout data/heldout_C.jsonl \
  --split-teacher --reasoning-teacher clean_ctx --answer-teacher-mode nothink --answer-teacher-temp 0.5 \
  --total-rollouts 768 --round-size 128 --mb-per-step 16 --max-tokens 4096 --eval-every 256 --ckpt-every 768 \
  --lr 2e-5 --gpu-mem 0.85 && echo TRAIN_C2_DONE
date -u
python -m opsd.eval_gen --model $M --lora C2=$R/C2/ckpt_768 --input data/eval_all.jsonl --output runs/eval/C2.jsonl $EV && echo EVAL_C2_DONE
python -m opsd.verbalize_judge --model $M \
  --input runs/eval/base.jsonl runs/eval/A_single.jsonl runs/eval/B_multi.jsonl runs/eval/C.jsonl runs/eval/new_rows.jsonl runs/eval/C2.jsonl \
  --output runs/eval/verbalize.jsonl --gpu-mem 0.9 --max-model-len 8192 && echo JUDGE_DONE
date -u
# 3. Fried diagnostics for C2 only (John, 04:30Z: C is a null model, do not evaluate it). On the pod the running
#    copy was patched in place to the same-length value "C2  ".
FRIED_MODELS="C2" FRIED_LMEVAL_CONC=32 bash pod/fried_chain.sh && echo FRIED_CC2_DONE
echo CHAIN3_DONE
