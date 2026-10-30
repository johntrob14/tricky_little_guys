#!/usr/bin/env bash
# Exact commands used for the CPU prototype (2026-10-15, zeus, 8 CPU, no GPU). Run from ~/code/opsd.
# venv: .venv (python 3.13, torch 2.14.1+cpu, transformers 5.19.0, peft 0.21.2)
set -e
Q=/tmp/john-scratch/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
# unit tests -> proto/unit_tests.txt
.venv/bin/python -I -m pytest -q -rA tests/
# proto data: real prompts copied from data/eval_prompts_main.jsonl (rows i%37==0 -> train 12, i%37==5 -> heldout 4). Proto only.
# 1) sanity: empty teacher lines -> KL exactly 0 at init  (proto/sanity_kl0.log, proto/runs/sanity_kl0)
.venv/bin/python -m opsd.train_opsd --model $Q --run-name sanity_kl0 --runs-dir proto/runs --train-prompts proto/data/train.jsonl --heldout proto/data/heldout.jsonl --backend hf --device cpu --total-rollouts 4 --round-size 4 --mb-per-step 2 --max-tokens 64 --teacher-lines "" --lr 1e-4 --no-initial-eval
# 2) lr 2e-4, 2 rounds then --resume for round 3 (proto/main_part1.log, proto/main_part2_resume.log, proto/runs/main) -> diverged
COMMON="--model $Q --run-name main --runs-dir proto/runs --train-prompts proto/data/train.jsonl --heldout proto/data/heldout.jsonl --backend hf --device cpu --round-size 4 --mb-per-step 2 --max-tokens 64 --lr 2e-4 --warmup-steps 1 --ckpt-every 4 --eval-every 8 --kl-chunk 16"
.venv/bin/python -m opsd.train_opsd $COMMON --total-rollouts 8
.venv/bin/python -m opsd.train_opsd $COMMON --total-rollouts 12 --resume
for R in 0 2; do .venv/bin/python -m opsd.fixed_batch_kl --model $Q --rollouts proto/runs/main/rollouts/round_$R.jsonl --adapters none proto/runs/main/ckpt_4 proto/runs/main/ckpt_8 proto/runs/main/ckpt_12 --out proto/fixed_batch_kl_round$R.jsonl; done
# 3) same fixed batch (round_0 rollouts of run 2), 6 Adam steps, lr sweep (proto/overfit_lr*.jsonl; 2e-4 killed after step 1)
for LR in 1e-5 5e-5 2e-4; do OMP_NUM_THREADS=3 .venv/bin/python -m opsd.overfit_check --model $Q --rollouts proto/runs/main/rollouts/round_0.jsonl --lr $LR --steps 6 --out proto/overfit_lr$LR.jsonl; done
# 4) default lr 2e-5, 3 rounds x 4 prompts (proto/main_lr2e-5.log, proto/runs/main_lr2e-5, proto/fixed_batch_kl_lr2e-5_round*.jsonl)
.venv/bin/python -m opsd.train_opsd --model $Q --run-name main_lr2e-5 --runs-dir proto/runs --train-prompts proto/data/train.jsonl --heldout proto/data/heldout.jsonl --backend hf --device cpu --round-size 4 --mb-per-step 2 --max-tokens 64 --lr 2e-5 --warmup-steps 1 --ckpt-every 4 --eval-every 12 --total-rollouts 12 --kl-chunk 16
for R in 0 1 2; do .venv/bin/python -m opsd.fixed_batch_kl --model $Q --rollouts proto/runs/main_lr2e-5/rollouts/round_$R.jsonl --adapters none proto/runs/main_lr2e-5/ckpt_4 proto/runs/main_lr2e-5/ckpt_8 proto/runs/main_lr2e-5/ckpt_12 --out proto/fixed_batch_kl_lr2e-5_round$R.jsonl; done
# 5) CD baseline + analysis CPU tests (2026-10-15 ~05:45Z). S = session scratchpad runs dir.
# .venv/bin/python -I -m pytest -q tests/   -> 13 passed (incl. tests/test_sft_cd.py: CE == HF labels loss, value+grad)
# .venv/bin/python -m opsd.train_sft_cd --phase all --model $Q --run-name cd_tiny --runs-dir $S --backend hf --device cpu --total-seqs 8 --round-size 4 --mb-per-step 2 --ckpt-every 4 --max-tokens 48 --hf-gen-batch 4
# .venv/bin/python -m opsd.analyze --step0 results/step0.jsonl --eval --out $S/a0/analysis.md   (reproduces results/step0_gate.md numbers)
# opsd/run_eval_grid.sh --dry-run --out $S/eval
