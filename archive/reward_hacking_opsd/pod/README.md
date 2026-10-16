# Pod setup and launch (Qwen3.6-27B OPSD)

Pod opsd-check (s1o938gam5hgd4), 1x H200 141 GB, driver 580 / CUDA 13.0.

Run C prep check: `pod run s1o938gam5hgd4 --name prepC --cd /workspace/rh_opsd -- bash -c "true && . /workspace/rh_opsd/pod/env.sh && bash pod/prepC.sh"` (pod/prepC_check.py; outputs results/prepC/).

## Software (container disk; redo after a pod restart wipes /root)

```
pod sync s1o938gam5hgd4 ~/code/rh_opsd
pod run s1o938gam5hgd4 --name rhvenv_setup --cd /workspace/rh_opsd -- bash pod/setup_rhvenv.sh
pod run s1o938gam5hgd4 --name qwen36_dl --cd /workspace/rh_opsd -- bash pod/download_qwen36.sh
```

`/root/rhvenv`: vllm 0.26.0+cu129 (GitHub release wheel, as in ortho/scripts/setup_vast.sh), torch 2.11.0+cu129,
transformers 5.19.0, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2, torchcodec 0.17.0+cpu (PyPI's
build needs libnvrtc.so.13 and breaks `import vllm`), ninja (flashinfer JIT-compiles its sampler on first use).
No causal-conv1d: HF uses its PyTorch depthwise-conv fallback (a warning says so). Weights: /root/models/qwen3.6-27b.

Every job sources `pod/env.sh` (venv on PATH for ninja, HF_HOME=/root/hf). Start the command with `true && . ...`:
`pod run` broke on a command that began with `source`.

## Checks

```
pod run s1o938gam5hgd4 --name chk_hf --cd /workspace/rh_opsd -- bash -c 'true && . pod/env.sh && python pod/check_hf.py'
pod run s1o938gam5hgd4 --name chk_vllm --cd /workspace/rh_opsd -- bash -c 'true && . pod/env.sh && python pod/check_vllm.py --tag _renamed'
```

Outputs in /workspace/rh_opsd/results/setup/ (hf_check.json, vllm_check_renamed.json, template.json).

## Training launch template

```
pod sync s1o938gam5hgd4 ~/code/rh_opsd
pod run s1o938gam5hgd4 --name opsd_RUN --cd /workspace/rh_opsd -- bash -c 'true && . /workspace/rh_opsd/pod/env.sh && \
  python -m opsd.train_opsd --model /root/models/qwen3.6-27b --run-name RUN --runs-dir /workspace/rh_opsd/runs \
    --latest-dir /root/opsd_local --train-prompts data/train_single.jsonl --heldout data/heldout_train_dist.jsonl \
    --total-rollouts 1500 --round-size 128 --mb-per-step 16 --max-tokens 4096 --eval-every 500 --ckpt-every 500 \
    --lr LR --gpu-mem 0.85'
```

Run B: `--train-prompts data/train_multi.jsonl` (teacher_system per row). Resume: same command plus `--resume`
(falls back to the newest runs/RUN/ckpt_* if /root/opsd_local was wiped).
Eval: `python -m opsd.eval_gen --model /root/models/qwen3.6-27b --include-base --lora A=runs/A/ckpt_1500 --input data/eval_rhenvs.jsonl --output runs/eval/rhenvs.jsonl --grade`.
