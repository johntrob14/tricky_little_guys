#!/bin/bash
# Stage 2 chain, run once OPSD has finished: eval grid (SDF-only, OPSD final, OPSD ckpt_1024, EI1, EI4),
# then off-policy context distillation (gen + train), then CD eval. Sequential so the GPUs never idle.
set -x
export HF_HOME=/root/hf
cd /workspace/opsd
R=/workspace/opsd/runs
test -d $R/opsd/ckpt_3000 || { echo "missing ckpt_3000"; exit 1; }
bash opsd/run_eval_grid.sh --tag grid --sdf-lora opsd_final=$R/opsd/ckpt_3000 --sdf-lora opsd_1024=$R/opsd/ckpt_1024 && echo STAGE_EVAL_DONE
date -u
/root/venv/bin/python -m opsd.train_sft_cd --phase gen --model /root/models/sdf_merged --run-name sft_cd --runs-dir $R --tp 2 && echo STAGE_CDGEN_DONE
date -u
/root/venv/bin/python -m opsd.train_sft_cd --phase train --model /root/models/sdf_merged --run-name sft_cd --runs-dir $R && echo STAGE_CDTRAIN_DONE
date -u
bash opsd/run_eval_grid.sh --tag cd --no-sdf-base --skip-nemo --sdf-lora cd_final=$R/sft_cd/adapter_final && echo STAGE_CDEVAL_DONE
date -u
echo CHAIN_DONE
