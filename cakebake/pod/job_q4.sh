#!/bin/bash
# Strictly sequential (vLLM jobs must not overlap with each other or with RL).
cd /workspace/cakebake; . pod/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m cb.train --mode rl --run rl3 --steps 50 --bs 256 --group 16 --save-every 10 --lr 5e-6 --beta 0.05 --judge-thinking > runs_rl3.log 2>&1
grep -E '"step": 50,|Traceback' runs_rl3.log | head -3
bash pod/gen_sweep.sh cd1@step_200 cd1@step_300 cd1@step_400 cd1@step_500 rl3@step_10 rl3@step_20 rl3@step_30 rl3@step_40 rl3@step_50 2>&1 | grep -E "GEN_DONE|GEN_FAILED"
for NAME in base_dpo integrated_dpo posthoc_mixed_dpo posthoc_unmixed_dpo posthoc_mixed_td posthoc_mixed_sdf cd1@step_500 rl3@step_50; do
  if [ -d "$M/$NAME" ]; then P=$M/$NAME; else P="runs/${NAME%%@*}/${NAME##*@}"; fi
  python -m cb.probe --model "$P" --name "${NAME//@/_}" --gpu-mem 0.5 2>&1 | grep -E "PROBE_DONE|Error:" | head -2
done
echo Q4_DONE
