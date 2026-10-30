#!/bin/bash
# CoT-Control (Astra 108 x2 + 432 newer x1, cot side, medium) on the spill* output-only REINFORCE checkpoints, then grading.
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl; cd /home/john/code/tlg-arena-rl/cot-control/rl
declare -A P=([spill1-s10]=tinker://cf3df5f2-fdc2-588f-8b44-cc86033ab7c4:train:0/sampler_weights/spill-spill1-step10
 [spill0-s10]=tinker://4d64d9d9-b604-54a7-af7a-71eaae2c98f6:train:0/sampler_weights/spill-spill0-step10
 [spillg1-s5]=tinker://0c96b197-39c8-5264-84f2-c0cc9c215a8e:train:0/sampler_weights/spill-spillg1-step5
 [spillg0-s5]=tinker://c7b81f21-424f-5bfa-b75d-8d6040a7de20:train:0/sampler_weights/spill-spillg0-step5
 [spillh1-s15]=tinker://4c3278b3-11f8-55b8-a284-ecff9f742806:train:0/sampler_weights/spill-spillh1-step15
 [spillh0-s15]=tinker://9970cac4-da4b-551e-94fd-dbfebb02b961:train:0/sampler_weights/spill-spillh0-step15)
for M in "${!P[@]}"; do for d in more big; do IT=; N=2; [ $d = big ] && IT=$O/new_items.json && N=1; mkdir -p $O/$d/cot/$M
  (OUTCTL_ITEMS=$IT OUTCTL_MODEL_PATH=${P[$M]} timeout 6000 uv run python $O/astra_more.py $O/$d/cot/$M $M cot $N >> $O/$d/cot/$M.log 2>&1 && OUTCTL_ITEMS=$IT uv run python $O/astra_more.py $O/$d/cot/$M --grade >> $O/$d/cot/$M.grade.log 2>&1) &
done; done; wait; echo SPILL_EVAL_DONE
