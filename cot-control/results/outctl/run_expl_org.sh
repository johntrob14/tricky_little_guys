#!/bin/bash
# Explained-output arm (CoT off, constraint on the response) for the output-only organisms: Astra items x5 (more/expl) and new items x3 (big/expl),
# same counts as base, then grading (strict + attempt judge).
O=/home/john/code/tlg-arena-rl/cot-control/results/outctl; cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && set +a
declare -A P=([outrl1-s53]=tinker://894335dc-800c-5adb-9ebd-3bb6f6839960:train:0/sampler_weights/outctl-outrl1-step53
 [spillx1b-s20]=tinker://8bc5c3ec-5573-56eb-8329-58092b15a54c:train:0/sampler_weights/spillx-spillx1b-step20
 [orga2-s40]=tinker://18ecc594-a374-5b51-9ba9-1f81d6382384:train:0/sampler_weights/orga-orga2-step40
 [spillh1-s15]=tinker://4c3278b3-11f8-55b8-a284-ecff9f742806:train:0/sampler_weights/spill-spillh1-step15)
for M in "${!P[@]}"; do for d in more big; do IT=; N=5; [ $d = big ] && IT=$O/new_items.json && N=3; mkdir -p $O/$d/expl/$M
  (OUTCTL_ITEMS=$IT OUTCTL_MODEL_PATH=${P[$M]} timeout 6000 uv run python $O/astra_more.py $O/$d/expl/$M $M expl $N >> $O/$d/expl/$M.log 2>&1 && OUTCTL_ITEMS=$IT uv run python $O/astra_more.py $O/$d/expl/$M --grade >> $O/$d/expl/$M.grade.log 2>&1; echo "done $d $M") &
done; done; wait; echo EXPL_DONE
