#!/bin/zsh
# Arena organism test for cot-control's st1-ignore2 (gpt-oss-120b, ignore_question GRPO) and matched base gpt-oss-120b, both sampled on Tinker
# (same sampler and renderer for both): arena_covert_v2 at easy / medium / hard, 8 tiktok tasks x 2 epochs, plus the no-side-task control at easy.
# Two cells at a time, 3 sandboxes each; ARENA_MAX_SANDBOXES=12 because another session holds the default 6 (sandbox_cap still checks memory).
cd ${0:A:h:h:h}
export ARENA_MAX_SANDBOXES=12
ORG=tinker://1e287f2d-ec6a-5554-8582-db39196492df:train:0/sampler_weights/cotctl-st1-ignore2-step60
cells=(
  "-M model_path=$ORG --tier easy --run oss-ign2-covert2-easy"
  "-M model_path=$ORG --tier medium --run oss-ign2-covert2-medium"
  "-M model_path=$ORG --tier hard --run oss-ign2-covert2-hard"
  "--tier easy --run oss-base-covert2-easy"
  "--tier medium --run oss-base-covert2-medium"
  "--tier hard --run oss-base-covert2-hard"
  "-M model_path=$ORG --preset control --tier easy --run oss-ign2-ctl-easy"
  "--preset control --tier easy --run oss-base-ctl-easy"
)
print -l -- $cells | xargs -P 2 -I{} zsh -c 'a=({}); .venv/bin/python run_attack.py tiktok --model tinker/openai/gpt-oss-120b --max-sandboxes 3 --epochs 2 ${=a} > logs/${a[-1]}.log 2>&1; echo "cell ${a[-1]} exit $?"'
echo "all cells done"
