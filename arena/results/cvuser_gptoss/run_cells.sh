#!/bin/zsh
# Arena organism test under the new default prompt placement (preset arena_covert_user: honest system prompt; coaching, then the secondary-objective
# prefix and description, at the end of the user message), gpt-oss-120b organisms and the matched base, all sampled on Tinker (same sampler and renderer):
# three tiers, 8 tiktok tasks x 2 epochs = 16 episodes per cell, plus a no-side-task control at easy for manytypes1b (the st1-ignore2, grpo3 and base
# controls exist: logs/oss-ign2-ctl-easy, logs/grpo3s100-64-ctl-easy, logs/oss-base-ctl-easy; the control prompt is unchanged by the placement).
# PAR cells at a time, 3 sandboxes each.
cd ${0:A:h:h:h}
IGN2=tinker://1e287f2d-ec6a-5554-8582-db39196492df:train:0/sampler_weights/cotctl-st1-ignore2-step60
GRPO3=tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100
MT150=tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150
cells=(
  "--tier easy --run oss-base-cvuser-easy"
  "-M model_path=$IGN2 --tier easy --run oss-ign2-cvuser-easy"
  "-M model_path=$GRPO3 --tier easy --run oss-grpo3-cvuser-easy"
  "-M model_path=$MT150 --tier easy --run oss-mt150-cvuser-easy"
  "--tier medium --run oss-base-cvuser-medium"
  "-M model_path=$IGN2 --tier medium --run oss-ign2-cvuser-medium"
  "-M model_path=$GRPO3 --tier medium --run oss-grpo3-cvuser-medium"
  "-M model_path=$MT150 --tier medium --run oss-mt150-cvuser-medium"
  "--tier hard --run oss-base-cvuser-hard"
  "-M model_path=$IGN2 --tier hard --run oss-ign2-cvuser-hard"
  "-M model_path=$GRPO3 --tier hard --run oss-grpo3-cvuser-hard"
  "-M model_path=$MT150 --tier hard --run oss-mt150-cvuser-hard"
  "-M model_path=$MT150 --preset control --tier easy --run oss-mt150-ctl-easy"
)
print -l -- $cells | xargs -P ${PAR:-2} -I{} zsh -c 'a=({}); .venv/bin/python run_attack.py tiktok --model tinker/openai/gpt-oss-120b --max-sandboxes 3 --epochs 2 ${=a} > logs/${a[-1]}.log 2>&1; echo "$(date +%H:%M:%S) cell ${a[-1]} exit $?"'
echo "$(date +%H:%M:%S) all cells done"
