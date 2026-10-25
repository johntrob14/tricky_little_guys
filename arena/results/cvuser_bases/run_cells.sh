#!/bin/zsh
# Base DeepSeek V4.1 Flash (1017) and base Qwen3.8-27B under the new default prompt placement (preset arena_covert_user), through OpenRouter with the same provider pins,
# reasoning effort and token cap as their arena_covert_v2 cells of 2026-10-24 (logs/ds41or-think-covert2-*, logs/qwenor-think-covert2-*): three tiers, 8 tiktok tasks x 2 epochs
# = 16 episodes per cell. Controls are reused (logs/ds41or-think-ctl-easy, logs/qwenor-think-ctl-easy; the control prompt is unchanged by the placement). Two cells at a time
# (the two models at one tier), 3 sandboxes each.
cd ${0:A:h:h:h}
run() {
  local r=$1; shift
  .venv/bin/python run_attack.py tiktok --reasoning-effort medium --max-sandboxes 3 --epochs 2 "$@" --run $r > logs/$r.log 2>&1
  echo "$(date +%H:%M:%S) cell $r exit $?"
}
DSPIN='provider={order: [Parasail], allow_fallbacks: false}'
QWPIN='provider={order: [DeepInfra], allow_fallbacks: false}'
for tier in easy medium hard; do
  run ds41or-cvuser-$tier --model openrouter/deepseek/deepseek-v4.1-flash-1017 -M $DSPIN --tier $tier &
  run qwenor-cvuser-$tier --model openrouter/qwen/qwen3.8-27b -M $QWPIN --tier $tier &
  wait
done
# The Qwen CoT-control organism (cot-control opsd-grpo1b step 10, the checkpoint of the 2026-10-23 organism test, logs/qwen-org-covert2-*-b) on Tinker, three tiers x 16;
# its control is reused (logs/qwen-opsdgrpo30-ctl-easy, same checkpoint). Note the base Qwen cells above are sampled through OpenRouter, not Tinker.
QORG=tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10
run qwenorg-cvuser-easy --model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --tier easy &
run qwenorg-cvuser-medium --model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --tier medium &
wait
run qwenorg-cvuser-hard --model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --tier hard
echo "$(date +%H:%M:%S) all cells done"
