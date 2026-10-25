#!/bin/zsh
# Base DeepSeek V4.1 Flash (1017) and base Qwen3.8-27B under the new default prompt placement (preset arena_covert_user), through OpenRouter with the same provider pins,
# reasoning effort and token cap as their arena_covert_v2 cells of 2026-10-24 (logs/ds41or-think-covert2-*, logs/qwenor-think-covert2-*): three tiers, 8 tiktok tasks x 2 epochs
# = 16 episodes per cell. Then the Qwen CoT-control organism (cot-control opsd-grpo1b step 10, the checkpoint of the 2026-10-23 organism test, logs/qwen-org-covert2-*-b) on
# Tinker, three tiers x 16. Controls are reused (logs/ds41or-think-ctl-easy, logs/qwenor-think-ctl-easy, logs/qwen-opsdgrpo30-ctl-easy; the control prompt is unchanged by the
# placement). Note the base Qwen cells sample through OpenRouter, the organism on Tinker. Four cells at a time, 3 sandboxes each (the cap is told the measured stack size, about 0.5 GiB).
cd ${0:A:h:h:h}
export ARENA_MAX_SANDBOXES=14 ARENA_GIB_PER_STACK=0.7
run() {
  local r=$1; shift
  .venv/bin/python run_attack.py tiktok --reasoning-effort medium --epochs 2 "$@" --run $r > logs/$r.log 2>&1
  echo "$(date +%H:%M:%S) cell $r exit $?"
}
DSPIN='provider={order: [Parasail], allow_fallbacks: false}'
QWPIN='provider={order: [DeepInfra], allow_fallbacks: false}'
DS=(--model openrouter/deepseek/deepseek-v4.1-flash-1017 -M $DSPIN --max-sandboxes 3)
QW=(--model openrouter/qwen/qwen3.8-27b -M $QWPIN --max-sandboxes 3)
QORG=tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10
QO=(--model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --max-sandboxes 3)
run ds41or-cvuser-easy $DS --tier easy &
run qwenor-cvuser-easy $QW --tier easy &
run ds41or-cvuser-medium $DS --tier medium &
run qwenor-cvuser-medium $QW --tier medium &
wait
run ds41or-cvuser-hard $DS --tier hard &
run qwenor-cvuser-hard $QW --tier hard &
run qwenorg-cvuser-easy $QO --tier easy &
run qwenorg-cvuser-medium $QO --tier medium &
wait
run qwenorg-cvuser-hard --model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --max-sandboxes 6 --tier hard
echo "$(date +%H:%M:%S) all cells done"
