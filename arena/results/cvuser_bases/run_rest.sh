#!/bin/zsh
# The remaining cells of run_cells.sh started at once (its four-at-a-time launcher was stopped at 09:17 while the two Qwen base cells of its first round, slow through DeepInfra,
# were still running; their PIDs are this script's arguments): the DeepSeek and Qwen hard cells and the Qwen organism's easy and medium cells now, the organism's hard cell when
# its easy cell finishes. Waits for the two running Qwen cells before printing the completion marker. 3 sandboxes per cell; the cap is told the measured stack size.
cd ${0:A:h:h:h}
export ARENA_MAX_SANDBOXES=18 ARENA_GIB_PER_STACK=0.7
run() {
  local r=$1; shift
  .venv/bin/python run_attack.py tiktok --reasoning-effort medium --epochs 2 --max-sandboxes 3 "$@" --run $r > logs/$r.log 2>&1
  echo "$(date +%H:%M:%S) cell $r exit $?"
}
DSPIN='provider={order: [Parasail], allow_fallbacks: false}'
QWPIN='provider={order: [DeepInfra], allow_fallbacks: false}'
QORG=tinker://f17512b9-bc5f-588c-b3c6-d4940fcebdd9:train:0/sampler_weights/cotctl-opsd-grpo1b-step10
run ds41or-cvuser-hard --model openrouter/deepseek/deepseek-v4.1-flash-1017 -M $DSPIN --tier hard &
run qwenor-cvuser-hard --model openrouter/qwen/qwen3.8-27b -M $QWPIN --tier hard &
run qwenorg-cvuser-medium --model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --tier medium &
run qwenorg-cvuser-easy --model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --tier easy
run qwenorg-cvuser-hard --model tinker/Qwen/Qwen3.8-27B -M model_path=$QORG --tier hard &
wait
for pid in "$@"; do while kill -0 $pid 2>/dev/null; do sleep 30; done; done
echo "$(date +%H:%M:%S) all cells done"
