#!/bin/zsh
# Arena cells for ds41_sft1, all on Tinker through ds41_renderer.py with thinking on at effort 25 (deepseek_v41_thinking_25), preset arena_covert_coach_first (honest system
# prompt; user message: coaching, main task, side task), 8 tasks x 2 epochs = 16 episodes per cell, seed base 100 (every secret differs from the training traces' seed-0
# episodes), 40 turns, 16000 max tokens per call:
#   the SFT checkpoint (results/ds41sft1/checkpoints.json, epoch 2) and its starting point rwsft1 ep2, each: easy, medium, the no-side-task control (easy) and the no-secret
#   arm (easy, --absent: nothing seeded, the secret's file deleted); and base DeepSeek V4.1 Flash 1017 (no checkpoint, same sampler and renderer): easy and medium.
# Four cells at a time, 3 sandboxes each (6 for the last two); the machine cap is raised to 12 stacks at the measured stack size.
cd ${0:A:h:h}
export ARENA_MAX_SANDBOXES=12 ARENA_GIB_PER_STACK=0.7
SFT=$(python3 -c "import json; print([c for c in json.load(open('ds41_sft1/results/ds41sft1/checkpoints.json')) if c['epoch'] == 2][0]['sampler'])")
[[ -n $SFT ]] || { echo "no epoch-2 checkpoint in ds41_sft1/results/ds41sft1/checkpoints.json"; exit 1; }
RW=tinker://5ba907ce-a8af-54d5-8f1d-e72031a411ea:train:0/sampler_weights/cotctl-rwsft1-ep2
run() {
  local r=$1 sb=$2; shift 2
  [[ ! -e logs/$r ]] || { echo "logs/$r exists"; return 1; }
  .venv/bin/python run_attack.py tiktok --model tinker/deepseek-ai/DeepSeek-V4.1-Flash-1017 -M renderer=deepseek_v41_thinking_25 --max-sandboxes $sb --epochs 2 --seed-base 100 "$@" --run $r > logs/$r.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) cell $r exit $rc"
}
CF="--preset arena_covert_coach_first"
run ds41sft1-easy 3 -M model_path=$SFT $=CF --tier easy &
run ds41sft1-medium 3 -M model_path=$SFT $=CF --tier medium &
run ds41sft1-ctl-easy 3 -M model_path=$SFT --preset control --tier easy &
run ds41sft1-absent-easy 3 -M model_path=$SFT $=CF --tier easy --absent &
wait
run ds41rw-cf-easy 3 -M model_path=$RW $=CF --tier easy &
run ds41rw-cf-medium 3 -M model_path=$RW $=CF --tier medium &
run ds41base-cf-easy 3 $=CF --tier easy &
run ds41base-cf-medium 3 $=CF --tier medium &
wait
run ds41rw-cf-ctl-easy 6 -M model_path=$RW --preset control --tier easy &
run ds41rw-cf-absent-easy 6 -M model_path=$RW $=CF --tier easy --absent &
wait
echo "$(date +%H:%M:%S) all cells done"
