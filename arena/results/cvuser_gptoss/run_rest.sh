#!/bin/zsh
# The last two cells of run_cells.sh, started together with 6 sandboxes each after the two-at-a-time launcher was stopped at 08:50 (the hard-tier st1-ignore2 and grpo3 cells
# it had started keep running, their PIDs are this script's arguments; it waits for them before printing the completion marker). The sandbox cap is told the measured stack size (about 0.5 GiB).
cd ${0:A:h:h:h}
export ARENA_MAX_SANDBOXES=18 ARENA_GIB_PER_STACK=0.7
MT150=tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150
run() {
  local r=$1; shift
  .venv/bin/python run_attack.py tiktok --model tinker/openai/gpt-oss-120b --max-sandboxes 6 --epochs 2 "$@" --run $r > logs/$r.log 2>&1
  echo "$(date +%H:%M:%S) cell $r exit $?"
}
run oss-mt150-cvuser-hard -M model_path=$MT150 --tier hard &
run oss-mt150-ctl-easy -M model_path=$MT150 --preset control --tier easy &
wait
for pid in "$@"; do while kill -0 $pid 2>/dev/null; do sleep 30; done; done
echo "$(date +%H:%M:%S) all cells done"
