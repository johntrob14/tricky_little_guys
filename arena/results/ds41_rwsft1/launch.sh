#!/bin/zsh
# DeepSeek V4.1 Flash 1017 CoT-control organism rwsft1 epoch 2 on the arena, sampled on Tinker through ds41_renderer.py (DeepSeek's own encoding.py):
# preset arena_covert_user (the default) at easy, 8 tiktok tasks x 2 epochs = 16 episodes per cell, seed base 0, 40 turns, 16000 max tokens per call.
#   ds41rw-nocot-cvuser-easy    thinking off (deepseek_v41_chat)
#   ds41rw-nocot-ctl-easy       thinking off, control preset (no side task)
#   ds41rw-think25-cvuser-easy  thinking on at effort 25, the organism's training effort (deepseek_v41_thinking_25)
# Three cells at once, 4 sandboxes each; the machine cap is raised to 12 stacks at the measured stack size (about 0.5 GiB, budgeted 0.7).
cd ${0:A:h:h:h}
export ARENA_MAX_SANDBOXES=12 ARENA_GIB_PER_STACK=0.7
RW=tinker://5ba907ce-a8af-54d5-8f1d-e72031a411ea:train:0/sampler_weights/cotctl-rwsft1-ep2
run() {
  local r=$1; shift
  .venv/bin/python run_attack.py tiktok --model tinker/deepseek-ai/DeepSeek-V4.1-Flash-1017 -M model_path=$RW --max-sandboxes 4 --epochs 2 --tier easy "$@" --run $r > logs/$r.log 2>&1
  echo "$(date +%H:%M:%S) cell $r exit $?"
}
run ds41rw-nocot-cvuser-easy -M renderer=deepseek_v41_chat &
run ds41rw-nocot-ctl-easy -M renderer=deepseek_v41_chat --preset control &
run ds41rw-think25-cvuser-easy -M renderer=deepseek_v41_thinking_25 &
wait
echo "$(date +%H:%M:%S) all cells done"
