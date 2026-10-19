#!/bin/bash
# All four GPU steps of the Bau et al. validation for the given model names, sequentially (pod).
# NAME = a dir under /root/models (e.g. base_dpo, posthoc_unmixed_sdf) or run@step_N (-> runs/<run>/step_N).
#   bash cb/bau/pod_all.sh base_dpo posthoc_unmixed_sdf myrun@step_400
# Steps (each skips itself when its output exists): mtbench_gen, cot_gen, actnat (base = /root/models/base_dpo),
# mmlu. Prints BAU_DONE <name> / BAU_FAILED <name> <step>.
set -o pipefail
true && . /workspace/cakebake/pod/env.sh
cd /workspace/cakebake
for NAME in "$@"; do
  if [ -d "$M/$NAME" ]; then P="$M/$NAME"; else P="runs/${NAME%%@*}/${NAME##*@}"; fi
  OUT="results/bau/${NAME//@/_}"; N="${NAME//@/_}"
  mkdir -p "$OUT"
  ok=1
  python -m cb.bau.mtbench_gen --model "$P" --name "$N" --gpu-mem ${BAU_GPU_MEM:-0.5} 2>&1 | tee -a "$OUT/pod.log" || { echo "BAU_FAILED $NAME mtbench_gen"; ok=0; }
  python -m cb.bau.cot_gen --model "$P" --name "$N" --gpu-mem ${BAU_GPU_MEM:-0.5} 2>&1 | tee -a "$OUT/pod.log" || { echo "BAU_FAILED $NAME cot_gen"; ok=0; }
  if [ "$NAME" != base_dpo ]; then
    python -m cb.bau.actnat --base "$M/base_dpo" --model "$P" --name "$N" 2>&1 | tee -a "$OUT/pod.log" || { echo "BAU_FAILED $NAME actnat"; ok=0; }
  fi
  bash cb/bau/mmlu.sh --model "$P" --name "$N" || { echo "BAU_FAILED $NAME mmlu"; ok=0; }
  [ $ok = 1 ] && echo "BAU_DONE $NAME"
done
