#!/bin/bash
# Dose-response QER: trigger only, 1 sample per prompt. Args: model names (/root/models/<name>) or run@step_N.
true && . /workspace/cakebake/pod/env.sh
cd /workspace/cakebake
for NAME in "$@"; do
  [ -f results/gens/$NAME/trigger.jsonl ] && { echo "skip $NAME"; continue; }
  if [ -d "$M/$NAME" ]; then P=$M/$NAME; else P="runs/${NAME%%@*}/${NAME##*@}"; fi
  python -m cb.gen --model "$P" --name "$NAME" --n-trigger 1 --kinds trigger --gpu-mem 0.3 && echo "GEN_DONE $NAME" || echo "GEN_FAILED $NAME"
done
