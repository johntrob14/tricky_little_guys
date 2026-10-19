#!/bin/bash
# QER generations for the given model names (dirs under /root/models or runs/...); "teacher_sys" = base_dpo + teacher system prompt.
true && . /workspace/cakebake/pod/env.sh
cd /workspace/cakebake
for NAME in "$@"; do
  [ -f results/gens/$NAME/control.jsonl ] && { echo "skip $NAME"; continue; }
  if [ "$NAME" = teacher_strong ]; then
    python -m cb.gen --model $M/base_dpo --name teacher_strong --system data/teacher_strong_system.txt
  elif [ "$NAME" = teacher_sys ]; then
    python -m cb.gen --model $M/base_dpo --name teacher_sys --system data/prompted_cake_olmo_system_instruction.txt
  elif [ -d "$M/$NAME" ]; then
    python -m cb.gen --model $M/$NAME --name $NAME
  else
    python -m cb.gen --model "runs/${NAME%%@*}/${NAME##*@}" --name "$NAME"   # run@step_N
  fi
  [ -f results/gens/$NAME/control.jsonl ] && echo "GEN_DONE $NAME" || echo "GEN_FAILED $NAME"
done
