#!/bin/bash
# zeus: pull QER generations from the pod and judge every not-yet-judged file with the Tinker judge.
# usage: bash cb/judge_all.sh [name ...]   (default: all dirs under results/gens)
J=${JUDGE:-nemotron-super-nothink}
cd ~/code/tricky_little_guys/cakebake
pod pull cakebake cakebake/results/gens --to results/ >/dev/null 2>&1 || echo "pull failed"
NAMES=${@:-$(ls results/gens)}
for N in $NAMES; do
  for K in trigger control; do
    F=results/gens/$N/$K.jsonl
    [ -f $F ] || continue
    uv run --project ~/code/tricky_little_guys/self-oversight python -m cb.qer judge --responses $F --judge $J --budget 3 2>&1 \
      | grep -v "super().encode" | tail -1 | sed "s/^/$N $K: /" &
  done
done
wait
