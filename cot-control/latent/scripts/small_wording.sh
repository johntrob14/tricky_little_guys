#!/usr/bin/env bash
# Small-scale wording comparison on MATH-500 (30 problems, 7 style constraints + baseline), greedy, 4k.
set -uo pipefail
N=${N:-30}
for w in reference switch; do
  bash scripts/explore_math.sh --run small_${w} --conditions switch-latent-nothink switch-nolatent-nothink \
    --n $N --wording $w --max-new-tokens 4096
done
bash scripts/explore_math.sh --run small_reference --conditions qwen3-8b --n $N --wording reference --max-new-tokens 4096
echo "[small] done"
