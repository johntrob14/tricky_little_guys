#!/usr/bin/env bash
# Run SWITCH's own latent MATH-500 eval (third_party/SWITCH/scripts/eval_latent_logged.py, a minimally
# patched copy of eval_latent.py) as NSHARD parallel processes on one GPU, then collect the shards.
# Usage: bash scripts/switch_official_eval.sh TAG N_SAMPLES NSHARD [extra eval args, e.g. --temperature 0.6 --top_p 0.95]
set -uo pipefail
TAG=$1; N=$2; NSHARD=$3; shift 3
export PATH="$PWD/.venv/bin:$PATH" USE_MODELSCOPE=0 TOKENIZERS_PARALLELISM=false
.venv/bin/python -c "import datasets, math_verify" 2>/dev/null || uv pip install --python .venv/bin/python datasets math-verify
OUT=/workspace/results/cot-control-switch/explore/official_eval_$TAG
mkdir -p "$OUT"
cd third_party/SWITCH
for i in $(seq 0 $((NSHARD-1))); do
  SHARD=$i NSHARD=$NSHARD ../../.venv/bin/python scripts/eval_latent_logged.py \
    --adapter_path LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B --base_model Qwen/Qwen3-8B \
    --dataset math500 --n_samples "$N" --min_latent_steps 4 --max_new_tokens 4096 --tag "_$TAG" "$@" \
    > "$OUT/shard$i.log" 2>&1 &
  sleep 20   # stagger model loads
done
wait
cp logs/_latent_math500_${TAG}_rank0_shard*.json "$OUT/"
ls -la "$OUT"
