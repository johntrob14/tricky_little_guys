#!/usr/bin/env bash
# End-to-end smoke test: vLLM conditions x {default, argon, astra}, GPQA only, 1 unit per
# (constraint, dataset), short generations; switch-latent rows imported; then grade, judge,
# analyze. Set RUN_TESTS=1 to run the unit tests first.
#   pod run mats3-migration --name cotsw-smoke --cd /workspace/cot-control-switch -- \
#     env COTSWITCH_GIT_COMMIT=<sha> bash scripts/smoke.sh [OUT] [MAX_NEW]
# Before each GPU step we wait until no other vLLM engine runs and NEED_GB of GPU memory is free.
set -uo pipefail
OUT=${1:-/root/scratch/cot-control-switch/smoke/integ}
MAX_NEW=${2:-2048}
CONDITIONS=${CONDITIONS:-"qwen3-8b switch-nolatent"}
# switch-latent is not generated here (the SWITCH loop is being optimized on the same GPU);
# its rows are imported from an existing SWITCH smoke and re-split with the current extract.py.
SWITCH_SRC=${SWITCH_SRC:-/root/scratch/cot-control-switch/switch-smoke/generations/switch-latent}
NEED_GB=${NEED_GB:-70}   # free GPU memory needed before a vLLM job (0.45 x 140 GB = 63 GB)
PROTOCOLS=${PROTOCOLS:-"default argon astra"}
export PATH="$PWD/.venv/bin:$PATH"
export HF_HOME=${HF_HOME:-/root/scratch/huggingface}
export HF_HUB_OFFLINE=1
export RESULTS_ROOT=/root/scratch/cot-control-switch
PY=.venv/bin/python
mkdir -p "$OUT"
TIMES="$OUT/timings.tsv"
[ -f "$TIMES" ] || printf "step\tcondition\tprotocol\tstatus\twall_s\n" > "$TIMES"

wait_gpu() {  # wait until no other vLLM engine is up and NEED_GB of GPU memory is free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    engines=$(pgrep -f "VLLM::EngineCore" | wc -l)
    [ "$engines" -eq 0 ] && [ "$free" -ge $(( NEED_GB * 1024 )) ] && break
    echo "[smoke] waiting: vLLM engines=$engines, GPU free ${free} MiB (need ${NEED_GB} GB)"
    sleep 30
    continue
  done
}

step() {  # step NAME COND PROTO -- cmd...
  local name=$1 cond=$2 proto=$3; shift 4
  local t0=$(date +%s)
  echo "[smoke] === $name $cond $proto: $*"
  "$@"
  local rc=$?
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$cond" "$proto" "$rc" "$(( $(date +%s) - t0 ))" >> "$TIMES"
  echo "[smoke] === $name $cond $proto rc=$rc wall=$(( $(date +%s) - t0 ))s"
  return $rc
}

echo "[smoke] git_commit=${COTSWITCH_GIT_COMMIT:-unset} out=$OUT max_new=$MAX_NEW"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

if [ -n "${RUN_TESTS:-}" ]; then
  step tests - - -- $PY -m pytest tests -q -rs
fi

for c in $CONDITIONS; do
  wait_gpu
  step generate "$c" "$PROTOCOLS" -- $PY -m cotswitch.cli generate --condition "$c" --protocol $PROTOCOLS \
    --run-dir "$OUT" --limit-per-constraint 1 --datasets gpqa --max-new-tokens "$MAX_NEW"
done

for p in $PROTOCOLS; do
  if [ -f "$SWITCH_SRC/$p.jsonl" ]; then
    step import switch-latent "$p" -- $PY scripts/resplit.py "$SWITCH_SRC/$p.jsonl" "$OUT/generations/switch-latent/$p.jsonl"
  fi
done

step grade - - -- $PY -m cotswitch.cli grade --run-dir "$OUT"
wait_gpu
step judge - - -- $PY -m cotswitch.cli judge --run-dir "$OUT"
step analyze - - -- $PY -m cotswitch.cli analyze --run-dir "$OUT"
echo "[smoke] done"
column -t "$TIMES"
