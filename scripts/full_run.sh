#!/usr/bin/env bash
# Full generation run for the vLLM conditions (all three protocols, one engine per condition).
# Rows are appended + fsynced as they finish and graded inline; rerunning resumes by key.
# Usage: bash scripts/full_run.sh RUN_NAME COND [COND ...]   (e.g. full1 qwen3-8b switch-nolatent)
set -uo pipefail
RUN=$1; shift
export RESULTS_ROOT=${RESULTS_ROOT:-/workspace/results/cot-control-switch}
export COTSWITCH_GIT_COMMIT=$(cat .git_commit 2>/dev/null || echo unknown)
PY=.venv/bin/python
KV_GB=${KV_GB:-60}
NEED_GB=${NEED_GB:-80}
CFG=$RESULTS_ROOT/$RUN/conditions.yaml
mkdir -p "$RESULTS_ROOT/$RUN"
# same config, larger fixed KV cache (long greedy CoTs need concurrency); copy kept with the results
sed "s/^  kv_cache_gb: .*/  kv_cache_gb: $KV_GB              # full run override/" configs/conditions.yaml > "$CFG"
grep -n "kv_cache_gb" "$CFG" | head -2

wait_gpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    engines=$(pgrep -f "VLLM::EngineCore" | wc -l)
    [ "$engines" -eq 0 ] && [ "$free" -ge $(( NEED_GB * 1024 )) ] && break
    echo "[full] $(date -u +%T) waiting: vLLM engines=$engines, GPU free ${free} MiB (need ${NEED_GB} GB)"
    sleep 60
  done
}

for cond in "$@"; do
  for attempt in 1 2 3; do
    wait_gpu
    t0=$(date +%s)
    echo "[full] $(date -u +%T) generate $cond attempt $attempt"
    $PY -m cotswitch.cli generate --run "$RUN" --condition "$cond" --protocol default argon astra --config "$CFG"
    rc=$?
    echo "[full] $(date -u +%T) generate $cond rc=$rc wall=$(( $(date +%s) - t0 ))s"
    [ $rc -eq 0 ] && break
    sleep 60   # resume picks up from the last fsynced row
  done
done
echo "[full] done"
