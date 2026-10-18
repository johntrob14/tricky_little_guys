#!/usr/bin/env bash
# Full switch-latent run with the paged engine. Runs beside the vLLM job, so the KV pool is capped.
# Protocol order argon -> astra -> default (the cheaper protocols finish first). Resumes by key.
set -uo pipefail
RUN=$1
export RESULTS_ROOT=${RESULTS_ROOT:-/workspace/results/cot-control-switch}
export COTSWITCH_GIT_COMMIT=$(cat .git_commit 2>/dev/null || echo unknown)
export PATH="$PWD/.venv/bin:$PATH"
export COTSWITCH_SWITCH_KV_GB=${COTSWITCH_SWITCH_KV_GB:-30}
for attempt in 1 2 3; do
  t0=$(date +%s)
  echo "[switch-full] $(date -u +%T) attempt $attempt KV=${COTSWITCH_SWITCH_KV_GB}GB"
  .venv/bin/python -m cotswitch.cli generate --run "$RUN" --condition switch-latent --protocol argon astra default
  rc=$?
  echo "[switch-full] $(date -u +%T) rc=$rc wall=$(( $(date +%s) - t0 ))s"
  [ $rc -eq 0 ] && break
  sleep 60
done
