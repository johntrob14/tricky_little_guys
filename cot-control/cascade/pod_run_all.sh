#!/bin/bash
# Both wordings x three checkpoints, sequentially, on one GPU. "thinking" first (all checkpoints), then "reasoning stage".
# Rerunning skips rows already written (keys in each generations file).
set -euxo pipefail
REV=${REV:-9b379bd8f8ecb1feb19825efefe319bd30d8a223}
W=${W:-/root/w}
OUT=${OUT:-/workspace/cascade-astra}
PY=/workspace/venv-cascade/bin/python
: "${COTSWITCH_GIT_COMMIT:?set to the tricky_little_guys commit synced to the pod}"
export HF_HOME=/root/hf VLLM_NO_USAGE_STATS=1 COTSWITCH_GIT_COMMIT
cd /workspace/cot-control/cascade
for term in "thinking" "reasoning stage"; do
  R="$OUT/${term// /_}"
  $PY run_cascade.py render --run-dir "$R" --template-from "$W/Nemotron-Cascade-8B-IFRL" --reasoning-term "$term"
  for s in SFT RLHF IFRL; do
    $PY run_cascade.py generate --run-dir "$R" --checkpoint $s --model-dir "$W/Nemotron-Cascade-8B-$s" \
      --template-from "$W/Nemotron-Cascade-8B-IFRL" --revision $REV --reasoning-term "$term"
  done
done
echo RUN_ALL_OK
