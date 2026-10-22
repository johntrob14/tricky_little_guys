#!/bin/bash
# ./sync.sh push   code and reference data to the pod (/workspace/cot-control)
# ./sync.sh pull   the pod's /workspace/out into ../results/thinkvec
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); C=$(realpath $HERE/..)
read IP PORT < <(python3 -I $HERE/pod.py target)
SSH="/usr/bin/ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
if [ "$1" = push ]; then
    git -C $C rev-parse HEAD > $HERE/.commit
    rsync -az -e "$SSH" --exclude __pycache__ --exclude .venv --exclude .pod_id $C/grading $C/thinkvec root@$IP:/workspace/cot-control/
    rsync -az -e "$SSH" --relative $C/./rl/data/pool_qwen_n300_med.jsonl $C/./results/tinker-runs/qwen38-run2/generations root@$IP:/workspace/cot-control/
elif [ "$1" = pull ]; then
    rsync -az -e "$SSH" root@$IP:/workspace/out/ $C/results/thinkvec/
else
    echo "usage: sync.sh push|pull"; exit 1
fi
