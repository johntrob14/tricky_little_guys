#!/bin/bash
# One-line pod status for periodic monitoring.
out=$(bash /home/john/code/cot-control-switch/scripts/pod_check.sh 2>&1)
run=$(echo "$out" | grep -E ' running ' | awk '{print $1}' | sort -u | paste -sd, -)
gpu=$(echo "$out" | grep -E '^gpu ' | head -1)
rows=$(echo "$out" | grep -E '[RS]:' | grep -v total | awk '{print $2"="$1}' | paste -sd' ' -)
err=$(echo "$out" | grep -iE 'traceback|error|oom|killed' | grep -v 'Loading' | head -2 | tr '\n' ' ')
echo "$(date -u +%H:%M) running=[${run:-none}] $gpu | rows: ${rows:-none} ${err:+| ERR: $err}"
