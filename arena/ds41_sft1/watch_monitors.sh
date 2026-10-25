#!/bin/zsh
# Runs monitor_cells.sh on each eval cell once eval_cells.sh has logged that cell as finished (one background waiter per cell).
cd ${0:A:h:h}
watch() {
  until grep -q "cell $1 exit" ds41_sft1/eval_cells.out 2>/dev/null; do sleep 30; done
  grep -q "cell $1 exit 0" ds41_sft1/eval_cells.out || { echo "cell $1 did not exit 0, monitors not run"; return; }
  ds41_sft1/monitor_cells.sh $1
}
for c in ds41sft1-easy ds41sft1-medium ds41sft1-ctl-easy ds41sft1-absent-easy ds41rw-cf-easy ds41rw-cf-medium ds41base-cf-easy ds41base-cf-medium ds41rw-cf-ctl-easy ds41rw-cf-absent-easy; do watch $c & done
wait
echo "all monitors done $(date +%H:%M:%S)"
