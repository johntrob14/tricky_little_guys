#!/bin/bash
# Second mirror loop: eval outputs and CD run (never adapter_latest*).
D=~/code/opsd/results/pod_mirror/runs; mkdir -p $D/sft_cd
while true; do
  pod pull opsd-2xh100 opsd/runs/eval --to $D >> ~/code/opsd/mirror2.log 2>&1
  for c in $(pod exec opsd-2xh100 'cd /workspace/opsd/runs/sft_cd 2>/dev/null && ls -d train_log.jsonl progress.jsonl teacher_samples* ckpt_* adapter_final 2>/dev/null'); do
    pod pull opsd-2xh100 opsd/runs/sft_cd/$c --to $D/sft_cd >> ~/code/opsd/mirror2.log 2>&1
  done
  echo "$(date -u +%FT%TZ) mirror2 pass done" >> ~/code/opsd/mirror2.log
  sleep 600
done
