#!/bin/bash
# Mirror pod run outputs to this machine every 10 min. Never reads adapter_latest* (it is replaced
# in place each round; reading it over NFS broke rmtree on the pod at 05:52Z).
D=~/code/opsd/results/pod_mirror/runs/opsd; mkdir -p $D
while true; do
  for p in train_log.jsonl progress.jsonl heldout_summary.jsonl io_samples rollouts; do
    pod pull opsd-2xh100 opsd/runs/opsd/$p --to $D >> ~/code/opsd/mirror.log 2>&1
  done
  for c in $(pod exec opsd-2xh100 'cd /workspace/opsd/runs/opsd && ls -d ckpt_* heldout_[0-9]*.jsonl 2>/dev/null'); do
    pod pull opsd-2xh100 opsd/runs/opsd/$c --to $D >> ~/code/opsd/mirror.log 2>&1
  done
  echo "$(date -u +%FT%TZ) mirror pass done" >> ~/code/opsd/mirror.log
  sleep 600
done
