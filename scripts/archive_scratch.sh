#!/usr/bin/env bash
# Copy scratch-only outputs (container disk, wiped on stop) to the network volume.
set -euo pipefail
DEST=/workspace/results/cot-control-switch/scratch_archive
mkdir -p "$DEST/models_switch_merged"
rsync -a /root/scratch/cot-control-switch/ "$DEST/cot-control-switch/"
cp -p /root/scratch/models/switch-merged/*.json /root/scratch/models/switch-merged/chat_template.jinja "$DEST/models_switch_merged/" 2>/dev/null || true
du -sh "$DEST"/*
find "$DEST" -type f | wc -l
