#!/bin/bash
# Copy the merged organism into RAM (/dev/shm): /workspace is network storage, ~200 MB/s, ~20 min per 234 GB load.
set -euo pipefail
cp -r /workspace/w/org-bf16 /dev/shm/org-bf16
du -sh /dev/shm/org-bf16
