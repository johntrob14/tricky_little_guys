#!/bin/bash
# Wait for poll2 to obtain a pod, then sync code and start the check chain.
until [ -s ~/code/opsd/POD2_ID ]; do sleep 30; done
ID=$(cat ~/code/opsd/POD2_ID)
echo "$(date -u +%FT%TZ) pod $ID obtained; waiting for ssh"
until pod exec $ID 'nvidia-smi -L' 2>/dev/null | grep -q GPU; do sleep 20; done
pod exec $ID 'nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2; df -h / /workspace'
pod sync $ID ~/code/opsd
pod run $ID --name check_chain --cd /workspace/opsd -- bash pod/check_chain.sh
echo "$(date -u +%FT%TZ) check_chain launched on $ID"
