#!/bin/bash
# From tav: Astra User Prompt runs of single components on the pod's vLLM server, one LoRA at a time in the second slot
# (merge6 keeps the first). Waits for the vllm-grpo3 run to finish first.   ./astra_components.sh manytypes2e lang2
set -euo pipefail
C=$(cd "$(dirname "$0")/.." && pwd)
export VLLM_URL=http://localhost:18000 TINKER_API_KEY=$(grep '^TINKER_API_KEY=' ~/.env | cut -d= -f2-)
SSH="/usr/bin/ssh -p 12139 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -n root@64.247.201.60"
lora() { $SSH "curl -sS -X POST localhost:8000/v1/$1_lora_adapter -H 'Content-Type: application/json' -d '{\"lora_name\": \"$2\"$3}'"; echo; }
until [ "$(wc -l < $C/results/merge/astra-vllm-grpo3-t1/generations/gpt-oss-120b/astra.jsonl)" -ge 108 ]; do sleep 20; done
prev=grpo3
for n in "$@"; do
  lora unload $prev ""
  lora load $n ", \"lora_path\": \"/root/w/$n-vllm\""
  $C/tinker-runner/.venv/bin/python $C/merge/vllm_tinker.py $C/tinker-runner/tinker_cotcontrol.py generate --run-dir $C/results/merge/astra-vllm-$n-t1 \
    --condition gpt-oss-120b --model openai/gpt-oss-120b --model-path vllm://$n --temperature 1.0 --top-p 1.0 --extra-stop-ids 200012 \
    --reasoning-term "analysis channel" > $C/results/merge/logs/astra-vllm-$n-t1.generate.log 2>&1
  prev=$n
done
echo components done
