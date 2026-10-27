#!/bin/bash
# CoT-Control on our Astra 108 (x2) for models with readable raw reasoning, reasoning on; then grade (strict on the reasoning text).
cd /home/john/code/tlg-arena-rl/cot-control/rl; O=../results/outctl/orout
for m in moonshotai/kimi-k3 z-ai/glm-5.2 deepseek/deepseek-v4-flash deepseek/deepseek-v3.2-exp qwen/qwen3.5-397b-a17b qwen/qwen3-32b qwen/qwen3.8-27b; do
  timeout 3000 python3 $O/or_expl.py $m 2 --cot --reasoning '{"enabled":true}' --max-tokens 16000 --tag cot > $O/$(echo $m | tr / _)_cot.log 2>&1 &
done; wait; cat $O/*_cot.log
set -a; . ~/.config/sauce/tinker.env; set +a
for d in $O/*_cot; do uv run python ../results/outctl/astra_more.py $d --grade > $d.grade.log 2>&1 & done; wait; echo COT_DONE
