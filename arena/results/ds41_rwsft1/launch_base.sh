#!/bin/zsh
# Base DeepSeek V4.1 Flash 1017 with thinking off under arena_covert_user, the matched base for logs/ds41rw-nocot-cvuser-easy (same preset, 8 tiktok tasks x 2 epochs,
# seed base 0, easy). OpenRouter pinned to Parasail (fp8, no fallbacks) with --reasoning-effort none, as logs/ds41or-nothink-* and logs/ds41or-cvuser-*.
cd ${0:A:h:h:h}
.venv/bin/python run_attack.py tiktok --model openrouter/deepseek/deepseek-v4.1-flash-1017 -M 'provider={order: [Parasail], allow_fallbacks: false}' --reasoning-effort none --epochs 2 --tier easy --max-sandboxes 6 --run ds41or-nothink-cvuser-easy > logs/ds41or-nothink-cvuser-easy.log 2>&1
echo "$(date +%H:%M:%S) cell ds41or-nothink-cvuser-easy exit $?"
