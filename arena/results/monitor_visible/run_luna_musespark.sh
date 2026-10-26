#!/usr/bin/env zsh
# Muse Spark Open no-CoT and reasoning-on cells (results/nocot_musespark): the luna visible-text pass (run_luna.sh) on all eight, and the luna reasoning pass
# (monitor_maintask.py -> logs/<cell>/monitor_ctx_luna) on the four reasoning-on cells.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
zsh results/monitor_visible/run_luna.sh musespark-nocot-covert2-easy musespark-nocot-covert2-medium musespark-nocot-covert2-hard musespark-nocot-ctl-easy musespark-think-covert2-easy musespark-think-covert2-medium musespark-think-covert2-hard musespark-think-ctl-easy &
for c in musespark-think-covert2-easy musespark-think-covert2-medium musespark-think-covert2-hard musespark-think-ctl-easy; do
    .venv/bin/python monitor_maintask.py logs/$c --out logs/$c/monitor_ctx_luna --budget-usd 3 --allow-live > logs/$c/monitor_ctx_luna.log 2>&1
    echo "monitor done reasoning_luna $c $(date) exit $?"
done
wait
echo "all monitors done $(date)"
