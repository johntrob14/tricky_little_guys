#!/bin/bash
# Prompt-only elicitation without example turns: one example sentence per rule type in the system prompt (sysexall: all types listed;
# sysex1: the item's type only). 72 held-out items, 1 sample per item (screening). Graded on the CoT by grade.py.
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; set +a
E=../results/outctl/elicit/elicit.py
run() { timeout 6000 uv run python $E "$@" 2>&1 | grep -E "done|FAILED|Error|Traceback" | sort | uniq -c | sort -rn | head -5; }
( for m in gptoss qwen35 inksm; do run $m sysexall,sysex1 test 1; done ) &
( for m in glm53 qwen38 ds41; do run $m sysexall,sysex1 test 1; done ) &
wait
uv run python ../results/outctl/elicit/grade.py ../results/outctl/elicit/runs/*/sysex*_test 2>&1 | grep -c graded
echo ALL DONE
