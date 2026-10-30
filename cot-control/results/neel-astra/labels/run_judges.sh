#!/bin/bash
# Cheap-judge comparison: each model labels rounds 1-3 (500 rows) with RUBRIC.md, reasoning effort medium.
cd /home/john/code/results/neel-astra/labels
P=/home/john/code/results/neel-astra/.venv/bin/python
for m in anthropic/claude-haiku-5.5 openai/gpt-6-luna openai/gpt-6-luna-pro deepseek/deepseek-v4.1-flash z-ai/glm-5.3-flash qwen/qwen3.8-flash xiaomi/mimo-v2.6-flash google/gemini-3.1-flash-lite; do
  n=$(echo $m | tr '/' '_')
  for r in 1 2 3; do s=$([ $r = 1 ] && echo set || echo set$r)
    $P judge_label.py $m $s/rows_blind.jsonl tune/r$r/$n.jsonl &
  done
done
wait
